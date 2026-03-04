// Copyright 2025 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Bowen Wang <bowwang@iis.ee.ethz.ch>

#include <benchmark.h>
#include <debug.h>
#include <snrt.h>
#include <stdio.h>

#include "data/data_vfxmacc.h"

//////////////////////
// Kernel Selection //
//////////////////////
#define _BASELINE_KERNEL    (0)
#define _IMPROVED_KERNEL    (1)
#define _ADAPTED_KERNEL     (2)

#define _SEL_KERENL         (_ADAPTED_KERNEL)

float     *a;          // activation matrix
float     *w;          // compact weight matrix 
uint32_t  *nm_index;   // index matrix
float     *golden;     // expected output vector
float     *res;        // computation results
float     *zeros;      // TEMP: to init the VTL memory unit
float     *tmp0, *tmp1; 

///////////
// Check //
///////////
static inline int fp32_check(float *a, float *b) {
  const float threshold = 0.001f;

  // Absolute value
  float comp     = 0.0f;
  float comp_acc = 0.0f;
  for (uint32_t m=0; m<M; m++){
    for (uint32_t p=0; p<P; p++){
      comp = b[m*P + p] - a[m*P + p];
      if (comp < 0) comp = -comp;
      if (comp > threshold) {
        comp_acc += comp;
        printf("[%d, %d] EXP - %8x, GOT - %8x \n", m, p, *(int32_t *)&a[m*P + p], *(int32_t *)&b[m*P + p]);
      }
    }
  }
  printf("COMP - %8x \n", *(int32_t *)&comp_acc);
  return comp_acc > threshold;
}

////////////////////////////////////////////////////
// Runtime helper function to config vreg mapping //
// Accepted values: [0:31]                        //
////////////////////////////////////////////////////

static inline uint32_t bit_if_valid(uint32_t vr) {
  return (vr < 32) ? (1u << vr) : 0u;
}

static inline void vtl_cfg (uint32_t vr0, uint32_t vr1, uint32_t vr2, uint32_t vr3) {
  uint32_t vreg_bitmap =
      bit_if_valid(vr0) |
      bit_if_valid(vr1) |
      bit_if_valid(vr2) |
      bit_if_valid(vr3);

  asm volatile(
      "csrrw  x0, 0x7c3, %0\n" // v16 is in VTL to gather and scatter
      :
      : "r" (vreg_bitmap)
      : "memory"
    );
}

//////////////////////////
// Sparse Format Config //
//////////////////////////

#define IDX_ENC   ((IDX_WIDTH == 1) ? 0 : (IDX_WIDTH == 2) ? 1 : (IDX_WIDTH == 4) ? 2 : 3)
#define BLK_ENC   ((M_SPARSE  == 1) ? 0 : (M_SPARSE  == 2) ? 1 : (M_SPARSE  == 4) ? 2 : 3)
#define RATIO_ENC (((M_SPARSE / N_SPARSE) == 2) ? 1 : 2)

static inline void sparse_fmt_cfg(void) {
  asm volatile ("csrwi 0x7c4, %0" :: "i"(IDX_ENC)   : "memory");
  asm volatile ("csrwi 0x7c5, %0" :: "i"(BLK_ENC)   : "memory");
  asm volatile ("csrwi 0x7c6, %0" :: "i"(RATIO_ENC) : "memory");
}

//////////
// main //
//////////

int main() {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  // Allocate the matrices
  if (cid == 0) {
    a        = (float *)   snrt_l1alloc(M * N          * sizeof(float)    );
    w        = (float *)   snrt_l1alloc(N * P_W        * sizeof(float)    );
    nm_index = (uint32_t *)snrt_l1alloc(NM_INDEX_WORDS * sizeof(uint32_t) ); // NM_INDEX_WORDS = N * P_W * IDX_WIDTH / 8 (bytes)
    res      = (float *)   snrt_l1alloc(M * P          * sizeof(float)    );
    golden   = (float *)   snrt_l1alloc(M * P          * sizeof(float)    ); 
    zeros    = (float *)   snrt_l1alloc(M * P          * sizeof(float)    ); 
  }

  // Initialize the matrices
  if (cid == 0) {
    for (uint32_t i = 0; i < M*P; i++) {res[i] = 0.0f; zeros[i]=0.0f;}
    snrt_dma_start_1d(a,        a_dram,        M * N          * sizeof(float));
    snrt_dma_start_1d(w,        w_dram,        N * P_W        * sizeof(float));
    snrt_dma_start_1d(nm_index, nm_index_dram, NM_INDEX_WORDS * sizeof(uint32_t));
    snrt_dma_start_1d(golden,   golden_dram,   M * P          * sizeof(float));
    snrt_dma_wait_all();
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // test the move instruction
  if (cid == 0){
    // unsigned int vl, svl, savl;
    // unsigned int avl  = P_W;

    vtl_cfg(16, 18, 32, 32);
    sparse_fmt_cfg();

    unsigned int p = 0;
    while (p < P_W) {
      size_t gvl;
      #if (M_SPARSE/N_SPARSE == 2)
      asm volatile("vsetvli %[gvl], %[vl], e32, m4, ta, ma"
                 : [gvl] "=r"(gvl)
                 : [vl] "r"(P_W - p));
      #else
      asm volatile("vsetvli %[gvl], %[vl], e32, m2, ta, ma"
                 : [gvl] "=r"(gvl)
                 : [vl] "r"(P_W - p));
      #endif

      const float *w_   = w + p;
      const uint32_t *idx_ = nm_index + p/(sizeof(uint32_t) * 8/IDX_WIDTH);
      #if (M_SPARSE/N_SPARSE == 2)
      // 2:4
      float *res_ = res + (p<<1);
      #else
      // 1:4
      float *res_ = res + (p<<2);
      #endif

      for (unsigned m=0; m<M; m+=2){
        const float *a_ = a + m * N;
        const float *a__ = a_;

        asm volatile("vlx32.v v8, (%0)" ::"r"(idx_));
        const uint32_t *idx__ = idx_ + NM_INDEX_ROW_WORDS;

        asm volatile("vle32.v v8, (%0);" ::"r"(w_));
        const float *w__ = w_ + P_W;

        float *res__ = res_ + m * P;

        float t0, t1;

        t0 = *a__;
        a__ += N;
        t1 = *a__;

        unsigned int n = 0;

        while (n<N) {
          a__ = a_ + ++n;

          asm volatile("vlx32.v v4, (%0)" ::"r"(idx__));
          idx__ += NM_INDEX_ROW_WORDS;

          asm volatile("vle32.v v4, (%0);" ::"r"(w__));
          w__ += P_W;

          if (n==1) {
            asm volatile("vfxmacc.vf v16, %0, v8" ::"f"(t0));
            t0 = *a__;
            a__ += N;
            asm volatile("vfxmacc.vf v18, %0, v8" ::"f"(t1));
            t1 = *a__;
          } else {
            asm volatile("vfxmacc.vf v16, %0, v8" ::"f"(t0));
            t0 = *a__;
            a__ += N;
            asm volatile("vfxmacc.vf v18, %0, v8" ::"f"(t1));
            t1 = *a__;
          }

          a__ = a_ + ++n;

          if (n == N)
            break;

          asm volatile("vlx32.v v8, (%0)" ::"r"(idx__));
          idx__ += NM_INDEX_ROW_WORDS;
          asm volatile("vle32.v v8, (%0);" ::"r"(w__));
          w__ += P_W;

          asm volatile("vfxmacc.vf v16, %0, v4" ::"f"(t0));
          t0 = *a__;
          a__ += N;
          asm volatile("vfxmacc.vf v18, %0, v4" ::"f"(t1));
          t1 = *a__;
        }
        asm volatile("vfxmacc.vf v16, %0, v4" ::"f"(t0));
        asm volatile("vse32.v v16, (%0);" ::"r"(res__));
        asm volatile("vle32.v v16, (%0)"  ::"r"(zeros));
        res__ += P;
        asm volatile("vfxmacc.vf v18, %0, v4" ::"f"(t1));
        asm volatile("vse32.v v18, (%0);" ::"r"(res__));
        asm volatile("vle32.v v18, (%0)"  ::"r"(zeros));
      }

      p += gvl;
    }

    
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  if (cid == 0){
    // check
    if (fp32_check(golden, res)) {
      printf("WRONG!   \n");
    } else {
      printf("CORRECT! \n");
    }
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // Check and display results
  if (cid == 0) {
    printf("\n----- (%dx%d) SpMM - vfx -----\n", N, P);
    printf("DONE \n");
  }

  // Wait for core 0 to finish displaying results
  snrt_cluster_hw_barrier();

  return 0;
}
