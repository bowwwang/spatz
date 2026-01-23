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

#define _BASELINE_KERNEL    (0)
#define _INTERLEAVED_KERNEL (1)
#define _IMPROVED_KERNEL    (2)

#define _SEL_KERENL         (_IMPROVED_KERNEL)

float     *a;          // activation vector
float     *w;          // compact weight matrix 
uint32_t  *nm_index;   // index matrix
float     *golden;     // expected output vector
float     *res;        // computation results
float     *zeros;      // TEMP: to init the VTL memory unit
float     *tmp0, *tmp1; 

static inline int fp32_check(float *a, float *b) {
  const float threshold = 0.001f;

  // Absolute value
  float comp = 0.0f;
  for (uint32_t i=0; i<P/2; i++){
    comp = b[i] - a[i];
    if (comp < 0) comp = -comp;
    if (comp > threshold) printf("[%d] EXP - %8x, GOT - %8x \n", i, *(int32_t *)&a[i], *(int32_t *)&b[i]);
  }
  for (uint32_t i=0; i<P/2; i++){
    comp += b[i] - a[i];
  }
  if (comp < 0)
    comp = -comp;

  printf("COMP - %8x \n", *(int32_t *)&comp);

  return comp > threshold;
}

// Runtime helper function to config vreg mapping
// Accepted values: [0:31]

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

int main() {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  // Allocate the matrices
  if (cid == 0) {
    a        = (float *)   snrt_l1alloc(N              * sizeof(float)    );
    w        = (float *)   snrt_l1alloc(N * P_W        * sizeof(float)    );
    nm_index = (uint32_t *)snrt_l1alloc(NM_INDEX_WORDS * sizeof(uint32_t) ); // NM_INDEX_WORDS = N * P_W * IDX_WIDTH / 8 (bytes)
    res      = (float *)   snrt_l1alloc(P              * sizeof(float)    );
    golden   = (float *)   snrt_l1alloc(P              * sizeof(float)    ); 
    zeros    = (float *)   snrt_l1alloc(P              * sizeof(float)    ); 
  }

  // Initialize the matrices
  if (cid == 0) {
    for (uint32_t i = 0; i < P; i++) {res[i] = 0.0f; zeros[i]=0.0f;}
    snrt_dma_start_1d(a,        a_dram,        N              * sizeof(float));
    snrt_dma_start_1d(w,        w_dram,        N * P_W        * sizeof(float));
    snrt_dma_start_1d(nm_index, nm_index_dram, NM_INDEX_WORDS * sizeof(uint32_t));
    snrt_dma_start_1d(golden,   golden_dram,   P              * sizeof(float));
    snrt_dma_wait_all();
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // test the move instruction
  if (cid == 0){
    unsigned int vl, svl, savl;
    unsigned int avl  = P_W;

    vtl_cfg(16, 20, 24, 28);

    asm volatile(
      // "csrrwi x0, 0x7c3, 16\n" // v16 is in VTL to gather and scatter
      "csrrwi x0, 0x7c4, 1\n"  // set idx width to 2-bit
      "csrrwi x0, 0x7c5, 1\n"  // set blk size to 4
      "csrrwi x0, 0x7c6, 1\n"  // set sparse ratio to 50%
      ::: "memory"
    );

    // pointers
    float    * _w        = w;
    uint32_t * _nm_index = nm_index;
    float    * _res      = res;

    do {
      // Outer loop, P dimension
      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(avl));

      // pointers for inner loop
      float    * _a        = a;
      float    * __w       = _w;
      uint32_t *__nm_index = _nm_index;

      ////////////////////////////////////
      // BASELINE kernel implementation //
      ////////////////////////////////////
      #if _SEL_KERENL == _BASELINE_KERNEL
      for (uint32_t n = 0; n < N; n++){
        // load scalar activation
        asm volatile("flw      ft0,  (%0)" ::"r"(_a));
        // load index 
        asm volatile("vlx32.v v16,   (%0)" ::"r"(__nm_index));
        // load compact weight vector 
        asm volatile("vle32.v v8,    (%0)" ::"r"(__w));
        // index-macc
        asm volatile("vfxmacc.vf v16, ft0, v8" ::);
        // bump the inner loop pointers
        _a         += 1;
        __w        += P_W;
        __nm_index += NM_INDEX_ROW_WORDS;
      }
      #endif 

      //////////////////////////////////////
      // INTERLEAVE kernel implementation //
      //////////////////////////////////////
      #if _SEL_KERENL == _INTERLEAVED_KERNEL
      for (uint32_t n = 0; n < N; n++){
        // load scalar activation
        asm volatile("flw      ft0,  (%0)" ::"r"(_a));
        _a         += 1;
        // load index 
        asm volatile("vlx32.v v16,   (%0)" ::"r"(__nm_index));
        // load compact weight vector 
        asm volatile("vle32.v v8,    (%0)" ::"r"(__w));
        __nm_index += NM_INDEX_ROW_WORDS;
        __w        += P_W;
        // index-macc
        asm volatile("vfxmacc.vf v16, ft0, v8" ::);
      }
      #endif

      ////////////////////////////////////
      // IMPROVED_KERNEL implementation //
      ////////////////////////////////////
      #if _SEL_KERENL == _IMPROVED_KERNEL
      for (uint32_t n = 0; n < N-1; n+=2){
        // load scalar activation
        asm volatile("flw      ft0,  (%0)" ::"r"(_a));
        // load index 
        asm volatile("vlx32.v v16,   (%0)" ::"r"(__nm_index));
        // load compact weight vector 
        asm volatile("vle32.v v8,    (%0)" ::"r"(__w));
        _a         += 1;
        __w        += P_W;
        // index-macc
        asm volatile("vfxmacc.vf v16, ft0, v8" ::);
        __nm_index += NM_INDEX_ROW_WORDS;

        // load scalar activation
        asm volatile("flw      ft1,  (%0)" ::"r"(_a));
        // load index 
        asm volatile("vlx32.v v16,   (%0)" ::"r"(__nm_index));
        // load compact weight vector 
        asm volatile("vle32.v v12,    (%0)" ::"r"(__w));
        _a         += 1;
        __w        += P_W;
        // index-macc
        asm volatile("vfxmacc.vf v16, ft1, v12" ::);
        __nm_index += NM_INDEX_ROW_WORDS;
      }
      #endif 

      // move out
      savl = vl * (M_SPARSE/N_SPARSE);
      do {
        // we use m8 here because the effective LMUL is duplicated with 2:4 format
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(svl) : "r"(savl));
        asm volatile("vse32.v v16,   (%0)" ::"r"(_res) );
        // zero out the VTL memory unit, vmv is not available
        asm volatile("vle32.v v16,   (%0)" ::"r"(zeros));
        // Bump pointers
        savl -= svl;
        _res += svl;
      } while (savl > 0);

      // Bump outer loop pointers
      avl       -= vl;
      _w        += vl;
      _nm_index += vl / (sizeof(uint32_t) * 8/IDX_WIDTH);
    } while (avl > 0);

  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  if (cid == 0){
    // check
    if (fp32_check(golden, res)) {
      printf("WRONG! Expect %8x (golen[1]), Got %8x (res[1]).\n", *(int32_t *)&golden[1], *(int32_t *)&res[1]);
    } else {
      printf("CORRECT! \n");
    }
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // End dump
  // if (cid == 0)
  //   stop_kernel();

  // Check and display results
  if (cid == 0) {
    printf("\n----- (%dx%d) SpMV - vfx -----\n", N, P);
    printf("DONE \n");
  }

  // Wait for core 0 to finish displaying results
  snrt_cluster_hw_barrier();

  return 0;
}
