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

#define AVL       (AVL_NNZ)

float     *a;
uint32_t  *nm_index;
float     *golden;
float     *res;
float     *zeros; 

static inline int fp32_check(float *a, float *b) {
  const float threshold = 0.001f;

  // Absolute value
  float comp = 0.0f;
  for (uint32_t i=0; i< ORIG_LEN; i++){
    comp += b[i] - a[i];
  }
  if (comp < 0)
    comp = -comp;

  printf("COMP - %8x \n", *(int32_t *)&comp);

  return comp > threshold;
}

int main() {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  // Allocate the matrices
  if (cid == 0) {
    a        = (float *)   snrt_l1alloc(AVL_NNZ        * sizeof(float)    );
    nm_index = (uint32_t *)snrt_l1alloc(NM_INDEX_WORDS * sizeof(uint32_t) );
    res      = (float *)   snrt_l1alloc(ORIG_LEN       * sizeof(float)    );
    golden   = (float *)   snrt_l1alloc(ORIG_LEN       * sizeof(float)    ); 
    zeros    = (float *)   snrt_l1alloc(ORIG_LEN       * sizeof(float)    ); 
  }

  // Initialize the matrices
  if (cid == 0) {
    for (uint32_t i = 0; i < ORIG_LEN; i++) {res[i] = 0.0f; zeros[i]=0.0f;}
    snrt_dma_start_1d(a,        a_dram,        AVL_NNZ        * sizeof(float));
    snrt_dma_start_1d(nm_index, nm_index_dram, NM_INDEX_WORDS * sizeof(uint32_t));
    snrt_dma_start_1d(golden,   golden_dram,   ORIG_LEN       * sizeof(float));
    snrt_dma_wait_all();
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // test the move instruction
  if (cid == 0){
    unsigned int vl, svl;
    unsigned int avl  = AVL;
    unsigned int savl = ORIG_LEN;

    float b = VFXMACC_B;

    asm volatile(
      "csrrwi x0, 0x7c3, 16\n" // v16 is in VTL to gather and scatter
      "csrrwi x0, 0x7c4, 1\n"  // set idx width to 2-bit
      "csrrwi x0, 0x7c5, 1\n"  // set blk size to 4
      "csrrwi x0, 0x7c6, 1\n"  // set sparse ratio to 50%
      ::: "memory"
    );

    // pointers
    float *    _res      = res;
    uint32_t * _nm_index = nm_index;
    float * _a = a;

    do {
      // Set the vl
      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(avl));
      // load index 
      asm volatile("vlx32.v v16,   (%0)" ::"r"(_nm_index));
      // load input vector 
      asm volatile("vle32.v v8,    (%0)" ::"r"(_a));
      // index-macc
      asm volatile("vfxmacc.vf v16, %0, v8" ::"f"(b));
      // move out
      savl = vl * (M/N);
      do {
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(svl) : "r"(savl));
        asm volatile("vse32.v v16,   (%0)" ::"r"(_res) );
        asm volatile("vle32.v v16,   (%0)" ::"r"(zeros));
        // Bump pointers
        savl -= svl;
        _res += svl;
      } while (savl > 0);

      // Bump pointers
      avl -= vl;
      _a  += vl;
      _nm_index += vl / (sizeof(uint32_t) * 8/IDX_WIDTH);
    } while (avl > 0);

  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  if (cid == 0){
    // unsigned int svl;
    // unsigned int savl = ORIG_LEN;
    // do {
    //   // Set the vl
    //   asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(svl) : "r"(savl));
    //   // in VTL we put b
    //   asm volatile("vse32.v v16,   (%0)" ::"r"(res));
    //   // Bump pointers
    //   savl -= svl;
    // } while (savl > 0);

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
    printf("\n----- (%d) vfxmacc -----\n", AVL);
    printf("DONE \n");
  }

  // Wait for core 0 to finish displaying results
  snrt_cluster_hw_barrier();

  return 0;
}
