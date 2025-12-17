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

// #include "kernel/vfxmacc.c"

#define AVL       (32)
#define IDX_WIDTH (2)

float    *a;
uint32_t  *nm_index;
float  *golden;
float    *res;

static inline int fp32_check(float *a, float *b) {
  const float threshold = 0.001f;

  // Absolute value
  float comp = 0.0f;
  for (uint32_t i=0; i< AVL; i++){
    comp += b[i] - a[i];
  }
  if (comp < 0)
    comp = -comp;

  return comp > threshold;
}

int main() {
  const unsigned int num_cores = snrt_cluster_core_num();
  const unsigned int cid = snrt_cluster_core_idx();

  // Allocate the matrices
  if (cid == 0) {
    a        = (float *)snrt_l1alloc(AVL * sizeof(float)    );
    // nm_index = (__fp16 *)snrt_l1alloc(AVL * IDX_WIDTH          ); // in 2:4, 2-bit per index
    nm_index = (uint32_t *)snrt_l1alloc(AVL * sizeof(uint32_t)    );
    res      = (float *)snrt_l1alloc(AVL * 2 * sizeof(float)); // we will have a 50% scatter rate
    golden   = (float *)snrt_l1alloc(AVL * 2 * sizeof(float)); 
  }

  // Initialize the matrices
  if (cid == 0) {
    // init output vector
    for (uint32_t i=0; i<2*AVL; i++){
      res[i]       = 0.0f;
      golden[i]    = 0.0f;
      nm_index[i]  = 0;
    }
    // init input vector
    for (uint32_t i=0; i<AVL; i++){
      a[i]          = 1.0f;
      golden[2*i+1] = 2.0f;
    }
    // init index, we set only 2'b01 and 2'b11 for testing
    // for (uint32_t i=0; i<AVL/(sizeof(__fp16)/IDX_WIDTH); i++){
    //   nm_index[i] = 0xdddddddd;
    // }
    for (uint32_t i=0; i<AVL; i++){
      nm_index[i] = 0xdddddddd;
    }
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  // test the move instruction
  if (cid == 0){
    unsigned int vl;
    unsigned int avl = AVL;

    float b = 2.0f;
    // // v16 is in VTL to gather and scatter
    // asm volatile("csrrwi x0, 0x7c3, 16");
    // // set idx width to 2-bit
    // asm volatile("csrrwi x0, 0x7c4, 1");
    // // set blk size to 4
    // asm volatile("csrrwi x0, 0x7c5, 1");
    // // set sparse ratio to 50%
    // asm volatile("csrrwi x0, 0x7c6, 1");
    asm volatile(
      "csrrwi x0, 0x7c3, 16\n"
      "csrrwi x0, 0x7c4, 1\n"
      "csrrwi x0, 0x7c5, 1\n"
      "csrrwi x0, 0x7c6, 1\n"
      ::: "memory"
    );

    do {
      // Set the vl
      asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(vl) : "r"(avl));

      // load index 
      asm volatile("vlx32.v v16,   (%0)" ::"r"(nm_index));
      // load input vector 
      asm volatile("vle32.v v8,    (%0)" ::"r"(a));
      // index-macc
      asm volatile("vfxmacc.vf v16, %0, v8" ::"f"(b));
      // Bump pointers
      avl -= vl;
    } while (avl > 0);

  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  if (cid == 0){
    unsigned int vl;
    unsigned int avl = AVL * 2;
    do {
      // Set the vl
      asm volatile("vsetvli %0, %1, e32, m2, ta, ma" : "=r"(vl) : "r"(avl));
      // in VTL we put b
      asm volatile("vse32.v v16,   (%0)" ::"r"(res));
      // Bump pointers
      avl -= vl;
    } while (avl > 0);

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
