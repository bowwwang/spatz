// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Bowen Wang <bowwang@iis.ee.ethz.ch>

#include <benchmark.h>
#include <debug.h>
#include <snrt.h>
#include <stdio.h>

#include "data/layer.h"
#include "data/data_spmv.h"
#include "kernel/sp-SpMV.c"

static float    *a;
static float    *w;
static uint32_t *nm_index;
static float    *res;
static float    *golden;

// fp32 result verification with tolerance.
static int fp32_check(const float *ref, const float *got, uint32_t P) {
  const float threshold = 0.001f;
  float comp_acc = 0.0f;
  for (uint32_t i = 0; i < P; i++) {
    float d = got[i] - ref[i];
    if (d < 0) d = -d;
    if (d > threshold) {
      printf("[%u] EXP - %8x, GOT - %8x\n",
             i, *(int32_t *)&ref[i], *(int32_t *)&got[i]);
      comp_acc += d;
    }
  }
  return comp_acc > threshold;
}

int main(void) {
  const unsigned int cid = snrt_cluster_core_idx();

  // Allocate in L1
  if (cid == 0) {
    a        = (float    *)snrt_l1alloc(spmv_l.N                  * sizeof(float));
    w        = (float    *)snrt_l1alloc(spmv_l.N * spmv_l.P_W     * sizeof(float));
    nm_index = (uint32_t *)snrt_l1alloc(spmv_l.NM_INDEX_WORDS     * sizeof(uint32_t));
    res      = (float    *)snrt_l1alloc(spmv_l.P                  * sizeof(float));
    golden   = (float    *)snrt_l1alloc(spmv_l.P                  * sizeof(float));
  }

  // DMA the DRAM data into L1
  if (cid == 0) {
    snrt_dma_start_1d(a,        spmv_a_dram,
                      spmv_l.N                  * sizeof(float));
    snrt_dma_start_1d(w,        spmv_w_dram,
                      spmv_l.N * spmv_l.P_W     * sizeof(float));
    snrt_dma_start_1d(nm_index, spmv_nm_index_dram,
                      spmv_l.NM_INDEX_WORDS     * sizeof(uint32_t));
    snrt_dma_start_1d(golden,   spmv_golden_dram,
                      spmv_l.P                  * sizeof(float));
    snrt_dma_wait_all();
  }

  snrt_cluster_hw_barrier();

  // Run the kernel on core 0
  if (cid == 0) {
    start_kernel();
    sp_spmv(res, a, w, nm_index,
            spmv_l.N, spmv_l.P_W, spmv_l.NM_INDEX_ROW_WORDS,
            spmv_l.IDX_WIDTH, spmv_l.M_SPARSE, spmv_l.N_SPARSE);
    stop_kernel();
  }

  snrt_cluster_hw_barrier();

  if (cid == 0) {
    if (fp32_check(golden, res, spmv_l.P))
      printf("WRONG!\n");
    else
      printf("CORRECT!\n");
    printf("\n----- (%dx%d) SpMV - vfx -----\n", spmv_l.N, spmv_l.P);
    printf("DONE\n");
  }

  snrt_cluster_hw_barrier();
  return 0;
}
