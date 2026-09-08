// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// vqgemm, VLXBLK arm: C[M,N] = A[M,K] x W[K,N] with the 2-codebook VQ
// decode of W fused into the product (AQLM/VPTQ-class residual VQ):
// w_k[N] = scale[k] * (cb0[idx0[k,g]] + cb1[idx1[k,g]]), group g of lane
// range [g*cb_d, (g+1)*cb_d) being entry idx[k,g], exactly as in vqgemv.
//
// Tiling: 4 output rows live in the VRF (4 x m4 fp16 accumulators, vl=N).
// The decoded row w_k is produced ONCE per (tile, k) by two vlxblkei8/16
// gathers over the full vector and consumed by 4 vfmacc.vf -- the
// VRF-capacity tiling any fused kernel needs (the decode is redone M/4
// times; stated in the paper). The index vector is consumed as entry
// numbers directly (vsetblklen = cb_d elements per entry).
//
// Register map (e16): v2/v3 index vectors (m1); v8/v12/v16/v20 the four
// m4 row accumulators; v24-v27 + v28-v31 gather targets, v24 becomes w_k.

#include "vqgemm-vlxblk.h"

// ---------------
// u8 indices
// ---------------

void vqgemm_vlxblk_ei8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                       const __fp16 *cb1, const uint8_t *idx0,
                       const uint8_t *idx1, const __fp16 *scales,
                       const unsigned int M, const unsigned int N,
                       const unsigned int K, const unsigned int cb_d) {
  const unsigned int groups = N / cb_d;

  asm volatile("vsetblklen %0" ::"r"(cb_d));

  for (unsigned int m = 0; m < M; m += 4) {
    // 4 row accumulators over the full output row
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vmv.v.i v12, 0");
    asm volatile("vmv.v.i v16, 0");
    asm volatile("vmv.v.i v20, 0");

    for (unsigned int k = 0; k < K; ++k) {
      float scale, a0, a1, a2, a3;
      asm volatile("flh %0, 0(%1)" : "=f"(scale) : "r"(scales + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a0) : "r"(a + (m + 0) * K + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a1) : "r"(a + (m + 1) * K + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a2) : "r"(a + (m + 2) * K + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a3) : "r"(a + (m + 3) * K + k));

      // index vectors: groups entry numbers of row k
      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(groups));
      asm volatile("vle8.v v2, (%0)" ::"r"(idx0 + k * groups) : "memory");
      asm volatile("vle8.v v3, (%0)" ::"r"(idx1 + k * groups) : "memory");

      // gather + decode w_k once, accumulate into the 4 row tiles
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
      asm volatile("vlxblkei8.v v24, (%0), v2" ::"r"(cb0) : "memory");
      asm volatile("vlxblkei8.v v28, (%0), v3" ::"r"(cb1) : "memory");
      asm volatile("vfadd.vv v24, v24, v28");
      asm volatile("vfmul.vf v24, v24, %0" ::"f"(scale));
      asm volatile("vfmacc.vf v8, %0, v24" ::"f"(a0));
      asm volatile("vfmacc.vf v12, %0, v24" ::"f"(a1));
      asm volatile("vfmacc.vf v16, %0, v24" ::"f"(a2));
      asm volatile("vfmacc.vf v20, %0, v24" ::"f"(a3));
    }

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
    asm volatile("vse16.v v8, (%0)" ::"r"(c + (m + 0) * N) : "memory");
    asm volatile("vse16.v v12, (%0)" ::"r"(c + (m + 1) * N) : "memory");
    asm volatile("vse16.v v16, (%0)" ::"r"(c + (m + 2) * N) : "memory");
    asm volatile("vse16.v v20, (%0)" ::"r"(c + (m + 3) * N) : "memory");
  }
}

// ---------------
// u16 indices
// ---------------

void vqgemm_vlxblk_ei16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                        const __fp16 *cb1, const uint16_t *idx0,
                        const uint16_t *idx1, const __fp16 *scales,
                        const unsigned int M, const unsigned int N,
                        const unsigned int K, const unsigned int cb_d) {
  const unsigned int groups = N / cb_d;

  asm volatile("vsetblklen %0" ::"r"(cb_d));

  for (unsigned int m = 0; m < M; m += 4) {
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vmv.v.i v12, 0");
    asm volatile("vmv.v.i v16, 0");
    asm volatile("vmv.v.i v20, 0");

    for (unsigned int k = 0; k < K; ++k) {
      float scale, a0, a1, a2, a3;
      asm volatile("flh %0, 0(%1)" : "=f"(scale) : "r"(scales + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a0) : "r"(a + (m + 0) * K + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a1) : "r"(a + (m + 1) * K + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a2) : "r"(a + (m + 2) * K + k));
      asm volatile("flh %0, 0(%1)" : "=f"(a3) : "r"(a + (m + 3) * K + k));

      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(groups));
      asm volatile("vle16.v v2, (%0)" ::"r"(idx0 + k * groups) : "memory");
      asm volatile("vle16.v v3, (%0)" ::"r"(idx1 + k * groups) : "memory");

      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
      asm volatile("vlxblkei16.v v24, (%0), v2" ::"r"(cb0) : "memory");
      asm volatile("vlxblkei16.v v28, (%0), v3" ::"r"(cb1) : "memory");
      asm volatile("vfadd.vv v24, v24, v28");
      asm volatile("vfmul.vf v24, v24, %0" ::"f"(scale));
      asm volatile("vfmacc.vf v8, %0, v24" ::"f"(a0));
      asm volatile("vfmacc.vf v12, %0, v24" ::"f"(a1));
      asm volatile("vfmacc.vf v16, %0, v24" ::"f"(a2));
      asm volatile("vfmacc.vf v20, %0, v24" ::"f"(a3));
    }

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
    asm volatile("vse16.v v8, (%0)" ::"r"(c + (m + 0) * N) : "memory");
    asm volatile("vse16.v v12, (%0)" ::"r"(c + (m + 1) * N) : "memory");
    asm volatile("vse16.v v16, (%0)" ::"r"(c + (m + 2) * N) : "memory");
    asm volatile("vse16.v v20, (%0)" ::"r"(c + (m + 3) * N) : "memory");
  }
}
