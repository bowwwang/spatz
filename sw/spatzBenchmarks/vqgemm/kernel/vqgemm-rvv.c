// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// vqgemm, plain-RVV baseline: the same 4-row tiling as the VLXBLK arm
// (4 x m4 fp16 accumulators, vl=N), but the decoded row w_k is produced
// the natural RVV way: group-by-group, each codebook entry loaded by a
// scalar-computed address (idx * cb_d) with vle16, added, scaled, and
// stored into a scratch row; the row is then reloaded as ONE m4 vector
// and consumed by the 4 vfmacc.vf. Without indexed BLOCK loads this is
// the strongest direct translation (a per-element vluxei would need the
// entry numbers expanded to cb_d per-element byte offsets first).
//
// Register map (e16): v8/v12/v16/v20 the four m4 row accumulators;
// v24/v28 (m1) decode targets, v24 (m4) the reloaded w_k.

#include "vqgemm-rvv.h"

// ---------------
// 8-element entries (AQLM 2x8)
// ---------------

void vqgemm_rvv_d8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                   const __fp16 *cb1, const uint8_t *idx0,
                   const uint8_t *idx1, const __fp16 *scales, __fp16 *wrow,
                   const unsigned int M, const unsigned int N,
                   const unsigned int K) {
  const unsigned int cb_d = 8;
  const unsigned int groups = N / cb_d;

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

      // Decode w_k group-by-group into the scratch row (natural RVV).
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));
      for (unsigned int g = 0; g < groups; ++g) {
        const unsigned int ix = k * groups + g;
        const __fp16 *p0 = cb0 + (unsigned int)idx0[ix] * cb_d;
        const __fp16 *p1 = cb1 + (unsigned int)idx1[ix] * cb_d;

        asm volatile("vle16.v v24, (%0)" ::"r"(p0) : "memory");
        asm volatile("vle16.v v28, (%0)" ::"r"(p1) : "memory");
        asm volatile("vfadd.vv v24, v24, v28");
        asm volatile("vfmul.vf v24, v24, %0" ::"f"(scale));

        // Store padded to 32 B (16 elements): vector stores narrower than
        // 32 B leave cache ports idle and hang (erratum #1; see isaprobe).
        // Ascending g means each group's slot is last written by its own
        // store, so the upper 8 lanes of garbage are always overwritten;
        // wrow has +16 elements of slack for the final group.
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
        asm volatile("vse16.v v24, (%0)" ::"r"(wrow + g * cb_d) : "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));
      }

      // Reload the decoded row as one m4 vector, 4 MACs.
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
      asm volatile("vle16.v v24, (%0)" ::"r"(wrow) : "memory");
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
// 16-element entries (VPTQ v16)
// ---------------

void vqgemm_rvv_d16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                    const __fp16 *cb1, const uint16_t *idx0,
                    const uint16_t *idx1, const __fp16 *scales, __fp16 *wrow,
                    const unsigned int M, const unsigned int N,
                    const unsigned int K) {
  const unsigned int cb_d = 16;
  const unsigned int groups = N / cb_d;

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

      // Decode w_k group-by-group into the scratch row (natural RVV).
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));
      for (unsigned int g = 0; g < groups; ++g) {
        const unsigned int ix = k * groups + g;
        const __fp16 *p0 = cb0 + (unsigned int)idx0[ix] * cb_d;
        const __fp16 *p1 = cb1 + (unsigned int)idx1[ix] * cb_d;

        asm volatile("vle16.v v24, (%0)" ::"r"(p0) : "memory");
        asm volatile("vle16.v v28, (%0)" ::"r"(p1) : "memory");
        asm volatile("vfadd.vv v24, v24, v28");
        asm volatile("vfmul.vf v24, v24, %0" ::"f"(scale));

        // 16 x e16 = 32 B: the natural store is already port-filling
        asm volatile("vse16.v v24, (%0)" ::"r"(wrow + g * cb_d) : "memory");
      }

      // Reload the decoded row as one m4 vector, 4 MACs.
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(N));
      asm volatile("vle16.v v24, (%0)" ::"r"(wrow) : "memory");
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
