// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// vqgemm, plain-RVV baseline: the same 8-ROW BLOCKING as the VLXBLK arm
// (8 m2 accumulators v0..v14, vl = N, every decoded row feeds eight
// vfmacc.vf), but the decoded row w_k is produced the natural RVV way:
// group-by-group, each codebook entry loaded by a scalar-computed address
// (idx * cb_d) with vle16, added, scaled, and stored into a scratch row;
// the row is then reloaded as ONE m2 vector for the MACs. Without indexed
// BLOCK loads this is the strongest direct translation (a per-element
// vluxei would need the entry numbers expanded to cb_d per-element byte
// offsets first).
//
// SOFTWARE PIPELINE, applied where this kernel spends its time — the
// per-GROUP decode chain (vle16 x2 -> vfadd -> vfmul -> vse16), run
// N/cb_d x K x M/8 times: two m1 register sets (v16/v17, v18/v19)
// alternate across groups g, g+1 so the VLSU fetches the next group's two
// entries while the VFU decodes the current one. The scratch row is
// DOUBLE-BUFFERED by k parity, so the decode stores of row k+1 never wait
// for the reload of row k: the reload + 8 vfmacc of row k (VFU) overlap the
// decode of row k+1 (VLSU) with disjoint registers. The first k initializes
// the accumulators with vfmul.vf instead of vfmacc.vf (no zeroing pass).
// Requirements: M multiple of 8, groups = N/cb_d even, K even.
//
// Register map:
//   v0 v2 v4 v6 v8 v10 v12 v14   the eight row accumulators (m2)
//   v16 v17 / v18 v19            decode round 1 / round 2 (m1: cb0, cb1)
//   v20-21                       the reloaded w_k (m2)

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
  const unsigned int wstride = N + 16; // one scratch row incl. store slack

  for (unsigned int m = 0; m < M; m += 8) {
    const __fp16 *pa0 = a + (m + 0) * K;
    const __fp16 *pa1 = a + (m + 1) * K;
    const __fp16 *pa2 = a + (m + 2) * K;
    const __fp16 *pa3 = a + (m + 3) * K;
    const __fp16 *pa4 = a + (m + 4) * K;
    const __fp16 *pa5 = a + (m + 5) * K;
    const __fp16 *pa6 = a + (m + 6) * K;
    const __fp16 *pa7 = a + (m + 7) * K;

    for (unsigned int k = 0; k < K; ++k) {
      __fp16 *wbuf = wrow + (k & 1u) * wstride;
      const uint8_t *i0 = idx0 + k * groups;
      const uint8_t *i1 = idx1 + k * groups;

      float scale, t0, t1, t2, t3, t4, t5, t6, t7;
      asm volatile("flh %0, 0(%1)" : "=f"(scale) : "r"(scales + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t0) : "r"(pa0 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t1) : "r"(pa1 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t2) : "r"(pa2 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t3) : "r"(pa3 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t4) : "r"(pa4 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t5) : "r"(pa5 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t6) : "r"(pa6 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t7) : "r"(pa7 + k));

      // Decode w_k group-by-group into the scratch row, two-round pipelined.
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));

      const __fp16 *p0, *p1;

      // ---- prologue: load group 0 ----
      p0 = cb0 + (unsigned int)i0[0] * cb_d;
      p1 = cb1 + (unsigned int)i1[0] * cb_d;
      asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v17, (%0)" ::"r"(p1) : "memory");

      unsigned int g = 0;
      while (1) {
        // ---- load round 2 (group g + 1) ----
        p0 = cb0 + (unsigned int)i0[g + 1] * cb_d;
        p1 = cb1 + (unsigned int)i1[g + 1] * cb_d;
        asm volatile("vle16.v v18, (%0)" ::"r"(p0) : "memory");
        asm volatile("vle16.v v19, (%0)" ::"r"(p1) : "memory");

        // ---- decode + store round 1 (group g) ----
        asm volatile("vfadd.vv v16, v16, v17");
        asm volatile("vfmul.vf v16, v16, %0" ::"f"(scale));
        // Store padded to 32 B (16 elements): vector stores narrower than
        // 32 B leave cache ports idle and hang (erratum #1). Ascending g
        // means each group's slot is last written by its own store, so the
        // upper 8 lanes of garbage are always overwritten; the scratch row
        // has +16 elements of slack for the final group.
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
        asm volatile("vse16.v v16, (%0)" ::"r"(wbuf + g * cb_d) : "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));

        g += 2;
        if (g == groups)
          break;

        // ---- load round 1 (group g) ----
        p0 = cb0 + (unsigned int)i0[g] * cb_d;
        p1 = cb1 + (unsigned int)i1[g] * cb_d;
        asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
        asm volatile("vle16.v v17, (%0)" ::"r"(p1) : "memory");

        // ---- decode + store round 2 (group g - 1) ----
        asm volatile("vfadd.vv v18, v18, v19");
        asm volatile("vfmul.vf v18, v18, %0" ::"f"(scale));
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
        asm volatile("vse16.v v18, (%0)" ::"r"(wbuf + (g - 1) * cb_d) : "memory");
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));
      }

      // ---- epilogue: decode + store round 2 (last group) ----
      asm volatile("vfadd.vv v18, v18, v19");
      asm volatile("vfmul.vf v18, v18, %0" ::"f"(scale));
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
      asm volatile("vse16.v v18, (%0)" ::"r"(wbuf + (groups - 1) * cb_d) : "memory");

      // Reload the decoded row as one m2 vector, 8 MACs (k = 0 initializes
      // the accumulators with vfmul).
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(N));
      asm volatile("vle16.v v20, (%0)" ::"r"(wbuf) : "memory");
      if (k == 0) {
        asm volatile("vfmul.vf v0, v20, %0" ::"f"(t0));
        asm volatile("vfmul.vf v2, v20, %0" ::"f"(t1));
        asm volatile("vfmul.vf v4, v20, %0" ::"f"(t2));
        asm volatile("vfmul.vf v6, v20, %0" ::"f"(t3));
        asm volatile("vfmul.vf v8, v20, %0" ::"f"(t4));
        asm volatile("vfmul.vf v10, v20, %0" ::"f"(t5));
        asm volatile("vfmul.vf v12, v20, %0" ::"f"(t6));
        asm volatile("vfmul.vf v14, v20, %0" ::"f"(t7));
      } else {
        asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
        asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
        asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
        asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
        asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
        asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
        asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
        asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
      }
    }

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(N));
    asm volatile("vse16.v v0, (%0)" ::"r"(c + (m + 0) * N) : "memory");
    asm volatile("vse16.v v2, (%0)" ::"r"(c + (m + 1) * N) : "memory");
    asm volatile("vse16.v v4, (%0)" ::"r"(c + (m + 2) * N) : "memory");
    asm volatile("vse16.v v6, (%0)" ::"r"(c + (m + 3) * N) : "memory");
    asm volatile("vse16.v v8, (%0)" ::"r"(c + (m + 4) * N) : "memory");
    asm volatile("vse16.v v10, (%0)" ::"r"(c + (m + 5) * N) : "memory");
    asm volatile("vse16.v v12, (%0)" ::"r"(c + (m + 6) * N) : "memory");
    asm volatile("vse16.v v14, (%0)" ::"r"(c + (m + 7) * N) : "memory");
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
  const unsigned int wstride = N + 16;

  for (unsigned int m = 0; m < M; m += 8) {
    const __fp16 *pa0 = a + (m + 0) * K;
    const __fp16 *pa1 = a + (m + 1) * K;
    const __fp16 *pa2 = a + (m + 2) * K;
    const __fp16 *pa3 = a + (m + 3) * K;
    const __fp16 *pa4 = a + (m + 4) * K;
    const __fp16 *pa5 = a + (m + 5) * K;
    const __fp16 *pa6 = a + (m + 6) * K;
    const __fp16 *pa7 = a + (m + 7) * K;

    for (unsigned int k = 0; k < K; ++k) {
      __fp16 *wbuf = wrow + (k & 1u) * wstride;
      const uint16_t *i0 = idx0 + k * groups;
      const uint16_t *i1 = idx1 + k * groups;

      float scale, t0, t1, t2, t3, t4, t5, t6, t7;
      asm volatile("flh %0, 0(%1)" : "=f"(scale) : "r"(scales + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t0) : "r"(pa0 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t1) : "r"(pa1 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t2) : "r"(pa2 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t3) : "r"(pa3 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t4) : "r"(pa4 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t5) : "r"(pa5 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t6) : "r"(pa6 + k));
      asm volatile("flh %0, 0(%1)" : "=f"(t7) : "r"(pa7 + k));

      // 16 x e16 = 32 B entries: one vtype for the whole decode, the
      // natural store is already port-filling.
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));

      const __fp16 *p0, *p1;

      // ---- prologue: load group 0 ----
      p0 = cb0 + (unsigned int)i0[0] * cb_d;
      p1 = cb1 + (unsigned int)i1[0] * cb_d;
      asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v17, (%0)" ::"r"(p1) : "memory");

      unsigned int g = 0;
      while (1) {
        // ---- load round 2 (group g + 1) ----
        p0 = cb0 + (unsigned int)i0[g + 1] * cb_d;
        p1 = cb1 + (unsigned int)i1[g + 1] * cb_d;
        asm volatile("vle16.v v18, (%0)" ::"r"(p0) : "memory");
        asm volatile("vle16.v v19, (%0)" ::"r"(p1) : "memory");

        // ---- decode + store round 1 (group g) ----
        asm volatile("vfadd.vv v16, v16, v17");
        asm volatile("vfmul.vf v16, v16, %0" ::"f"(scale));
        asm volatile("vse16.v v16, (%0)" ::"r"(wbuf + g * cb_d) : "memory");

        g += 2;
        if (g == groups)
          break;

        // ---- load round 1 (group g) ----
        p0 = cb0 + (unsigned int)i0[g] * cb_d;
        p1 = cb1 + (unsigned int)i1[g] * cb_d;
        asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
        asm volatile("vle16.v v17, (%0)" ::"r"(p1) : "memory");

        // ---- decode + store round 2 (group g - 1) ----
        asm volatile("vfadd.vv v18, v18, v19");
        asm volatile("vfmul.vf v18, v18, %0" ::"f"(scale));
        asm volatile("vse16.v v18, (%0)" ::"r"(wbuf + (g - 1) * cb_d) : "memory");
      }

      // ---- epilogue: decode + store round 2 (last group) ----
      asm volatile("vfadd.vv v18, v18, v19");
      asm volatile("vfmul.vf v18, v18, %0" ::"f"(scale));
      asm volatile("vse16.v v18, (%0)" ::"r"(wbuf + (groups - 1) * cb_d) : "memory");

      // Reload the decoded row as one m2 vector, 8 MACs (k = 0 initializes
      // the accumulators with vfmul).
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(N));
      asm volatile("vle16.v v20, (%0)" ::"r"(wbuf) : "memory");
      if (k == 0) {
        asm volatile("vfmul.vf v0, v20, %0" ::"f"(t0));
        asm volatile("vfmul.vf v2, v20, %0" ::"f"(t1));
        asm volatile("vfmul.vf v4, v20, %0" ::"f"(t2));
        asm volatile("vfmul.vf v6, v20, %0" ::"f"(t3));
        asm volatile("vfmul.vf v8, v20, %0" ::"f"(t4));
        asm volatile("vfmul.vf v10, v20, %0" ::"f"(t5));
        asm volatile("vfmul.vf v12, v20, %0" ::"f"(t6));
        asm volatile("vfmul.vf v14, v20, %0" ::"f"(t7));
      } else {
        asm volatile("vfmacc.vf v0, %0, v20" ::"f"(t0));
        asm volatile("vfmacc.vf v2, %0, v20" ::"f"(t1));
        asm volatile("vfmacc.vf v4, %0, v20" ::"f"(t2));
        asm volatile("vfmacc.vf v6, %0, v20" ::"f"(t3));
        asm volatile("vfmacc.vf v8, %0, v20" ::"f"(t4));
        asm volatile("vfmacc.vf v10, %0, v20" ::"f"(t5));
        asm volatile("vfmacc.vf v12, %0, v20" ::"f"(t6));
        asm volatile("vfmacc.vf v14, %0, v20" ::"f"(t7));
      }
    }

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(N));
    asm volatile("vse16.v v0, (%0)" ::"r"(c + (m + 0) * N) : "memory");
    asm volatile("vse16.v v2, (%0)" ::"r"(c + (m + 1) * N) : "memory");
    asm volatile("vse16.v v4, (%0)" ::"r"(c + (m + 2) * N) : "memory");
    asm volatile("vse16.v v6, (%0)" ::"r"(c + (m + 3) * N) : "memory");
    asm volatile("vse16.v v8, (%0)" ::"r"(c + (m + 4) * N) : "memory");
    asm volatile("vse16.v v10, (%0)" ::"r"(c + (m + 5) * N) : "memory");
    asm volatile("vse16.v v12, (%0)" ::"r"(c + (m + 6) * N) : "memory");
    asm volatile("vse16.v v14, (%0)" ::"r"(c + (m + 7) * N) : "memory");
  }
}
