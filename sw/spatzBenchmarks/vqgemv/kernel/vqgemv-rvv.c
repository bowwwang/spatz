// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// vqgemv, plain-RVV baseline: the natural direct kernel. One output group
// (one codebook entry width) per outer iteration; each entry is loaded by
// a scalar-computed address (idx * cb_d) with vle16. This is the strongest
// translation available without indexed BLOCK loads: a per-element vluxei
// would need the entry number expanded to cb_d per-element byte offsets
// first.
//
// SOFTWARE PIPELINE — the exact mirror of the VLXBLK arm's, so the two
// arms differ only in how the codebook rows are fetched. Per k the work is
// one dependent chain (scalar index load -> address -> vle16 x2 -> vfadd ->
// vfmul -> vfmacc); two ROUNDS alternate two register sets so the VLSU
// fetches round 2's rows while the VFU works on round 1, and vice versa.
//   prologue : load round 1 (k = 0)
//   hot loop : load round 2 (k+1) | arith round 1 (k) |
//              load round 1 (k+2) | arith round 2 (k+1)
//   epilogue : arith round 2 (K-1)
// K must be even (paper configs: K = 128).
//
// Register map (m1):
//   v0      accumulator
//   round 1 cb0 v16  cb1 v18
//   round 2 cb0 v8   cb1 v10

#include "vqgemv-rvv.h"
#include <stddef.h>

// ---------------
// 8-element entries (AQLM 2x8)
// ---------------

void vqgemv_rvv_d8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                   const __fp16 *cb1, const uint8_t *idx0,
                   const uint8_t *idx1, const __fp16 *scales,
                   const unsigned int K, const unsigned int N) {
  const unsigned int cb_d = 8;
  const unsigned int groups = N / cb_d;

  for (unsigned int g = 0; g < groups; ++g) {
    // The padded store below leaves vl = 16: re-establish the group vtype
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));
    asm volatile("vmv.v.x v0, zero");

    // k-row pointers of the NEXT row to load (row k is at idx + k*groups)
    const uint8_t *i0 = idx0 + g;
    const uint8_t *i1 = idx1 + g;
    const __fp16 *pa = a;
    const __fp16 *ps = scales;

    const __fp16 *p0, *p1;
    float av1, sc1, av2, sc2;

    // ---- prologue: load round 1 (k = 0) ----
    p0 = cb0 + (unsigned int)*i0 * cb_d;
    p1 = cb1 + (unsigned int)*i1 * cb_d;
    asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
    asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
    asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle16.v v18, (%0)" ::"r"(p1) : "memory");
    i0 += groups;
    i1 += groups;
    pa += 1;
    ps += 1;

    unsigned int k = 0;
    while (1) {
      // ---- load round 2 (k + 1) ----
      p0 = cb0 + (unsigned int)*i0 * cb_d;
      p1 = cb1 + (unsigned int)*i1 * cb_d;
      asm volatile("flh %0, 0(%1)" : "=f"(av2) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc2) : "r"(ps));
      asm volatile("vle16.v v8, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v10, (%0)" ::"r"(p1) : "memory");
      i0 += groups;
      i1 += groups;
      pa += 1;
      ps += 1;

      // ---- arithmetic round 1 (k) ----
      asm volatile("vfadd.vv v16, v16, v18");
      asm volatile("vfmul.vf v16, v16, %0" ::"f"(sc1));
      asm volatile("vfmacc.vf v0, %0, v16" ::"f"(av1));

      k += 2;
      if (k == K)
        break;

      // ---- load round 1 (k) ----
      p0 = cb0 + (unsigned int)*i0 * cb_d;
      p1 = cb1 + (unsigned int)*i1 * cb_d;
      asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
      asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v18, (%0)" ::"r"(p1) : "memory");
      i0 += groups;
      i1 += groups;
      pa += 1;
      ps += 1;

      // ---- arithmetic round 2 (k - 1) ----
      asm volatile("vfadd.vv v8, v8, v10");
      asm volatile("vfmul.vf v8, v8, %0" ::"f"(sc2));
      asm volatile("vfmacc.vf v0, %0, v8" ::"f"(av2));
    }

    // ---- epilogue: arithmetic round 2 (K - 1) ----
    asm volatile("vfadd.vv v8, v8, v10");
    asm volatile("vfmul.vf v8, v8, %0" ::"f"(sc2));
    asm volatile("vfmacc.vf v0, %0, v8" ::"f"(av2));

    // Store padded to 32 B (16 elements): vector stores narrower than
    // 32 B leave cache ports idle and hang (erratum #1). Ascending g
    // means each group's slot is last written by its own store, so the
    // upper 8 lanes of garbage are always overwritten; c has +16
    // elements of slack for the final group.
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vse16.v v0, (%0)" ::"r"(c + g * cb_d) : "memory");
  }
}

// ---------------
// 16-element entries (VPTQ v16)
// ---------------

void vqgemv_rvv_d16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                    const __fp16 *cb1, const uint16_t *idx0,
                    const uint16_t *idx1, const __fp16 *scales,
                    const unsigned int K, const unsigned int N) {
  const unsigned int cb_d = 16;
  const unsigned int groups = N / cb_d;

  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(cb_d));

  for (unsigned int g = 0; g < groups; ++g) {
    asm volatile("vmv.v.x v0, zero");

    const uint16_t *i0 = idx0 + g;
    const uint16_t *i1 = idx1 + g;
    const __fp16 *pa = a;
    const __fp16 *ps = scales;

    const __fp16 *p0, *p1;
    float av1, sc1, av2, sc2;

    // ---- prologue: load round 1 (k = 0) ----
    p0 = cb0 + (unsigned int)*i0 * cb_d;
    p1 = cb1 + (unsigned int)*i1 * cb_d;
    asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
    asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
    asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle16.v v18, (%0)" ::"r"(p1) : "memory");
    i0 += groups;
    i1 += groups;
    pa += 1;
    ps += 1;

    unsigned int k = 0;
    while (1) {
      // ---- load round 2 (k + 1) ----
      p0 = cb0 + (unsigned int)*i0 * cb_d;
      p1 = cb1 + (unsigned int)*i1 * cb_d;
      asm volatile("flh %0, 0(%1)" : "=f"(av2) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc2) : "r"(ps));
      asm volatile("vle16.v v8, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v10, (%0)" ::"r"(p1) : "memory");
      i0 += groups;
      i1 += groups;
      pa += 1;
      ps += 1;

      // ---- arithmetic round 1 (k) ----
      asm volatile("vfadd.vv v16, v16, v18");
      asm volatile("vfmul.vf v16, v16, %0" ::"f"(sc1));
      asm volatile("vfmacc.vf v0, %0, v16" ::"f"(av1));

      k += 2;
      if (k == K)
        break;

      // ---- load round 1 (k) ----
      p0 = cb0 + (unsigned int)*i0 * cb_d;
      p1 = cb1 + (unsigned int)*i1 * cb_d;
      asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
      asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v18, (%0)" ::"r"(p1) : "memory");
      i0 += groups;
      i1 += groups;
      pa += 1;
      ps += 1;

      // ---- arithmetic round 2 (k - 1) ----
      asm volatile("vfadd.vv v8, v8, v10");
      asm volatile("vfmul.vf v8, v8, %0" ::"f"(sc2));
      asm volatile("vfmacc.vf v0, %0, v8" ::"f"(av2));
    }

    // ---- epilogue: arithmetic round 2 (K - 1) ----
    asm volatile("vfadd.vv v8, v8, v10");
    asm volatile("vfmul.vf v8, v8, %0" ::"f"(sc2));
    asm volatile("vfmacc.vf v0, %0, v8" ::"f"(av2));

    // 16 x e16 = 32 B: the natural store is already port-filling
    asm volatile("vse16.v v0, (%0)" ::"r"(c + g * cb_d) : "memory");
  }
}
