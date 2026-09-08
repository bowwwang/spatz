// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// vqgemv, VLXBLK arm: c[N] = sum_k a[k] * scale[k] * (cb0[idx0[k,g]] +
// cb1[idx1[k,g]]) with the 2-codebook decode fused into the GEMV. One
// vlxblkei8/16 gathers a whole vl of decoded entries (vectorized ACROSS
// output groups: lanes [g*cb_d, (g+1)*cb_d) hold entry idx[k,g]).
//
// SOFTWARE PIPELINE (sp-fmatmul style). Per k the work is one dependent
// chain: index load -> block gather -> vfadd -> vfmul -> vfmacc. Spatz
// issues in order without renaming, so reusing one register set per k
// exposes the whole L1->VRF latency on every iteration. Two ROUNDS
// alternate two register sets: while the VFU works on round 1, the VLSU
// is already fetching round 2's indices and codebook blocks, and vice
// versa.
//   prologue : load round 1 (k = 0)
//   hot loop : load round 2 (k+1) | arith round 1 (k) |
//              load round 1 (k+2) | arith round 2 (k+1)
//   epilogue : arith round 2 (K-1)
// K must be even (paper configs: K = 128).
//
// ONE vtype per group (e16, m2, vl = gvl): the index loads carry their own
// EEW (vle8/vle16) and run under that vtype, over-reading gvl index
// elements (the following k-rows, then the data header's tail padding);
// the gathers consume only the first gvl/cb_d of them. No per-k vsetvli.
//
// LMUL = m2: at VLEN = 1024 one m2 group is 128 fp16 = the whole N = 128
// output (same elements per instruction as the former m4 at N = 128), and
// it keeps the EEW=16 index vectors at EMUL = 2 so both rounds fit.
//
// Register map:
//   v0-1    accumulator
//   round 1 idx0 v28(-29)  idx1 v30(-31)  cb0 v16-17  cb1 v18-19
//   round 2 idx0 v24(-25)  idx1 v26(-27)  cb0 v8-9    cb1 v10-11

#include "vqgemv-vlxblk.h"
#include <stddef.h>

// ---------------
// u8 indices
// ---------------

void vqgemv_vlxblk_ei8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                       const __fp16 *cb1, const uint8_t *idx0,
                       const uint8_t *idx1, const __fp16 *scales,
                       const unsigned int K, const unsigned int N,
                       const unsigned int cb_d) {
  const unsigned int groups = N / cb_d;

  asm volatile("vsetblklen %0" ::"r"(cb_d));

  unsigned int g = 0;
  while (g < groups) {
    // One vtype for the whole group
    size_t gvl;
    asm volatile("vsetvli %[gvl], %[vl], e16, m2, ta, ma"
                 : [gvl] "=r"(gvl)
                 : [vl] "r"((groups - g) * cb_d));
    const unsigned int group_vl = gvl / cb_d;

    asm volatile("vmv.v.x v0, zero");

    // k-row pointers of the NEXT row to load (row k is at idx + k*groups)
    const uint8_t *p0 = idx0 + g;
    const uint8_t *p1 = idx1 + g;
    const __fp16 *pa = a;
    const __fp16 *ps = scales;

    float av1, sc1, av2, sc2;

    // ---- prologue: load round 1 (k = 0) ----
    asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
    asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
    asm volatile("vle8.v v28, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle8.v v30, (%0)" ::"r"(p1) : "memory");
    asm volatile("vlxblkei8.v v16, (%0), v28" ::"r"(cb0) : "memory");
    asm volatile("vlxblkei8.v v18, (%0), v30" ::"r"(cb1) : "memory");
    p0 += groups;
    p1 += groups;
    pa += 1;
    ps += 1;

    unsigned int k = 0;
    while (1) {
      // ---- load round 2 (k + 1) ----
      asm volatile("flh %0, 0(%1)" : "=f"(av2) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc2) : "r"(ps));
      asm volatile("vle8.v v24, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle8.v v26, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei8.v v8, (%0), v24" ::"r"(cb0) : "memory");
      asm volatile("vlxblkei8.v v10, (%0), v26" ::"r"(cb1) : "memory");
      p0 += groups;
      p1 += groups;
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
      asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
      asm volatile("vle8.v v28, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle8.v v30, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei8.v v16, (%0), v28" ::"r"(cb0) : "memory");
      asm volatile("vlxblkei8.v v18, (%0), v30" ::"r"(cb1) : "memory");
      p0 += groups;
      p1 += groups;
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

    asm volatile("vse16.v v0, (%0)" ::"r"(c + g * cb_d) : "memory");
    g += group_vl;
  }
}

// ---------------
// u16 indices
// ---------------

void vqgemv_vlxblk_ei16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                        const __fp16 *cb1, const uint16_t *idx0,
                        const uint16_t *idx1, const __fp16 *scales,
                        const unsigned int K, const unsigned int N,
                        const unsigned int cb_d) {
  const unsigned int groups = N / cb_d;

  asm volatile("vsetblklen %0" ::"r"(cb_d));

  unsigned int g = 0;
  while (g < groups) {
    size_t gvl;
    asm volatile("vsetvli %[gvl], %[vl], e16, m2, ta, ma"
                 : [gvl] "=r"(gvl)
                 : [vl] "r"((groups - g) * cb_d));
    const unsigned int group_vl = gvl / cb_d;

    asm volatile("vmv.v.x v0, zero");

    const uint16_t *p0 = idx0 + g;
    const uint16_t *p1 = idx1 + g;
    const __fp16 *pa = a;
    const __fp16 *ps = scales;

    float av1, sc1, av2, sc2;

    // ---- prologue: load round 1 (k = 0) ----
    asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
    asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
    asm volatile("vle16.v v28, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle16.v v30, (%0)" ::"r"(p1) : "memory");
    asm volatile("vlxblkei16.v v16, (%0), v28" ::"r"(cb0) : "memory");
    asm volatile("vlxblkei16.v v18, (%0), v30" ::"r"(cb1) : "memory");
    p0 += groups;
    p1 += groups;
    pa += 1;
    ps += 1;

    unsigned int k = 0;
    while (1) {
      // ---- load round 2 (k + 1) ----
      asm volatile("flh %0, 0(%1)" : "=f"(av2) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc2) : "r"(ps));
      asm volatile("vle16.v v24, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v26, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei16.v v8, (%0), v24" ::"r"(cb0) : "memory");
      asm volatile("vlxblkei16.v v10, (%0), v26" ::"r"(cb1) : "memory");
      p0 += groups;
      p1 += groups;
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
      asm volatile("flh %0, 0(%1)" : "=f"(av1) : "r"(pa));
      asm volatile("flh %0, 0(%1)" : "=f"(sc1) : "r"(ps));
      asm volatile("vle16.v v28, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v30, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei16.v v16, (%0), v28" ::"r"(cb0) : "memory");
      asm volatile("vlxblkei16.v v18, (%0), v30" ::"r"(cb1) : "memory");
      p0 += groups;
      p1 += groups;
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

    asm volatile("vse16.v v0, (%0)" ::"r"(c + g * cb_d) : "memory");
    g += group_vl;
  }
}
