// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// sls-2, VLXBLK arm: SparseLengthsSum / embedding-bag pooling over an fp32
// table with 128-B rows (row_d = 32):  out_b[d] = sum_l t[id_bl][d], fp32 in,
// fp32 accumulate, fp32 out. The paper's full-precision point (128-B blocks).
//
// EIGHT ROWS PER GATHER, ONE ADD PER GATHER: at VLEN = 1024 a 128-B fp32 row
// is exactly one register, an e32 m8 gather (vl = 256) fetches 8 rows =
// 1 KiB, and the 8-slot accumulator (slot i = rows l with l % 8 == i, 32
// lanes each) is itself one m8 group — so each gather is followed by ONE
// vfadd.vv (m8). A 40-row bag = 5 gathers + 5 adds (the first add is from a
// zero group v0-7 set once per kernel: no vmv, no zeroing pass per bag) +
// one 1-KiB store. Index loads fetch 16 ids (32 B, full bus width) and the
// gather consumes 8.
//
// PIPELINE at gather granularity: consecutive gathers alternate the two
// m8 row groups A (v8-15) and B (v24-31), so gather k+1 is in flight while
// the add of gather k executes; 5 gathers per bag -> the assignment
// alternates across bags, so the bag loop is unrolled by two.
//   bag 2j  : G0->A G1->B G2->A G3->B G4->A
//   bag 2j+1: G0->B G1->A G2->B G3->A G4->B
// Requirements (not asserted): row_d = 32; lp = 40; nb even; idx carries
// >= 16 elements of tail padding.
//
// Register map:
//   v0-7     zero group (m8), set once
//   v16-23   fp32 accumulator, 256 lanes (m8) = the 8 slot partials
//   A rows v8-15 (m8)   B rows v24-31 (m8)   idx v2 / v3 (16 ids each)

#include "sls32-vlxblk.h"
#include <stdio.h>

void sls32_vlxblk(float *out, const float *t, const uint16_t *idx,
                  const unsigned int nb, const unsigned int lp,
                  const unsigned int row_d, const unsigned int dbg_every) {
  const unsigned int g_el = 8u * row_d;      // 8 rows as e32 elements (256)
  const unsigned int out_el = 8u * row_d;    // 8 slot partials per bag (256)

  asm volatile("vsetblklen %0" ::"r"(row_d));
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
  asm volatile("vmv.v.i v0, 0");             // zero group, once

  for (unsigned int b = 0; b < nb; b += 2) {
    if (dbg_every && (b % dbg_every) == 0)
      printf("DBGB %u\n", b);
    const uint16_t *bi = idx + b * lp;         // bag 2j
    const uint16_t *bj = bi + lp;              // bag 2j+1

    // ================= bag 2j: G0->A G1->B G2->A G3->B G4->A =================
    // G0 -> A
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v2, (%0)" ::"r"(bi) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
    // G1 -> B
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v3, (%0)" ::"r"(bi + 8u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v24, (%0), v3" ::"r"(t) : "memory");
    // acc = 0 + G0
    asm volatile("vfadd.vv v16, v0, v8");
    // G2 -> A
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v2, (%0)" ::"r"(bi + 16u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
    // acc += G1
    asm volatile("vfadd.vv v16, v16, v24");
    // G3 -> B
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v3, (%0)" ::"r"(bi + 24u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v24, (%0), v3" ::"r"(t) : "memory");
    // acc += G2
    asm volatile("vfadd.vv v16, v16, v8");
    // G4 -> A
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v2, (%0)" ::"r"(bi + 32u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
    // acc += G3
    asm volatile("vfadd.vv v16, v16, v24");
    // next bag's G0 -> B (in flight while this bag finishes)
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v3, (%0)" ::"r"(bj) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v24, (%0), v3" ::"r"(t) : "memory");
    // acc += G4 ; store bag 2j (1 KiB)
    asm volatile("vfadd.vv v16, v16, v8");
    asm volatile("vse32.v v16, (%0)" ::"r"(out + b * out_el) : "memory");

    // ================= bag 2j+1: G0->B (loaded) G1->A G2->B G3->A G4->B =================
    // G1 -> A
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v2, (%0)" ::"r"(bj + 8u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
    // acc = 0 + G0
    asm volatile("vfadd.vv v16, v0, v24");
    // G2 -> B
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v3, (%0)" ::"r"(bj + 16u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v24, (%0), v3" ::"r"(t) : "memory");
    // acc += G1
    asm volatile("vfadd.vv v16, v16, v8");
    // G3 -> A
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v2, (%0)" ::"r"(bj + 24u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
    // acc += G2
    asm volatile("vfadd.vv v16, v16, v24");
    // G4 -> B
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v3, (%0)" ::"r"(bj + 32u) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(g_el));
    asm volatile("vlxblkei16.v v24, (%0), v3" ::"r"(t) : "memory");
    // acc += G3
    asm volatile("vfadd.vv v16, v16, v8");
    // acc += G4 ; store bag 2j+1
    asm volatile("vfadd.vv v16, v16, v24");
    asm volatile("vse32.v v16, (%0)" ::"r"(out + (b + 1u) * out_el) : "memory");
  }
}
