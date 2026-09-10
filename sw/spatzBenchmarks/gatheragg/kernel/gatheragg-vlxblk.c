// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// gatheragg (paper row "gnnagg"), VLXBLK arm: GNN pull-mode neighbour
// aggregation, out[b, :] = sum_l t[idx[b, l], :] over lp neighbours of
// 64 fp32 (one 256-B row = one e32 m2 register pair).
//
// Gather-across-nodes: the 4 nodes of one e32 m8 group are pooled
// simultaneously - round l gathers the l-th neighbour row of all 4 nodes
// with ONE vlxblkei16 (4 blocks of 256 B = 1 KiB, one index per node) and
// accumulates with ONE whole-group vfadd into the m8 accumulator v24-31,
// whose 4 quarters are the 4 nodes' outputs (one 1-KiB vse32 per group).
// Every load group is consumed at the LMUL it was loaded with (erratum
// #7: hazards are tracked per base register only).
//
// Two-round software pipeline per group: rows of even rounds land in set
// A (v8-15), odd rounds in set B (v16-23); the next round's index load
// (16 ids = 32 B, the first 4 used; the transposed index layout makes
// round l's 4 ids contiguous) and gather are issued before the current
// round's add. Round 0 initializes the accumulator with vfmul by 1.0 (no
// zeroing). Requirements: row_d == 64, nb a multiple of 4, lp >= 2, idx
// padded by >= 16 ids.

#include "gatheragg-vlxblk.h"

void agg_vlxblk(float *out, const float *t, const uint16_t *idx,
                const unsigned int nb, const unsigned int row_d,
                const unsigned int lp) {
  const unsigned int upg = 4u;     // nodes per m8 group (4 x 64 = 256 lanes)
  const unsigned int ec = 256u;    // e32 elements per group
  const unsigned int idx_el = 16u; // ids per index load (32 B), 4 used
  float fone;

  asm volatile("fmv.w.x %0, %1" : "=f"(fone) : "r"(0x3f800000u));
  asm volatile("vsetblklen %0" ::"r"(row_d));

  for (unsigned int g = 0; g < nb; g += upg) {
    const uint16_t *bi = idx + g * lp; // group stride = lp * upg ids

    // prologue: round 0 -> A, round 1 -> B
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
    asm volatile("vle16.v v2, (%0)" ::"r"(bi) : "memory");
    asm volatile("vle16.v v3, (%0)" ::"r"(bi + upg) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(ec));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
    asm volatile("vlxblkei16.v v16, (%0), v3" ::"r"(t) : "memory");
    // preload
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
    asm volatile("vle16.v v2, (%0)" ::"r"(bi + 2*upg) : "memory");
    asm volatile("vle16.v v3, (%0)" ::"r"(bi + 3*upg) : "memory");

    // acc = round 0
    asm volatile("vfmul.vf v24, v8, %0" ::"f"(fone));

    for (unsigned int l = 1; l < lp; l += 2) {
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(ec));
      asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
      asm volatile("vfadd.vv v24, v24, v16");

      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
      asm volatile("vle16.v v2, (%0)" ::"r"(bi + (l + 3) * upg) : "memory");
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(ec));

      asm volatile("vlxblkei16.v v16, (%0), v3" ::"r"(t) : "memory");
      asm volatile("vfadd.vv v24, v24, v8");
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
      asm volatile("vle16.v v3, (%0)" ::"r"(bi + (l + 4) * upg) : "memory");
    }

    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(ec));
    asm volatile("vse32.v v24, (%0)" ::"r"(out + g * row_d) : "memory");
  }
}
