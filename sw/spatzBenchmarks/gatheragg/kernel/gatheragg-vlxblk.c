// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// gatheragg, VLXBLK arm: out[b, :] = sum_l t[idx[b, l], :] (indexed-row
// gather + sum pooling; DLRM SparseLengthsSum fp32 at row_d = 32, GNN
// pull-mode neighbour aggregation at row_d = 64).
//
// Gather-across-units (the proven vqgemv pattern): upg = 128 / row_d
// destinations are pooled simultaneously -- iteration l gathers the l-th
// row of all upg units into one full m4 group (one index per unit) and
// accumulates with a whole-group vfadd. NO register-group slicing: the
// old per-row m1/m2-slice + switch-fallthrough pattern hangs (d32) or
// zeroes (d64) in RTL.
//
// m4, not m8: no PASSING kernel computes at m8 (m8 vfadd is the remaining
// hang suspect after vlse16 was exonerated; vqdecode proves m8 GATHERS are
// fine). All shapes here match the proven vqgemv/vqgemm m4 patterns. The
// shapes are identical for d32 / d64 (vl = upg * row_d = 128 e32), so one
// function serves both geometries.
//
// STATUS: this arm HANGS in RTL (open bug A, late hang at ~87.5%); the
// instruction sequence and register allocation are kept verbatim from the
// version under waveform debug. The vle baseline is verified.

#include "gatheragg-vlxblk.h"
#include <stdio.h>

void agg_vlxblk(float *out, const float *t, const uint16_t *idx,
                const unsigned int nb, const unsigned int row_d,
                const unsigned int lp, const unsigned int dbg_every) {
  const unsigned int upg = 128 / row_d; // units per m4 group (4x d32 / 2x d64)
  const unsigned int ec = upg * row_d;  // e32 elements per group (= 128)

  asm volatile("vsetblklen %0" ::"r"(row_d));

  // Accumulator v16-v19 (m4) holds upg units side by side; the gathered
  // rows land in v8-v11 (m4) in the same unit positions, so the
  // accumulate is one whole-group vfadd.
  for (unsigned int b = 0; b < nb; b += upg) {
    // Hang locator for the waveform session (bug A). dbg config: every 32
    // bags = 8 d32 groups = 320 gathers; 0 in measurement configs.
    if (dbg_every && (b % dbg_every) == 0)
      printf("DBGG %u\n", b);

    const uint16_t *bi = idx + b * lp;

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(ec));
    asm volatile("vmv.v.i v16, 0");

    for (unsigned int l = 0; l < lp; ++l) {
      // Index layout is TRANSPOSED (round-major within each unit group):
      // round l's upg indices are contiguous -> plain vle16 (vlse16
      // strided loads hang on this port; see ISA probe).
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(upg));
      asm volatile("vle16.v v2, (%0)" ::"r"(bi + l * upg) : "memory");

      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(ec));
      asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
      asm volatile("vfadd.vv v16, v16, v8");
    }

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(ec));
    asm volatile("vse32.v v16, (%0)" ::"r"(out + b * row_d) : "memory");
  }
}
