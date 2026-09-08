// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// gatheragg, vle baseline (rows >= one register: piecewise rule). One
// destination (bag / node) per iteration; each pooled row is loaded by a
// scalar-computed address (idx * row_d) with vle32 and accumulated with
// vfadd in ascending l (same add order as the vlxblk arm).
//
// The index array is TRANSPOSED (round-major within each group of
// upg = 128 / row_d destinations, the vlxblk arm's layout): unit b reads
// position (b / upg) * lp * upg + l * upg + (b % upg).
//
// The LMUL is part of the instruction: a 32-f32 row is exactly one
// register (m1), a 64-f32 row two (m2) -> one function per geometry,
// agg_vle() selects at runtime on row_d. Verified 1,500,616 (d32) /
// 846,325 (d64) cycles at n16384.

#include "gatheragg-vle.h"

void agg_vle(float *out, const float *t, const uint16_t *idx,
             const unsigned int nb, const unsigned int row_d,
             const unsigned int lp) {
  if (row_d == 32) {
    agg_vle_d32(out, t, idx, nb, lp);
  } else {
    agg_vle_d64(out, t, idx, nb, lp);
  }
}

// ---------------
// 32-f32 rows (m1)
// ---------------

void agg_vle_d32(float *out, const float *t, const uint16_t *idx,
                 const unsigned int nb, const unsigned int lp) {
  const unsigned int row_d = 32;
  const unsigned int upg = 128 / row_d;

  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bg = idx + (b / upg) * (lp * upg) + (b % upg);

    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(row_d));
    asm volatile("vmv.v.i v24, 0");

    for (unsigned int l = 0; l < lp; ++l) {
      const float *row = t + (unsigned int)bg[l * upg] * row_d;

      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(row_d));
      asm volatile("vle32.v v8, (%0)" ::"r"(row) : "memory");
      asm volatile("vfadd.vv v24, v24, v8");
    }

    asm volatile("vse32.v v24, (%0)" ::"r"(out + b * row_d) : "memory");
  }
}

// ---------------
// 64-f32 rows (m2)
// ---------------

void agg_vle_d64(float *out, const float *t, const uint16_t *idx,
                 const unsigned int nb, const unsigned int lp) {
  const unsigned int row_d = 64;
  const unsigned int upg = 128 / row_d;

  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bg = idx + (b / upg) * (lp * upg) + (b % upg);

    asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(row_d));
    asm volatile("vmv.v.i v24, 0");

    for (unsigned int l = 0; l < lp; ++l) {
      const float *row = t + (unsigned int)bg[l * upg] * row_d;

      asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(row_d));
      asm volatile("vle32.v v8, (%0)" ::"r"(row) : "memory");
      asm volatile("vfadd.vv v24, v24, v8");
    }

    asm volatile("vse32.v v24, (%0)" ::"r"(out + b * row_d) : "memory");
  }
}
