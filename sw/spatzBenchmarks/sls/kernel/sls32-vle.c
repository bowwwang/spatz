// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// sls-2 (fp32, 128-B rows), plain-RVV baseline: the row IS one register.
// At VLEN = 1024 an e32 m1 group holds 32 fp32 = exactly one 128-B table
// row, so each lookup is one unit-stride vle32 at a scalar-computed
// address (idx * row_d) plus one vfadd into its slot accumulator - the
// same 8-slot output layout as the vlxblk arm (row l accumulates into
// slot l % 8), so both arms are compared on identical arithmetic.
//
// ONE LOAD IN FLIGHT, deliberately: this is the shape of every baseline
// on this machine that runs over a table far larger than L1 (gnnagg-vle
// 256-B rows, the old gatheragg-d32 128-B rows). The fp16 sls-1 baseline,
// which keeps two 32-B loads in flight over a 2-MiB table, is the one
// that hangs (see the plan's baseline-hang table), so the extra load in
// flight buys nothing worth the risk here.
//
// Requirements: row_d == 32, lp a multiple of 8.

#include "sls32-vle.h"

void sls32_vle(float *out, const float *t, const uint16_t *idx,
               const unsigned int nb, const unsigned int lp,
               const unsigned int row_d) {
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(row_d));

  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bi = idx + b * lp;

    // rows 0..7 initialise the eight slot accumulators v16..v23
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[0] * row_d) : "memory");
    asm volatile("vmv.v.v v16, v8");
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[1] * row_d) : "memory");
    asm volatile("vmv.v.v v17, v8");
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[2] * row_d) : "memory");
    asm volatile("vmv.v.v v18, v8");
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[3] * row_d) : "memory");
    asm volatile("vmv.v.v v19, v8");
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[4] * row_d) : "memory");
    asm volatile("vmv.v.v v20, v8");
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[5] * row_d) : "memory");
    asm volatile("vmv.v.v v21, v8");
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[6] * row_d) : "memory");
    asm volatile("vmv.v.v v22, v8");
    asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bi[7] * row_d) : "memory");
    asm volatile("vmv.v.v v23, v8");

    // rows 8.. accumulate into slot (l % 8)
    for (unsigned int l = 8; l < lp; l += 8) {
      const uint16_t *bj = bi + l;
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[0] * row_d) : "memory");
      asm volatile("vfadd.vv v16, v16, v8");
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[1] * row_d) : "memory");
      asm volatile("vfadd.vv v17, v17, v8");
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[2] * row_d) : "memory");
      asm volatile("vfadd.vv v18, v18, v8");
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[3] * row_d) : "memory");
      asm volatile("vfadd.vv v19, v19, v8");
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[4] * row_d) : "memory");
      asm volatile("vfadd.vv v20, v20, v8");
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[5] * row_d) : "memory");
      asm volatile("vfadd.vv v21, v21, v8");
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[6] * row_d) : "memory");
      asm volatile("vfadd.vv v22, v22, v8");
      asm volatile("vle32.v v8, (%0)" ::"r"(t + (unsigned int)bj[7] * row_d) : "memory");
      asm volatile("vfadd.vv v23, v23, v8");
    }

    // eight 128-B slot stores
    float *o = out + b * 8u * row_d;
    asm volatile("vse32.v v16, (%0)" ::"r"(o + 0 * row_d) : "memory");
    asm volatile("vse32.v v17, (%0)" ::"r"(o + 1 * row_d) : "memory");
    asm volatile("vse32.v v18, (%0)" ::"r"(o + 2 * row_d) : "memory");
    asm volatile("vse32.v v19, (%0)" ::"r"(o + 3 * row_d) : "memory");
    asm volatile("vse32.v v20, (%0)" ::"r"(o + 4 * row_d) : "memory");
    asm volatile("vse32.v v21, (%0)" ::"r"(o + 5 * row_d) : "memory");
    asm volatile("vse32.v v22, (%0)" ::"r"(o + 6 * row_d) : "memory");
    asm volatile("vse32.v v23, (%0)" ::"r"(o + 7 * row_d) : "memory");
  }
}
