// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// sls, plain-RVV BASELINE: SparseLengthsSum / embedding-bag pooling over an
// fp16 table with 32-B rows (row_d = 16), same output as the VLXBLK arm:
// the 8 chunk-slot partial sums per bag, out[b][i][d] = sum_c t[id_{8c+i}][d]
// (row l of the bag accumulates into slot l % 8; fp16 in, fp16 accumulate).
// Each row is loaded by a scalar-computed address (vle16 of 16 x e16 = 32 B,
// full bus width) into one of two alternating registers so the next row's
// load is in flight while the current row is added (skill recipe A); the
// 8 slot accumulators are static registers v16..v23 (m1, 16 lanes), the
// first chunk initializes them with register copies, and each is stored
// with a 32-B vse16. Without indexed block loads the 8-rows-per-instruction
// assembly of the VLXBLK arm is not available, which is the comparison.
// Requirements (not asserted): row_d = 16; lp a multiple of 8.
//
// Register map:  v16..v23  slot accumulators (m1)   A row v4   B row v6

#include "sls-vle.h"
#include <stdio.h>

void sls_vle(__fp16 *out, const __fp16 *t, const uint16_t *idx,
             const unsigned int nb, const unsigned int lp,
             const unsigned int row_d, const unsigned int dbg_every) {
  const unsigned int nchunks = lp / 8u;

  for (unsigned int b = 0; b < nb; ++b) {
    if (dbg_every && (b % dbg_every) == 0)
      printf("DBGB %u\n", b);
    const uint16_t *bi = idx + b * lp;

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(row_d));

    // ---- chunk 0: init the 8 slot accumulators (row i -> v16+i) ----
    asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bi[0] * row_d) : "memory");
    asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bi[1] * row_d) : "memory");
    asm volatile("vmv.v.v v16, v4");
    asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bi[2] * row_d) : "memory");
    asm volatile("vmv.v.v v17, v6");
    asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bi[3] * row_d) : "memory");
    asm volatile("vmv.v.v v18, v4");
    asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bi[4] * row_d) : "memory");
    asm volatile("vmv.v.v v19, v6");
    asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bi[5] * row_d) : "memory");
    asm volatile("vmv.v.v v20, v4");
    asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bi[6] * row_d) : "memory");
    asm volatile("vmv.v.v v21, v6");
    asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bi[7] * row_d) : "memory");
    asm volatile("vmv.v.v v22, v4");
    asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bi[8] * row_d) : "memory");
    asm volatile("vmv.v.v v23, v6");

    // ---- chunks 1 .. nchunks-1: row 8c+i accumulates into v16+i ----
    for (unsigned int c = 1; c < nchunks; ++c) {
      const uint16_t *bj = bi + c * 8u;
      asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bj[1] * row_d) : "memory");
      asm volatile("vfadd.vv v16, v16, v4");
      asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bj[2] * row_d) : "memory");
      asm volatile("vfadd.vv v17, v17, v6");
      asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bj[3] * row_d) : "memory");
      asm volatile("vfadd.vv v18, v18, v4");
      asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bj[4] * row_d) : "memory");
      asm volatile("vfadd.vv v19, v19, v6");
      asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bj[5] * row_d) : "memory");
      asm volatile("vfadd.vv v20, v20, v4");
      asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bj[6] * row_d) : "memory");
      asm volatile("vfadd.vv v21, v21, v6");
      asm volatile("vle16.v v6, (%0)" ::"r"(t + (unsigned int)bj[7] * row_d) : "memory");
      asm volatile("vfadd.vv v22, v22, v4");
      if (c + 1 < nchunks)
        asm volatile("vle16.v v4, (%0)" ::"r"(t + (unsigned int)bj[8] * row_d) : "memory");
      asm volatile("vfadd.vv v23, v23, v6");
    }

    // ---- store the 8 slot partials (8 x 32 B) ----
    asm volatile("vse16.v v16, (%0)" ::"r"(out + b * 8u * row_d + 0u * row_d) : "memory");
    asm volatile("vse16.v v17, (%0)" ::"r"(out + b * 8u * row_d + 1u * row_d) : "memory");
    asm volatile("vse16.v v18, (%0)" ::"r"(out + b * 8u * row_d + 2u * row_d) : "memory");
    asm volatile("vse16.v v19, (%0)" ::"r"(out + b * 8u * row_d + 3u * row_d) : "memory");
    asm volatile("vse16.v v20, (%0)" ::"r"(out + b * 8u * row_d + 4u * row_d) : "memory");
    asm volatile("vse16.v v21, (%0)" ::"r"(out + b * 8u * row_d + 5u * row_d) : "memory");
    asm volatile("vse16.v v22, (%0)" ::"r"(out + b * 8u * row_d + 6u * row_d) : "memory");
    asm volatile("vse16.v v23, (%0)" ::"r"(out + b * 8u * row_d + 7u * row_d) : "memory");
  }
}
