// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// attn-vagg, VLXBLK arm: out[q, :] = sum_k p[q, k] * pool[idx[q, k], :],
// top-K sparse-attention V aggregation (Quest-class selection, FlashInfer
// page_size=1 records) over 256-B fp16 rows: hd = 128 e16 = EXACTLY m4.
//
// Per selected token: the index is loaded as a 1-element vector, one
// vlxblkei16 gathers the whole row (blk_len = hd elements, so the index is
// consumed as a row number directly) into v8 in its natural m4 position,
// and one vfmacc.vf with the score accumulates it into the m4 accumulator
// v24-v27. No register-group slicing (m8 @ e16 holds TWO 256-B rows, not
// four, and clamps vl at m2 slices).

#include "attn-vlxblk.h"

void attn_vlxblk(__fp16 *out, const __fp16 *pool, const uint16_t *idx,
                 const __fp16 *p, const unsigned int nq,
                 const unsigned int topk, const unsigned int hd) {
  asm volatile("vsetblklen %0" ::"r"(hd));

  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * topk;
    const __fp16 *qp = p + q * topk;

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(hd));
    asm volatile("vmv.v.i v24, 0");

    for (unsigned int k = 0; k < topk; ++k) {
      // index as a 1-element vector (row number; blk_len scales it)
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(1u));
      asm volatile("vle16.v v2, (%0)" ::"r"(qi + k) : "memory");

      // gather the whole row, then score-weighted accumulate
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(hd));
      asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(pool) : "memory");

      float pk;
      asm volatile("flh %0, 0(%1)" : "=f"(pk) : "r"(qp + k) : "memory");
      asm volatile("vfmacc.vf v24, %0, v8" ::"f"(pk));
    }

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(hd));
    asm volatile("vse16.v v24, (%0)" ::"r"(out + q * hd) : "memory");
  }
}
