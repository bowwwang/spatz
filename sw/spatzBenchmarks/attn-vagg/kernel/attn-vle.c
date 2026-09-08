// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// attn-vagg, plain-RVV baseline (piecewise rule at 4-register rows): the
// scalar-indexed vle loop. Per selected token: load the index (scalar),
// compute the row address, vle16 the whole 256-B row in the same m4 shape
// as the VLXBLK arm, score-weighted vfmacc.vf into the m4 accumulator
// v24-v27. Identical arithmetic order to the VLXBLK arm.

#include "attn-vle.h"

void attn_vle(__fp16 *out, const __fp16 *pool, const uint16_t *idx,
              const __fp16 *p, const unsigned int nq, const unsigned int topk,
              const unsigned int hd) {
  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * topk;
    const __fp16 *qp = p + q * topk;

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(hd));
    asm volatile("vmv.v.i v24, 0");

    for (unsigned int k = 0; k < topk; ++k) {
      const __fp16 *row = pool + (unsigned int)qi[k] * hd;

      asm volatile("vle16.v v8, (%0)" ::"r"(row) : "memory");

      float pk;
      asm volatile("flh %0, 0(%1)" : "=f"(pk) : "r"(qp + k) : "memory");
      asm volatile("vfmacc.vf v24, %0, v8" ::"f"(pk));
    }

    asm volatile("vse16.v v24, (%0)" ::"r"(out + q * hd) : "memory");
  }
}
