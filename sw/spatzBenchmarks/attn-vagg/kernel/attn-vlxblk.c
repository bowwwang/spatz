// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// attn-vagg (paper row "spattn", sa-gemv), VLXBLK arm v4: top-K sparse
// attention V aggregation, out[q, :] = sum_k p[q,k] * V[idx[q,k], :] over
// 256-B fp16 rows.
//
// ONE ROW PER GATHER AT e16 m2. At VLEN = 1024 an e16 m2 group is exactly
// 128 fp16 = one 256-B V row, so the gather fetches a single block
// (vsetblklen = hd) and the whole row is one register pair. Each token has
// its own score, so a wider group would need a per-half score vector -
// that shape (v1: m8 gather of two rows, m4 adds on the halves) HANGS in
// this RTL (erratum #7: narrower lane-wise reads of a wider load group's
// upper registers). Here every group is produced AND consumed at m2
// through its base register: rows v8-9 / v10-11, accumulator v24-25.
//
// INDEX PREFETCHED TWO TOKENS AHEAD (the optimization the user found on
// gnnagg, worth 26 cycles per gather there): v2/v3 always hold the ids of
// tokens k+3 / k+4 while the gathers consume k+1 / k+2, so a gather never
// waits for its own index load. Scores are loaded one token ahead with flh
// (hp-fmatmul idiom). topk even; the prologue does token 0, the loop does
// pairs (k, k+1) up to topk-2, and the epilogue adds the last token. The
// index array is padded (>= 16 ids): the last query prefetches 2 ids past
// its list.

#include "attn-vlxblk.h"

void attn_vlxblk(__fp16 *out, const __fp16 *pool, const uint16_t *idx,
                 const __fp16 *p, const unsigned int nq,
                 const unsigned int topk, const unsigned int hd) {
  const unsigned int idx_el = 16u; // ids per index load (32 B), 1 used
  float s, sn;

  asm volatile("vsetblklen %0" ::"r"(hd));

  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * topk;
    const __fp16 *qp = p + q * topk;

    // prologue: ids of tokens 0..1, gathers of tokens 0..1, ids of 2..3
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
    asm volatile("vle16.v v2, (%0)" ::"r"(qi) : "memory");
    asm volatile("vle16.v v3, (%0)" ::"r"(qi + 1) : "memory");
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(hd));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(pool) : "memory");
    asm volatile("vlxblkei16.v v10, (%0), v3" ::"r"(pool) : "memory");
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
    asm volatile("vle16.v v2, (%0)" ::"r"(qi + 2) : "memory");
    asm volatile("vle16.v v3, (%0)" ::"r"(qi + 3) : "memory");
    asm volatile("flh %[t], 0(%[a])" : [t] "=f"(s) : [a] "r"(qp));
    asm volatile("flh %[t], 0(%[a])" : [t] "=f"(sn) : [a] "r"(qp + 1));

    // token 0
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(hd));
    asm volatile("vfmul.vf v24, v8, %0" ::"f"(s));
    s = sn;

    // pairs (k, k+1); at loop entry s = p[k] and v10 holds token k
    for (unsigned int k = 1; k + 2 < topk; k += 2) {
      asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(pool) : "memory");
      asm volatile("vfmacc.vf v24, %0, v10" ::"f"(s));
      asm volatile("flh %[t], 0(%[a])" : [t] "=f"(sn) : [a] "r"(qp + k + 1));
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
      asm volatile("vle16.v v2, (%0)" ::"r"(qi + k + 3) : "memory");
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(hd));
      s = sn;

      asm volatile("vlxblkei16.v v10, (%0), v3" ::"r"(pool) : "memory");
      asm volatile("vfmacc.vf v24, %0, v8" ::"f"(s));
      asm volatile("flh %[t], 0(%[a])" : [t] "=f"(sn) : [a] "r"(qp + k + 2));
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
      asm volatile("vle16.v v3, (%0)" ::"r"(qi + k + 4) : "memory");
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(hd));
      s = sn;
    }

    // epilogue: the last token (its row is in v10, its score in s)
    asm volatile("vfmacc.vf v24, %0, v10" ::"f"(s));
    asm volatile("vse16.v v24, (%0)" ::"r"(out + q * hd) : "memory");
  }
}
