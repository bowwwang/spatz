// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// attn-vagg (paper row "spattn", sa-gemv), VLXBLK arm v1: top-K sparse
// attention V aggregation, out[q, :] = sum_k p[q, k] * V[idx[q, k], :] over
// 256-B fp16 rows (hd = 128 = one e16 m4 register group; FlashInfer
// page_size = 1 records, Quest-class token selection).
//
// Step = 2 tokens of one query: one 32-B index load (16 ids, the first 2
// used; ids are row-major, so consecutive tokens are contiguous) drives ONE
// e16 m8 block gather of the two rows (512 B, blk_len = hd) into a row set;
// the two scores are broadcast into the two m4 halves of the score group
// v0-7 (vfmv.v.f); the row set is scaled by the score group at m8 (vfmul),
// and the two halves are folded into the m4 accumulator v24-27 (two vfadd;
// step 0 initializes it with one vfadd of the two halves, no zeroing).
//
// Two-round software pipeline: row sets A (v8-15) and B (v16-23)
// alternate; the next step's index load + gather are issued before the
// current step's arithmetic. Scores are loaded with flh one step ahead,
// the hp-fmatmul reference idiom (user ruling 2026-09-09). One 256-B
// store per query. Index register v28. Requirements: hd == 128, topk even
// >= 4, idx padded by >= 16 ids.
//
// STATUS 2026-09-09: HANGS on the dev tile (attn-vagg-vlxblk-nq2) in RTL -
// under waveform debug (erratum #7 candidate: the m4 vfadd of the upper
// half v12 / v20 of the m8 row group). Kept verbatim for that session.

#include "attn-vlxblk.h"

void attn_vlxblk(__fp16 *out, const __fp16 *pool, const uint16_t *idx,
                 const __fp16 *p, const unsigned int nq,
                 const unsigned int topk, const unsigned int hd) {
  const unsigned int idx_el = 16u;     // ids per index load (32 B), 2 used
  const unsigned int half_el = 128u;   // one row = e16 m4
  const unsigned int pair_el = 256u;   // two rows = e16 m8
  float s0, s1;                        // scores of the current step
  float n0, n1;                        // scores of the next step

  asm volatile("vsetblklen %0" ::"r"(hd));

  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * topk;
    const __fp16 *qp = p + q * topk;

    // prologue: step 0 -> A, step 1 -> B; scores of step 0
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
    asm volatile("vle16.v v28, (%0)" ::"r"(qi) : "memory");
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(pair_el));
    asm volatile("vlxblkei16.v v8, (%0), v28" ::"r"(pool) : "memory");
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
    asm volatile("vle16.v v28, (%0)" ::"r"(qi + 2) : "memory");
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(pair_el));
    asm volatile("vlxblkei16.v v16, (%0), v28" ::"r"(pool) : "memory");
    asm volatile("flh %[t], 0(%[a])" : [t] "=f"(s0) : [a] "r"(qp + 0));
    asm volatile("flh %[t], 0(%[a])" : [t] "=f"(s1) : [a] "r"(qp + 1));

    // step 0 (set A): scores, scale, acc = lo + hi
    asm volatile("flh %[t], 0(%[a])" : [t] "=f"(n0) : [a] "r"(qp + 2));
    asm volatile("flh %[t], 0(%[a])" : [t] "=f"(n1) : [a] "r"(qp + 3));
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(half_el));
    asm volatile("vfmv.v.f v0, %0" ::"f"(s0));
    asm volatile("vfmv.v.f v4, %0" ::"f"(s1));
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(pair_el));
    asm volatile("vfmul.vv v8, v8, v0");
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(half_el));
    asm volatile("vfadd.vv v24, v8, v12");
    s0 = n0;
    s1 = n1;

    for (unsigned int k = 2; k < topk; k += 4) {
      // ---- load A (tokens k+2, k+3), if any; then step k (set B) ----
      if (k + 2 < topk) {
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
        asm volatile("vle16.v v28, (%0)" ::"r"(qi + k + 2) : "memory");
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(pair_el));
        asm volatile("vlxblkei16.v v8, (%0), v28" ::"r"(pool) : "memory");
        asm volatile("flh %[t], 0(%[a])" : [t] "=f"(n0) : [a] "r"(qp + k + 2));
        asm volatile("flh %[t], 0(%[a])" : [t] "=f"(n1) : [a] "r"(qp + k + 3));
      }
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(half_el));
      asm volatile("vfmv.v.f v0, %0" ::"f"(s0));
      asm volatile("vfmv.v.f v4, %0" ::"f"(s1));
      asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(pair_el));
      asm volatile("vfmul.vv v16, v16, v0");
      asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(half_el));
      asm volatile("vfadd.vv v24, v24, v16");
      asm volatile("vfadd.vv v24, v24, v20");
      s0 = n0;
      s1 = n1;

      // ---- load B (tokens k+4, k+5), if any; then step k+2 (set A) ----
      if (k + 2 < topk) {
        if (k + 4 < topk) {
          asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(idx_el));
          asm volatile("vle16.v v28, (%0)" ::"r"(qi + k + 4) : "memory");
          asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(pair_el));
          asm volatile("vlxblkei16.v v16, (%0), v28" ::"r"(pool) : "memory");
          asm volatile("flh %[t], 0(%[a])" : [t] "=f"(n0) : [a] "r"(qp + k + 4));
          asm volatile("flh %[t], 0(%[a])" : [t] "=f"(n1) : [a] "r"(qp + k + 5));
        }
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(half_el));
        asm volatile("vfmv.v.f v0, %0" ::"f"(s0));
        asm volatile("vfmv.v.f v4, %0" ::"f"(s1));
        asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(pair_el));
        asm volatile("vfmul.vv v8, v8, v0");
        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(half_el));
        asm volatile("vfadd.vv v24, v24, v8");
        asm volatile("vfadd.vv v24, v24, v12");
        s0 = n0;
        s1 = n1;
      }
    }

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(half_el));
    asm volatile("vse16.v v24, (%0)" ::"r"(out + q * hd) : "memory");
  }
}
