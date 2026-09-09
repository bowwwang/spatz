// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// sls, VLXBLK arm: SparseLengthsSum / embedding-bag pooling over an fp16
// table with 32-B rows (row_d = 16 elements):  out_b[d] = sum_l t[id_bl][d],
// fp16 in, fp16 accumulate, fp16 out — no conversion anywhere. (Paper point:
// the 32-B block access pattern; int8 dequant / per-row scale+bias dropped.)
//
// 32 + 8 ROWS PER BAG IN TWO GATHERS (m8 + m2). A bag's ids are contiguous:
// one vle16 loads 32 of them (64 B) and one vlxblkei16 under e16 m8
// (vl = 512) gathers 32 rows = 1 KiB into v8-15; a second vle16 loads the
// remaining ids (16 loaded = 32 B, 8 used) and an m2 gather (vl = 128)
// fetches the 8 tail rows into v4-5. Versus the m2 version (5 x {idx load,
// gather, add}) this halves the instruction count per bag and quadruples
// the bytes per gather; the bytes moved are unchanged.
//
// OUTPUT = the 8 chunk-slot partial sums per bag (out[b][8][16], 256 B),
// slot i = rows l with l % 8 == i, exactly as before: the m8 group's four
// m2 quarters (v8-9: rows 0-7, v10-11: 8-15, v12-13: 16-23, v14-15: 24-31)
// are legal m2 operands, so acc = q0 (copy), acc += q1, q2, q3, then acc +=
// tail (rows 32-39) reproduces the m2 kernel's summation order bit-exactly
// (same expected data). No lane reduction inside the kernel (a slide-based
// one stalled — user ruling 2026-09-09).
//
// SOFTWARE PIPELINE across BAGS: two load sets alternate (A: idx v2, rows
// v8-15, tail v4-5; B: idx v3, rows v24-31, tail v6-7) so bag b+1's gathers
// are in flight while bag b's adds and store execute.
//   prologue : load A(0)
//   loop     : load B(b+1) | arith A(b) | load A(b+2) | arith B(b+1)
//   epilogue : arith B(nb-1)
// Bus-width rule: every vector load/store here moves >= 32 B.
// Requirements (not asserted): row_d = 16; lp = 40 (one 32-row block +
// an 8-row tail); nb even; idx carries >= 16 elements of tail padding.
//
// Register map:
//   v16-17   fp16 accumulator, 128 lanes (m2) = the 8 slot partials
//   A  idx v2 (32 / 16 ids)   rows v8-15 (m8, 32 x 32 B)   tail v4-5 (m2, 8 rows)
//   B  idx v3                 rows v24-31                  tail v6-7

#include "sls-vlxblk.h"
#include <stdio.h>

void sls_vlxblk(__fp16 *out, const __fp16 *t, const uint16_t *idx,
                const unsigned int nb, const unsigned int lp,
                const unsigned int row_d, const unsigned int dbg_every) {
  const unsigned int blk_el = 32u * row_d;      // 32 rows as e16 elements (512)
  const unsigned int tail_el = 8u * row_d;      // 8 rows as e16 elements (128)
  const unsigned int out_el = 8u * row_d;       // 8 slot partials per bag (128)

  asm volatile("vsetblklen %0" ::"r"(row_d));

  const uint16_t *bi;

  // ---- prologue: load A(0) ----
  bi = idx;
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(32u));       // 32 ids = 64 B
  asm volatile("vle16.v v2, (%0)" ::"r"(bi) : "memory");
  asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(blk_el));    // 32 rows
  asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));       // 16 ids = 32 B, 8 used
  asm volatile("vle16.v v2, (%0)" ::"r"(bi + 32u) : "memory");
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(tail_el));   // 8 tail rows
  asm volatile("vlxblkei16.v v4, (%0), v2" ::"r"(t) : "memory");

  unsigned int b = 0;
  while (1) {
    if (dbg_every && (b % dbg_every) == 0)
      printf("DBGB %u\n", b);

    // ---- load B(b+1) ----
    bi = idx + (b + 1) * lp;
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(32u));
    asm volatile("vle16.v v3, (%0)" ::"r"(bi) : "memory");
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(blk_el));
    asm volatile("vlxblkei16.v v24, (%0), v3" ::"r"(t) : "memory");
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v3, (%0)" ::"r"(bi + 32u) : "memory");
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(tail_el));
    asm volatile("vlxblkei16.v v6, (%0), v3" ::"r"(t) : "memory");

    // ---- arith A(b): quarters of the m8 group + tail into the 8 slots ----
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(out_el));
    asm volatile("vmv.v.v v16, v8");                 // rows 0-7   (init)
    asm volatile("vfadd.vv v16, v16, v10");          // rows 8-15
    asm volatile("vfadd.vv v16, v16, v12");          // rows 16-23
    asm volatile("vfadd.vv v16, v16, v14");          // rows 24-31
    asm volatile("vfadd.vv v16, v16, v4");           // rows 32-39
    asm volatile("vse16.v v16, (%0)" ::"r"(out + b * out_el) : "memory");   // 256 B

    b += 1;
    if (b + 1 == nb) {
      // ---- epilogue: arith B(nb-1) ----
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(out_el));
      asm volatile("vmv.v.v v16, v24");
      asm volatile("vfadd.vv v16, v16, v26");
      asm volatile("vfadd.vv v16, v16, v28");
      asm volatile("vfadd.vv v16, v16, v30");
      asm volatile("vfadd.vv v16, v16, v6");
      asm volatile("vse16.v v16, (%0)" ::"r"(out + b * out_el) : "memory");
      break;
    }

    // ---- load A(b+1) ----
    bi = idx + (b + 1) * lp;
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(32u));
    asm volatile("vle16.v v2, (%0)" ::"r"(bi) : "memory");
    asm volatile("vsetvli zero, %0, e16, m8, ta, ma" ::"r"(blk_el));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(t) : "memory");
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
    asm volatile("vle16.v v2, (%0)" ::"r"(bi + 32u) : "memory");
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(tail_el));
    asm volatile("vlxblkei16.v v4, (%0), v2" ::"r"(t) : "memory");

    // ---- arith B(b) ----
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(out_el));
    asm volatile("vmv.v.v v16, v24");
    asm volatile("vfadd.vv v16, v16, v26");
    asm volatile("vfadd.vv v16, v16, v28");
    asm volatile("vfadd.vv v16, v16, v30");
    asm volatile("vfadd.vv v16, v16, v6");
    asm volatile("vse16.v v16, (%0)" ::"r"(out + b * out_el) : "memory");

    b += 1;
    // nb even: the loop always exits through the epilogue above
  }
}
