// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// slsint8, VLXBLK arm: SparseLengthsSum / embedding-bag over a row-wise
// quantized int8 table (FBGEMM EmbeddingSpMDM8Bit class), split layout:
// a table of row_d-byte u8 rows plus a parallel (scale, bias) fp16 array
// fetched with the same index.
// Per bag b: out_b[d] = sum_l ( s_l * row_l[d] ) + sum_l b_l
// (row-wise bias is element-uniform, so it folds into one scalar).
//
// Per lookup one vlxblkei16 gathers the 32-B u8 row; widen u8->u16->u32
// (vwmulu x2), vfcvt to f32, one vfmacc.vf with the row scale into the
// f32 accumulator v24 (m4 shapes throughout). Scale/bias pairs are
// scalar-loaded via the same index (flh + fcvt.s.h).

#include "sls-vlxblk.h"

void sls_vlxblk(float *out, const uint8_t *t, const __fp16 *sb,
                const uint16_t *idx, const unsigned int nb,
                const unsigned int lp, const unsigned int row_d) {
  // Gather at e16 (blk_len = row_d/2 = 16 x e16 = the same 32 bytes,
  // same entry numbers): the e8-data miss path hangs on the first
  // missing gather (wave-4 bisection; probe P7 passes L1-resident).
  // The register bytes are identical; the widen chain reinterprets
  // them at e8.
  const unsigned int row_h = row_d / 2;

  asm volatile("vsetblklen %0" ::"r"(row_h));

  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bi = idx + b * lp;
    float bias = 0.0f;

    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(row_d));
    asm volatile("vmv.v.i v24, 0");

    for (unsigned int l = 0; l < lp; ++l) {
      const uint32_t id = bi[l];

      // row scale (fp16 -> f32 in hardware; the software cast is broken)
      float hs, hb, fs, fb;
      asm volatile("flh %0, 0(%1)" : "=f"(hs) : "r"(sb + 2u * id));
      asm volatile("fcvt.s.h %0, %1" : "=f"(fs) : "f"(hs));
      // row bias: element-uniform, summed as a scalar
      asm volatile("flh %0, 0(%1)" : "=f"(hb) : "r"(sb + 2u * id + 1u));
      asm volatile("fcvt.s.h %0, %1" : "=f"(fb) : "f"(hb));
      bias += fb;

      // index vector: the single entry number
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(1u));
      asm volatile("vmv.s.x v2, %0" ::"r"(id));

      // gather the row as 16 x e16 (see the note above)
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(row_h));
      asm volatile("vlxblkei16.v v4, (%0), v2" ::"r"(t) : "memory");

      // dequant-accumulate the row (reinterpreted at e8, vl = row_d):
      // widen u8->u16 (v8), u16->u32 (v12), convert to f32, vfmacc with
      // the row scale into v24
      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(row_d));
      asm volatile("vwmulu.vx v8, v4, %0" ::"r"(1u));
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(row_d));
      asm volatile("vwmulu.vx v12, v8, %0" ::"r"(1u));
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(row_d));
      asm volatile("vfcvt.f.xu.v v12, v12");
      asm volatile("vfmacc.vf v24, %0, v12" ::"f"(fs));
    }

    // fold the bias sum in and store the bag
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(row_d));
    asm volatile("vfadd.vf v24, v24, %0" ::"f"(bias));
    asm volatile("vse32.v v24, (%0)" ::"r"(out + b * row_d) : "memory");
  }
}
