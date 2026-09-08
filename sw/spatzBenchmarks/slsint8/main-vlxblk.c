// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// slsint8, VLXBLK arm. Data (indices, per-bag checksums) from the
// generated DATAHEADER; the 2 MiB table and the 256 KiB scale/bias array
// are filled on-core from the closed-form patterns the generator mirrors
// (see script/gen_data.py).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/sls-vlxblk.c"

#include "bench_fill.h"

// Table fill: exact byte head t[i] = (i * 37) & 255 over `head` bytes
// (= BF_HEAD_BYTES = 122 rows), then vector-tiled to the full table.
// Mirrored bit-exactly in gen_data.py.
static void fill_table(uint8_t *t, const unsigned int n,
                       const unsigned int head) {
  const unsigned int h = n < head ? n : head;
  for (unsigned int i = 0; i < h; ++i)
    t[i] = (uint8_t)((i * 37u) & 255u);
  if (n > h)
    bench_fill_rep(t, n, h);
}

// Scale/bias fill: row r -> scale = (1 + r % 15) / 1024, bias =
// (r % 63 - 31) / 256 (exact dyadic fp16). Written for `head_rows` rows
// (a multiple of the 315-row pattern period and of 16 rows = 64 B), then
// vector-tiled: the tiling reproduces the closed form for every row.
// Mirrored bit-exactly in gen_data.py.
static void fill_sb(__fp16 *sb, const unsigned int nrows,
                    const unsigned int head_rows) {
  const unsigned int h = nrows < head_rows ? nrows : head_rows;
  for (unsigned int r = 0; r < h; ++r) {
    sb[2u * r] = (__fp16)((float)(1u + r % 15u) / 1024.0f);
    sb[2u * r + 1u] = (__fp16)((float)((int)(r % 63u) - 31) / 256.0f);
  }
  if (nrows > h)
    bench_fill_rep(sb, nrows * 4u, h * 4u);
}

// Verify every bag: the f32 sum of its row_d outputs against the float64
// checksum from gen_data.py (bit-exact fp32 emulation of the kernel's op
// order). Tolerance 1e-4 relative + 1e-3 absolute covers the on-core f32
// summation of 32 values (~2e-6 relative).
int verify_output(const float *out, const float *checksum,
                  const unsigned int nb, const unsigned int row_d) {
  for (unsigned int b = 0; b < nb; ++b) {
    float sum = 0.0f;
    for (unsigned int d = 0; d < row_d; ++d)
      sum += out[b * row_d + d];
    float chk = checksum[b];
    float err = sum > chk ? sum - chk : chk - sum;
    float mag = chk < 0 ? -chk : chk;
    if (err > 0.001f + 0.0001f * mag) {
      printf("FAILED b=%u sum=%f chk=%f\n", b, sum, chk);
      return b == 0 ? -1 : (int)b;
    }
  }
  return 0;
}

int main() {
  const unsigned int cid = snrt_cluster_core_idx();

#if USE_CACHE == 1
  uint32_t spm_size = 16;
#else
  uint32_t spm_size = 120;
#endif

  if (cid == 0) {
    // Init the cache
    l1d_init(spm_size);
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  int error = 0;
  unsigned int timer = 0;

  if (cid == 0) {
    fill_table(sl_tbl, sl_l.NROWS * sl_l.ROW_D, sl_l.HEAD);
    fill_sb(sl_sb, sl_l.NROWS, sl_l.SB_HEAD);
    bench_fill_zero(sl_out, sizeof(sl_out));

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    sls_vlxblk(sl_out, sl_tbl, sl_sb, sl_idx, sl_l.NB, sl_l.LP, sl_l.ROW_D);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(sl_out, sl_checksum, sl_l.NB, sl_l.ROW_D);

#ifdef PRINT_RESULT
    printf("slsint8 vlxblk nb=%u lp=%u row_d=%u nrows=%u cache=%d: took %u "
           "cycles %s\n",
           sl_l.NB, sl_l.LP, sl_l.ROW_D, sl_l.NROWS, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK");
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
