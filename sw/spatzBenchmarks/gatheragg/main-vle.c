// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// gatheragg, vle baseline arm. Identical data path to main-vlxblk.c; only
// the kernel differs.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/gatheragg-vle.c"

#include "bench_fill.h"

// Table fill: exact dyadic head (centred ramp, multiples of 2^-10 in
// [-0.477, 0.476)), then vector-tiled to the full table. Mirrored
// bit-exactly in gen_data.py; a ramp (not a hash) so that the distinct
// table rows have well-separated sums for the checksum verifier.
static void fill_table(float *t, const unsigned int n,
                       const unsigned int head) {
  const unsigned int h = n < head ? n : head;
  for (unsigned int i = 0; i < h; ++i)
    t[i] = (float)((int)i - 488) / 1024.0f;
  if (n > h)
    bench_fill_rep(t, n * 4u, h * 4u);
}

// Verify EVERY destination by a per-row checksum: the fp32 sum over its
// row_d outputs vs the float64 sum of the bit-exact fp32 emulation from
// gen_data.py. Tolerance 1e-4 * |chk| + 1e-3 covers the on-core fp32
// summation error (<= row_d ulp, ~4e-6 relative); one wrong gathered row
// shifts the checksum by O(1) (the distinct table rows have pairwise
// distinct sums, gap printed by gen_data.py).
int verify_output(const float *out, const float *chk, const unsigned int nb,
                  const unsigned int row_d) {
  for (unsigned int b = 0; b < nb; ++b) {
    float sum = 0.0f;
    for (unsigned int d = 0; d < row_d; ++d)
      sum += out[b * row_d + d];
    float exp = chk[b];
    float err = sum > exp ? sum - exp : exp - sum;
    float mag = exp < 0 ? -exp : exp;
    if (err > 0.001f + 0.0001f * mag) {
      printf("FAILED b=%u got=%f exp=%f\n", b, sum, exp);
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
    fill_table(ga_tbl, ga_l.NROWS * ga_l.ROW_D, ga_l.HEAD);
    bench_fill_zero(ga_out, sizeof(ga_out));

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    agg_vle(ga_out, ga_tbl, ga_idx, ga_l.NB, ga_l.ROW_D, ga_l.LP);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(ga_out, ga_chk, ga_l.NB, ga_l.ROW_D);

#ifdef PRINT_RESULT
    printf("gatheragg vle nb=%u row_d=%u lp=%u nrows=%u cache=%d: took %u "
           "cycles %s\n",
           ga_l.NB, ga_l.ROW_D, ga_l.LP, ga_l.NROWS, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK");
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
