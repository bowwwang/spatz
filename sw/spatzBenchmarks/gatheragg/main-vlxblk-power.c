// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// gatheragg (gnnagg), VLXBLK arm. All data (feature table, ids, expected
// output) is literal in the generated DATAHEADER (script/gen_data.py);
// nothing is generated on-core.

// POWER-SIMULATION BUILD (2026-09-10). Same kernel, same data geometry -
// block size, table/footprint and the per-iteration work are untouched; only
// the number of repeated iterations is reduced so a power run stays short.
// The timed region is bracketed by start_kernel()/stop_kernel() for the power
// tooling, and the result check is disabled (correctness is measured by the
// normal targets).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/gatheragg-vlxblk.c"

// Exact fp32 bit compare of every output lane against the bit-exact
// emulation from gen_data.py (integer loads only: no FPU loads after the
// kernel's vector stores).
static __attribute__((unused)) static int verify_output(const float *out, const uint32_t *expected_bits,
                         const unsigned int n, const unsigned int row_d) {
  const uint32_t *ow = (const uint32_t *)out;
  unsigned int fails = 0;
  for (unsigned int i = 0; i < n; ++i) {
    if (ow[i] != expected_bits[i]) {
      if (fails < 8)
        printf("FAILED b=%u d=%u got=0x%08x exp=0x%08x\n", i / row_d, i % row_d,
               (unsigned)ow[i], (unsigned)expected_bits[i]);
      fails++;
    }
  }
  if (fails)
    printf("FAILED total=%u/%u\n", fails, n);
  return fails > 255 ? 255 : (int)fails;
}

int main() {
  const unsigned int PWR_NB = 16u; // nodes (table stays 4 MiB, 25 neighbours per node)
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
    // Start timer
    start_kernel();
    timer = benchmark_get_cycle();

    agg_vlxblk(ga_out, (const float *)ga_tbl_bits, ga_idx, PWR_NB, ga_l.ROW_D,
               ga_l.LP);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;
    stop_kernel();

    // Result check REMOVED (user ruling 2026-09-10: the optimized kernel
    // mismatches and we do not care - this row is timing only). The
    // mismatch is consistent with an unenforced WAR on the gather's INDEX
    // register: the schedule overwrites v2/v3 two instructions after the
    // vlxblkei16 that reads them (erratum #7 on the index operand rather
    // than the destination group).
    (void)verify_output;

#ifdef PRINT_RESULT
    printf("gatheragg vlxblk nb=%u row_d=%u lp=%u nrows=%u cache=%d: took %u "
           "cycles %s\n",
           PWR_NB, ga_l.ROW_D, ga_l.LP, ga_l.NROWS, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-SKIPPED");
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
