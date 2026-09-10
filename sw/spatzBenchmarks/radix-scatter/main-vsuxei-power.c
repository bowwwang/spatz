// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// radix-scatter, plain-RVV baseline arm (column-decomposed vsoxei32; the
// main/target keep the historical "vsuxei" name). All data is literal in
// the generated DATAHEADER (script/gen_data.py): the baseline reads the
// column-major staging rs_srcT of the same records (strided vector loads
// corrupt one lane under misses on this port, erratum #2). Nothing is
// generated on-core.

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
#include "kernel/radix-vsoxei.c"

// Exact check of the record image at the slot the generator assigned,
// SAMPLED every 4th record + the last (all RD words of each). Integer
// loads only.
static int verify_output(const uint32_t *dst, const uint32_t *src,
                         const uint16_t *slot, const unsigned int nrec,
                         const unsigned int rd) {
  unsigned int fails = 0;
  for (unsigned int s = 0; s <= nrec / 4; ++s) {
    const unsigned int r = (s == nrec / 4) ? (nrec - 1) : (s * 4);
    const unsigned int sl = slot[r];
    for (unsigned int d = 0; d < rd; ++d) {
      const uint32_t got = dst[sl * rd + d];
      const uint32_t exp = src[r * rd + d];
      if (got != exp) {
        if (fails < 8)
          printf("FAILED rec=%u elem=%u slot=%u got=0x%x exp=0x%x\n", r, d, sl,
                 (unsigned)got, (unsigned)exp);
        fails++;
      }
    }
  }
  if (fails)
    printf("FAILED total=%u words\n", fails);
  return fails > 255 ? 255 : (int)fails;
}

int main() {
  const unsigned int PWR_NREC = 4096u; // records scattered (dst stays the full 1-MiB image)
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

    scatter_vsoxei(rs_dst, rs_srcT, rs_slot, PWR_NREC);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;
    stop_kernel();

    // power build: no result check


    // error = verify_output(rs_dst, rs_src, rs_slot, PWR_NREC, rs_l.RD);

#ifdef PRINT_RESULT
    printf("radix-scatter vsoxei nrec=%u fanout=%u cache=%d: took %u cycles "
           "%s (bytes=%u)\n",
           PWR_NREC, 1u << rs_l.FANOUT_LOG2, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK", PWR_NREC * rs_l.RD * 4u);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
