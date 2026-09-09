// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// radix-scatter, VSXBLK arm. All data (source records, partition slots,
// zero destination) is literal in the generated DATAHEADER
// (script/gen_data.py); nothing is generated on-core. Pure data movement:
// no FLOPs; report bytes moved per cycle.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/radix-vsxblk.c"

// Exact check of the record image at the slot the generator assigned,
// SAMPLED every 4th record + the last (all RD words of each): sampling
// bounds the scalar-core reference cost; a structural scatter bug hits
// sampled records too. Integer loads only.
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
    timer = benchmark_get_cycle();

    scatter_vsxblk(rs_dst, rs_src, rs_slot, rs_l.NREC);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(rs_dst, rs_src, rs_slot, rs_l.NREC, rs_l.RD);

#ifdef PRINT_RESULT
    printf("radix-scatter vsxblk nrec=%u fanout=%u cache=%d: took %u cycles "
           "%s (bytes=%u)\n",
           rs_l.NREC, 1u << rs_l.FANOUT_LOG2, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK", rs_l.NREC * rs_l.RD * 4u);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
