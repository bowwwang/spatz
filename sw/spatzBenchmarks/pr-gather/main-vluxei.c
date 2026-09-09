// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// pr-gather, baseline arm (scalar gather; the batch vluxei32 baseline hangs
// on the first missing gather, erratum #3, and misses are intrinsic here -
// the target keeps its historical `vluxei` name). All data is literal in
// the generated DATAHEADER (script/gen_data.py); nothing is generated
// on-core.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/pr-scalar.c"

// Compare outputs against the generated expectation in integer ULPs
// (fp64 bit patterns read with integer loads). 2^20 ULPs ~ 2.3e-10
// relative absorbs the association difference between the scalar arm
// (base + damp * sum) and the generator's order (base + sum(damp * c));
// one wrong index perturbs a sum by >= ~1e-2 relative.
static int verify_output(const double *out, const uint64_t *expected_bits,
                         const unsigned int n) {
  const uint32_t *ow = (const uint32_t *)out;
  const uint32_t *ew = (const uint32_t *)expected_bits;
  const int64_t tol = (int64_t)1 << 20;
  unsigned int fails = 0;
  for (unsigned int v = 0; v < n; ++v) {
    const uint64_t got = ((uint64_t)ow[2 * v + 1] << 32) | ow[2 * v];
    const uint64_t exp = ((uint64_t)ew[2 * v + 1] << 32) | ew[2 * v];
    int64_t d = (int64_t)(got - exp);
    if (d < 0)
      d = -d;
    if (d > tol) {
      if (fails < 8)
        printf("FAILED v=%u got=0x%08x%08x exp=0x%08x%08x\n", v,
               (unsigned)(got >> 32), (unsigned)got, (unsigned)(exp >> 32),
               (unsigned)exp);
      fails++;
    }
  }
  if (fails)
    printf("FAILED total=%u/%u\n", fails, n);
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

    pr_scalar(pr_out, (const double *)pr_contrib_bits, pr_nbr, pr_l.NACT,
              pr_l.DEG, pr_seed[0], pr_damp);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(pr_out, pr_expected_bits, pr_l.NACT);

#ifdef PRINT_RESULT
    printf("pr-gather scalar nv=%u nact=%u deg=%u cache=%d: took %u cycles "
           "%s (adds=%u)\n",
           pr_l.NV, pr_l.NACT, pr_l.DEG, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK", pr_l.NACT * pr_l.DEG);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
