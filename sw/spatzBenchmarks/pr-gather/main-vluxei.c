// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// pr-gather, baseline arm. Identical data path to main-vlxblk.c; only the
// kernel differs: the SCALAR pull loop (kernel/pr-scalar.c), because the
// vluxei32 element-vector translation hangs on the first missing gather
// (erratum #3). The file/target keep the historical `vluxei` name so the
// batch scripts' target name pr-gather-vluxei-n<nv> is unchanged.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/pr-scalar.c"

#include "bench_fill.h"

// Contribution fill: exact dyadic head (multiples of 2^-16 in
// (0, 2^-5], prime-multiplier scramble), then vector-tiled to the full
// table. Mirrored bit-exactly in gen_data.py.
static void fill_contrib(double *contrib, const unsigned int n,
                         const unsigned int head) {
  const unsigned int h = n < head ? n : head;
  for (unsigned int i = 0; i < h; ++i)
    contrib[i] = (double)(((i * 211u) & 2047u) + 1u) / 65536.0;
  if (n > h)
    bench_fill_rep(contrib, n * 8u, h * 8u);
}

// Neighbor ids: multiplicative hash in u16 lanes masked to NV (pow2),
// vector-generated 128 ids per step from the seed vector 0..127:
// nbr[i] = ((i * 0x9E3779B1) mod 2^16) & (NV - 1). Mirrored in
// gen_data.py (n is a multiple of 128 and NV a power of two by
// construction there).
static void fill_nbr(uint16_t *nbr, const uint16_t *seed,
                     const unsigned int n, const unsigned int mask) {
  for (unsigned int c = 0; c < n; c += 128) {
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(128u));
    asm volatile("vle16.v v8, (%0)" ::"r"(seed) : "memory");
    asm volatile("vadd.vx v8, v8, %0" ::"r"(c));
    asm volatile("vmul.vx v8, v8, %0" ::"r"(2654435761u));
    asm volatile("vand.vx v8, v8, %0" ::"r"(mask));
    asm volatile("vse16.v v8, (%0)" ::"r"(nbr + c) : "memory");
  }
}

// Verify all NACT outputs against the float64 sequential sums from
// gen_data.py. The scalar arm sums in the same order as the generator;
// the shared 1e-9 relative tolerance (kept identical to the vlxblk arm,
// whose vfredosum association differs in final ULPs) still detects a
// single wrong index, which perturbs the sum by >= ~6e-5 relative.
int verify_output(const double *out, const double *expected,
                  const unsigned int n) {
  for (unsigned int v = 0; v < n; ++v) {
    const double got = out[v];
    const double exp = expected[v];
    double err = got - exp;
    if (err < 0)
      err = -err;
    const double mag = exp < 0 ? -exp : exp;
    if (err > 1e-9 * mag + 1e-15) {
      printf("FAILED v=%u got=%f exp=%f\n", v, got, exp);
      return v == 0 ? -1 : (int)v;
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
    fill_contrib(pr_contrib, pr_l.NV, pr_l.HEAD);
    fill_nbr(pr_nbr, pr_seed16, pr_l.NACT * pr_l.DEG, pr_l.NV - 1u);
    bench_fill_zero(pr_out, sizeof(pr_out));

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    pr_scalar(pr_out, pr_contrib, pr_nbr, pr_l.NACT, pr_l.DEG, pr_base,
              pr_damp);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(pr_out, pr_expected, pr_l.NACT);

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
