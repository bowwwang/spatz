// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, VLXBLK arm. Pair list and expected forces from the generated
// DATAHEADER; the four field arrays (x, y, z, q; 380 KiB each) are filled
// on-core from the closed-form pattern the generator mirrors (see
// script/gen_data.py).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/nbforce-vlxblk.c"

#include "bench_fill.h"

// Field fill: coordinates/charges in [0, 4) as exact multiples of 2^-8
// (prime-multiplier pattern head of HEAD f32, then vector-tiled to the
// full array). Mirrored bit-exactly in gen_data.py.
static void fill_field(float *f, const unsigned int n, const unsigned int head,
                       const unsigned int mult) {
  const unsigned int h = n < head ? n : head;
  for (unsigned int i = 0; i < h; ++i)
    f[i] = 0.00390625f * (float)((i * mult) & 1023u);
  if (n > h)
    bench_fill_rep(f, n * 4u, h * 4u);
}

// Verify ALL NC_TILE x 4 atoms x 3 components against the float64
// reference from gen_data.py. Tolerance 1% + 0.01 abs: fp32 accumulation
// over LIST*4 j-atoms in vfredusum tree order, with cutoff clamps.
int verify_output(const float *f, const float *expected,
                  const unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    float got = f[i];
    float exp = expected[i];
    float err = got > exp ? got - exp : exp - got;
    float mag = exp < 0 ? -exp : exp;
    if (err > 0.01f + 0.01f * mag) {
      printf("FAILED c=%u a=%u d=%u got=%f exp=%f\n", i / 12, (i % 12) / 3,
             i % 3, got, exp);
      return i == 0 ? -1 : (int)i;
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
    const unsigned int n_atoms = nb_l.NC_DOM * 4u;
    fill_field(nb_x, n_atoms, nb_l.HEAD, 37u);
    fill_field(nb_y, n_atoms, nb_l.HEAD, 53u);
    fill_field(nb_z, n_atoms, nb_l.HEAD, 71u);
    fill_field(nb_q, n_atoms, nb_l.HEAD, 89u);
    bench_fill_zero(nb_f, sizeof(nb_f));

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    nbforce_vlxblk(nb_f, nb_x, nb_y, nb_z, nb_q, nb_list, nb_l.NC_TILE,
                   nb_l.LIST, nb_l.CUT2);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(nb_f, nb_expected, nb_l.NC_TILE * 12u);

#ifdef PRINT_RESULT
    printf("nbforce vlxblk nc=%u list=%u cache=%d: took %u cycles %s "
           "(pairs=%u)\n",
           nb_l.NC_TILE, nb_l.LIST, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK", nb_l.NC_TILE * nb_l.LIST);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
