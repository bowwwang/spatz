// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, plain-RVV baseline arm. Identical data path to main-vlxblk.c
// plus the untimed expansion of the u16 pair list into the per-element
// u32 index array the vluxei32 baseline consumes.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/nbforce-rvv.c"

#include "bench_fill.h"

// Expanded per-element index array (value = cluster*4 + lane; the kernel
// shifts <<2 to byte offsets). rvv arm only, hence declared here and not
// in the shared data header; sized from the literal pair list.
static uint32_t nb_list_exp[sizeof(nb_list) / sizeof(nb_list[0]) * 4u]
    __attribute__((section(".data"), aligned(128)));

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

// Untimed: u16 cluster ids -> 4 u32 element indices per pair.
static void expand_pairlist(uint32_t *exp, const uint16_t *list,
                            const unsigned int n_pairs) {
  for (unsigned int e = 0; e < n_pairs; ++e) {
    const uint32_t base = (uint32_t)list[e] * 4u;
    exp[4 * e + 0] = base + 0;
    exp[4 * e + 1] = base + 1;
    exp[4 * e + 2] = base + 2;
    exp[4 * e + 3] = base + 3;
  }
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
    expand_pairlist(nb_list_exp, nb_list, nb_l.NC_TILE * nb_l.LIST);
    bench_fill_zero(nb_f, sizeof(nb_f));

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    nbforce_rvv(nb_f, nb_x, nb_y, nb_z, nb_q, nb_list_exp, nb_l.NC_TILE,
                nb_l.LIST, nb_l.CUT2);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(nb_f, nb_expected, nb_l.NC_TILE * 12u);

#ifdef PRINT_RESULT
    printf("nbforce rvv nc=%u list=%u cache=%d: took %u cycles %s "
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
