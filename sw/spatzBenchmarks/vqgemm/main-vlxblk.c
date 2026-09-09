// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// vqgemm, VLXBLK arm. Data (A matrix, indices, scales, expected output)
// from the generated DATAHEADER (script/gen_data.py); this file contains
// no data generation.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/vqgemm-vlxblk.c"

// Hardware fp16 -> float (the toolchain's software cast is broken).
static inline float f16_to_f32(const __fp16 *p) {
  float h, v;
  asm volatile("flh %0, 0(%1)" : "=f"(h) : "r"(p));
  asm volatile("fcvt.s.h %0, %1" : "=f"(v) : "f"(h));
  return v;
}

// Verify all M*N outputs against the bit-exact fp16 emulation from
// gen_data.py. Tolerance = a few fp16 ulp (2^-10 relative + 2^-9 abs):
// one wrong gathered entry perturbs a lane by >= ~0.01, far above it.
int verify_output(const __fp16 *c, const float *expected,
                  const unsigned int M, const unsigned int N) {
  unsigned int fails = 0, first_bad_tile = 0xffffffffu;
  float max_err = 0.0f;
  for (unsigned int m = 0; m < M; ++m)
    for (unsigned int n = 0; n < N; ++n) {
      float got = f16_to_f32(c + m * N + n);
      float exp = expected[m * N + n];
      float err = got > exp ? got - exp : exp - got;
      float mag = exp < 0 ? -exp : exp;
      if (err > 0.002f + 0.002f * mag) {
        if (fails < 8)
          printf("FAILED m=%u n=%u got=%f exp=%f\n", m, n, got, exp);
        if (first_bad_tile == 0xffffffffu)
          first_bad_tile = m / 8;
        ++fails;
        if (err > max_err)
          max_err = err;
      }
    }
  if (fails)
    printf("FAILED total=%u/%u max_err=%f first_bad_tile=%u\n", fails, M * N,
           max_err, first_bad_tile);
  return (int)(fails > 255 ? 255 : fails);
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

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    if (vq_l.IDX_BYTES == 1) {
      vqgemm_vlxblk_ei8(vq_c, vq_a, vq_cb0, vq_cb1, (const uint8_t *)vq_idx0,
                        (const uint8_t *)vq_idx1, vq_scales, vq_l.M, vq_l.N,
                        vq_l.K, vq_l.CB_D);
    } else {
      vqgemm_vlxblk_ei16(vq_c, vq_a, vq_cb0, vq_cb1,
                         (const uint16_t *)vq_idx0, (const uint16_t *)vq_idx1,
                         vq_scales, vq_l.M, vq_l.N, vq_l.K, vq_l.CB_D);
    }
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(vq_c, vq_expected, vq_l.M, vq_l.N);

#ifdef PRINT_RESULT
    printf("vqgemm vlxblk M=%u N=%u K=%u CB_D=%u CBN=%u cache=%d: took %u "
           "cycles %s (macs=%u)\n",
           vq_l.M, vq_l.N, vq_l.K, vq_l.CB_D, vq_l.CBN, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK", vq_l.M * vq_l.N * vq_l.K);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
