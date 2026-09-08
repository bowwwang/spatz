// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// vqgemv, plain-RVV baseline arm. Identical data path to main-vlxblk.c;
// only the kernel differs.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/vqgemv-rvv.c"

// Hardware fp16 -> float (the toolchain's software cast is broken).
static inline float f16_to_f32(const __fp16 *p) {
  float h, v;
  asm volatile("flh %0, 0(%1)" : "=f"(h) : "r"(p));
  asm volatile("fcvt.s.h %0, %1" : "=f"(v) : "f"(h));
  return v;
}

// Verify all N outputs against the bit-exact fp16 emulation from
// gen_data.py. Tolerance = a few fp16 ulp (2^-10 relative + 2^-9 abs):
// one wrong gathered entry perturbs a lane by >= ~0.01, far above it.
int verify_output(const __fp16 *c, const float *expected,
                  const unsigned int N) {
  unsigned int fails = 0;
  float max_err = 0.0f;
  for (unsigned int n = 0; n < N; ++n) {
    float got = f16_to_f32(c + n);
    float exp = expected[n];
    float err = got > exp ? got - exp : exp - got;
    float mag = exp < 0 ? -exp : exp;
    if (err > 0.002f + 0.002f * mag) {
      if (fails < 8)
        printf("FAILED n=%u got=%f exp=%f\n", n, got, exp);
      ++fails;
      if (err > max_err)
        max_err = err;
    }
  }
  if (fails)
    printf("FAILED total=%u/%u max_err=%f\n", fails, N, max_err);
  return (int)fails;
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

    if (vq_l.CB_D == 8) {
      vqgemv_rvv_d8(vq_c, vq_a, vq_cb0, vq_cb1, (const uint8_t *)vq_idx0,
                    (const uint8_t *)vq_idx1, vq_scales, vq_l.K, vq_l.N);
    } else {
      vqgemv_rvv_d16(vq_c, vq_a, vq_cb0, vq_cb1, (const uint16_t *)vq_idx0,
                     (const uint16_t *)vq_idx1, vq_scales, vq_l.K, vq_l.N);
    }
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(vq_c, vq_expected, vq_l.N);

#ifdef PRINT_RESULT
    printf("vqgemv rvv K=%u N=%u CB_D=%u CBN=%u cache=%d: took %u cycles "
           "%s (macs=%u)\n",
           vq_l.K, vq_l.N, vq_l.CB_D, vq_l.CBN, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK", vq_l.K * vq_l.N);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
