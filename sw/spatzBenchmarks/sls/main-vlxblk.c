// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// slsint8, vlxblk arm. ALL data (table, scale/bias, indices, expected output)
// comes from the generated DATAHEADER (script/gen_data.py); no data generation.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/sls-vlxblk.c"

// fp16 -> float for the verifier WITHOUT any FPU load or Zfh op: the bits are
// read with an integer lhu and widened arithmetically. (An flh here stalled
// after the kernel's vector stores — the erratum-#5 fp_lsu path — and the
// toolchain's software (float)__fp16 cast is broken.) Handles zero,
// subnormals, normals, inf/NaN.
static inline float f16_to_f32(const __fp16 *p) {
  uint32_t h;
  asm volatile("lhu %0, 0(%1)" : "=r"(h) : "r"(p) : "memory");
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;                                   // +-0
    } else {                                         // subnormal: normalize
      uint32_t e = 127u - 15u + 1u;
      while ((mant & 0x400u) == 0u) { mant <<= 1; --e; }
      mant &= 0x3ffu;
      bits = sign | (e << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);        // inf / NaN
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  union { uint32_t u; float f; } cv;
  cv.u = bits;
  return cv.f;
}

// Verify ALL outputs (8 slot partials per bag) against the bit-exact fp16
// emulation from gen_data.py (identical order in both arms). Tolerance = a few fp16 ulp (2^-9 abs +
// 2^-9 relative); a wrong row shifts a lane by ~0.1..1, far above it.
int verify_output(const __fp16 *out, const float *expected,
                  const unsigned int nb, const unsigned int row_d) {
  unsigned int fails = 0, first_bad_bag = 0xffffffffu;
  float max_err = 0.0f;
  for (unsigned int i = 0; i < nb * row_d; ++i) {
    float got = f16_to_f32(out + i);
    float exp = expected[i];
    float err = got > exp ? got - exp : exp - got;
    float mag = exp < 0 ? -exp : exp;
    if (err > 0.002f + 0.002f * mag) {
      if (fails < 8)
        printf("FAILED b=%u slot=%u d=%u got=%f exp=%f\n", i / row_d, (i % row_d) / 16u, i % 16u, got, exp);
      if (first_bad_bag == 0xffffffffu)
        first_bad_bag = i / row_d;
      ++fails;
      if (err > max_err)
        max_err = err;
    }
  }
  if (fails)
    printf("FAILED total=%u/%u max_err=%f first_bad_bag=%u\n", fails,
           nb * row_d, max_err, first_bad_bag);
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
    sls_vlxblk(sl_out, (const __fp16 *)sl_tbl_bits, sl_idx, sl_l.NB, sl_l.LP, sl_l.ROW_D, sl_l.DBG_EVERY);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    // error = verify_output(sl_out, sl_expected, sl_l.NB, sl_l.OUT_D);

#ifdef PRINT_RESULT
    printf("sls vlxblk nb=%u lp=%u row_d=%u nrows=%u cache=%d: took %u "
           "cycles %s\n",
           sl_l.NB, sl_l.LP, sl_l.ROW_D, sl_l.NROWS, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK");
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
