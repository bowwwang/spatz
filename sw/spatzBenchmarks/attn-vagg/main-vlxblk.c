// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// attn-vagg, VLXBLK arm. Data (top-K indices, scores, expected output)
// from the generated DATAHEADER; the V pool filled on-core from the
// closed-form pattern the generator mirrors (see script/gen_data.py).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/attn-vlxblk.c"

#include "bench_fill.h"

// Hardware fp16 -> float (the toolchain's software cast is broken).
static inline float f16_to_f32(const __fp16 *p) {
  float h, v;
  asm volatile("flh %0, 0(%1)" : "=f"(h) : "r"(p));
  asm volatile("fcvt.s.h %0, %1" : "=f"(v) : "f"(h));
  return v;
}

// V-pool fill: exact dyadic head (multiples of 2^-10 in [-0.5, 0.5)) from
// an fmix32-style u32 mixer, then vector-tiled to the full 4 MiB pool.
// Mirrored bit-exactly in gen_data.py. The mixer is nonlinear in i on
// purpose: with a linear head two row patterns differ by a near-constant
// per lane (as small as 32/1024) and a wrong gathered row would hide under
// the verifier's slack.
static void fill_pool(__fp16 *pool, const unsigned int n,
                      const unsigned int head) {
  const unsigned int h = n < head ? n : head;
  for (unsigned int i = 0; i < h; ++i) {
    uint32_t x = i * 2654435761u;
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    pool[i] = (__fp16)((float)((int)((x >> 22) & 1023u) - 512) / 1024.0f);
  }
  if (n > h)
    bench_fill_rep(pool, n * 2u, h * 2u);
}

// Verify all NQ*HD outputs against the bit-exact fp16 emulation from
// gen_data.py. Tolerance = a few fp16 ulp (2^-9 relative + 2^-9 abs; ~3 ulp
// at |exp| ~ 2): ONE wrong gathered row perturbs lane d by p * |dx| and is
// flagged on >= 64 of the 128 lanes for every pair of row patterns.
int verify_output(const __fp16 *out, const float *expected,
                  const unsigned int n) {
  for (unsigned int i = 0; i < n; ++i) {
    float got = f16_to_f32(out + i);
    float exp = expected[i];
    float err = got > exp ? got - exp : exp - got;
    float mag = exp < 0 ? -exp : exp;
    if (err > 0.002f + 0.002f * mag) {
      printf("FAILED i=%u got=%f exp=%f\n", i, got, exp);
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
    fill_pool(attn_pool, attn_l.NTOK * attn_l.HD, attn_l.HEAD);
    bench_fill_zero(attn_out, sizeof(attn_out));

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    attn_vlxblk(attn_out, attn_pool, attn_idx, attn_p, attn_l.NQ, attn_l.TOPK,
                attn_l.HD);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(attn_out, attn_expected, attn_l.NQ * attn_l.HD);

#ifdef PRINT_RESULT
    // MACs = NQ * TOPK * HD (fp16)
    printf("attn-vagg vlxblk nq=%u topk=%u hd=%u ntok=%u cache=%d: took %u "
           "cycles %s (macs=%u)\n",
           attn_l.NQ, attn_l.TOPK, attn_l.HD, attn_l.NTOK, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK",
           attn_l.NQ * attn_l.TOPK * attn_l.HD);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
