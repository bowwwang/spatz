// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// Fast on-core data generation for RTL-simulated benchmarks. Scalar
// datagen at MiB scale dominates simulation wall-clock (~1K cycles/s),
// so tables are filled by writing a pattern head with scalar code and
// replicating it with doubling vector self-copies.
// Head length = BF_HEAD_BYTES = 64 * 61 bytes: every copy source and
// destination stays 64-B aligned (misaligned vector accesses hang this
// cache port — the wip misalignment path), while the prime factor 61
// keeps wrong-index gathers detectable: for every pow2 block size a
// wrong index aliases only at multiples of 61 blocks.
// Copies run at e32 (the most-exercised width on this core); byte
// counts must be multiples of 4.
//
// Both helpers clobber v8-v15; call only before the timed kernel.

#ifndef BENCH_FILL_H
#define BENCH_FILL_H

#include <stdint.h>

#define BF_HEAD_BYTES (64u * 61u) // 3,904 B; per-type element counts:
                                  // u8 3904 / f16 1952 / f32 976 / f64 488

// Replicate the already-written head buf[0..head_bytes) over
// buf[head_bytes..n_bytes) with doubling vector copies (period stays
// head_bytes).
static inline void bench_fill_rep(void *buf, uint32_t n_bytes,
                                  uint32_t head_bytes) {
  uint32_t *a = (uint32_t *)buf;
  uint32_t filled = head_bytes / 4u; // words
  const uint32_t n = n_bytes / 4u;
  while (filled < n) {
    uint32_t chunk = filled; // copy [0, filled) -> [filled, filled+chunk)
    if (chunk > n - filled)
      chunk = n - filled;
    const uint32_t *s = a;
    uint32_t *d = a + filled;
    uint32_t c = chunk;
    // Vector copies only while >= 8 words (32 B): a vector store of
    // vl*4 < 32 B leaves cache ports idle and HANGS (erratum #1).
    while (c >= 8u) {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(c));
      asm volatile("vle32.v v8, (%0)" ::"r"(s)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
      asm volatile("vse32.v v8, (%0)" ::"r"(d) : "memory");
      s += vl;
      d += vl;
      c -= vl;
    }
    while (c--) // 1-7 word tail: scalar (avoids the sub-32-B vse)
      *d++ = *s++;
    filled += chunk;
  }
}

// Vector zero-fill (replaces scalar memset on MiB buffers).
static inline void bench_fill_zero(void *buf, uint32_t n_bytes) {
  uint32_t *d = (uint32_t *)buf;
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(128u));
  asm volatile("vmv.v.i v8, 0");
  uint32_t c = n_bytes / 4u;
  while (c >= 8u) { // vector stores only while >= 32 B (erratum #1)
    uint32_t vl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(c));
    asm volatile("vse32.v v8, (%0)" ::"r"(d) : "memory");
    d += vl;
    c -= vl;
  }
  while (c--) // 1-7 word tail: scalar
    *d++ = 0u;
}

#endif // BENCH_FILL_H
