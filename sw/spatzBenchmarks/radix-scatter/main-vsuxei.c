// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// radix-scatter, plain-RVV baseline arm (vsoxei32, column decomposition).
// Identical data path to main-vsxblk.c plus the column-major staging of
// the source payload that the baseline reads (erratum #2 workaround, see
// kernel/radix-vsoxei.c); only the kernel differs. The file keeps the
// historical name main-vsuxei.c so the target stays
// radix-scatter-vsuxei-n<NREC>.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/radix-vsoxei.c"

#include "bench_fill.h"

// Column-major staging of the SAME payload for the baseline's source
// loads (built untimed): strided vector loads corrupt one lane under
// cache misses on this port (erratum #2, rec=18560 got=elem1-value;
// identical for vsuxei32 and vsoxei32 -> the scatter is innocent).
// Baseline arm only (same size as rs_src, declared after the header's
// buffers so the .data layout of the shared arrays is unchanged).
static uint32_t rs_srcT[sizeof(rs_src) / sizeof(rs_src[0])]
    __attribute__((section(".data"), aligned(128)));

// Slot fill: synthetic keys assign records round-robin to FANOUT
// partitions (bucket = r mod F, the canonical partition interleaving), so
// the prefix-sum offsets collapse to slot(r) = bucket * (NREC/F) + r/F.
// Generated vectorized (scalar datagen at this scale dominates sim
// wall-clock); mirrored exactly in gen_data.py.
// NOTE m4: flamingo is VLEN=1024, but e16 m1 VLMAX is 64 -- an m1 request
// of 128 silently clamps and leaves half of each chunk zero (the rec=0
// CHECK-FAIL of 2026-09-07).
static void fill_slots(uint16_t *slot, const uint16_t *seed,
                       const unsigned int nrec, const unsigned int fanout_log2) {
  const unsigned int nrec_log2 = 31u - (unsigned int)__builtin_clz(nrec);
  const unsigned int fmask = (1u << fanout_log2) - 1u;
  const unsigned int wsh = nrec_log2 - fanout_log2;
  for (unsigned int c = 0; c < nrec; c += 128) {
    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" ::"r"(128u));
    asm volatile("vle16.v v8, (%0)" ::"r"(seed) : "memory");
    asm volatile("vadd.vx v8, v8, %0" ::"r"(c));           // r
    asm volatile("vand.vx v12, v8, %0" ::"r"(fmask));      // bucket = r & (F-1)
    asm volatile("vsll.vx v12, v12, %0" ::"r"(wsh));       // bucket * (NREC/F)
    asm volatile("vsrl.vx v8, v8, %0" ::"r"(fanout_log2)); // r / F
    asm volatile("vor.vv v8, v8, v12");
    asm volatile("vse16.v v8, (%0)" ::"r"(slot + c) : "memory");
  }
}

// Source records: src[i] = i (unique words), vectorized.
static void fill_src(uint32_t *src, const uint32_t *seed,
                     const unsigned int n_words) {
  for (unsigned int c = 0; c < n_words; c += 128) {
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(128u));
    asm volatile("vle32.v v8, (%0)" ::"r"(seed) : "memory");
    asm volatile("vadd.vx v8, v8, %0" ::"r"(c));
    asm volatile("vse32.v v8, (%0)" ::"r"(src + c) : "memory");
  }
}

// Column staging: srcT[d * nrec + r] = 4r + d = src[r * 4 + d] (the
// vsll.vi by 2 is log2(RD) for the 16-B records both kernels are built
// around), vectorized per column.
static void fill_srcT(uint32_t *srcT, const uint32_t *seed,
                      const unsigned int nrec, const unsigned int rd) {
  for (unsigned int d = 0; d < rd; ++d) {
    for (unsigned int c = 0; c < nrec; c += 128) {
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(128u));
      asm volatile("vle32.v v8, (%0)" ::"r"(seed) : "memory");
      asm volatile("vadd.vx v8, v8, %0" ::"r"(c));
      asm volatile("vsll.vi v8, v8, 2");
      asm volatile("vadd.vx v8, v8, %0" ::"r"(d)); // value = 4r + d
      asm volatile("vse32.v v8, (%0)" ::"r"(srcT + d * nrec + c) : "memory");
    }
  }
}

// Exact check of the record image at the slot the generator assigned,
// SAMPLED every 4th record + the last (16,385 records = 65,540 words):
// sampling bounds the scalar-core reference cost; a structural scatter
// bug hits sampled records too. slot(r) is recomputed from the closed
// form, so the check also covers the on-core slot fill.
int verify_output(const uint32_t *dst, const unsigned int nrec,
                  const unsigned int rd, const unsigned int fanout_log2) {
  const unsigned int fanout = 1u << fanout_log2;
  const unsigned int run = nrec / fanout; // records per bucket
  for (unsigned int s = 0; s <= nrec / 4; ++s) {
    const unsigned int r = (s == nrec / 4) ? (nrec - 1) : (s * 4);
    const unsigned int slot = (r & (fanout - 1u)) * run + r / fanout;
    for (unsigned int d = 0; d < rd; ++d) {
      const uint32_t got = dst[slot * rd + d];
      const uint32_t exp = r * rd + d;
      if (got != exp) {
        printf("FAILED rec=%u elem=%u slot=%u got=0x%x exp=0x%x\n", r, d, slot,
               (unsigned)got, (unsigned)exp);
        return r == 0 ? -1 : (int)r;
      }
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
    // UNTIMED data build, identical for both arms (+ the column staging)
    fill_slots(rs_slot, rs_seed16, rs_l.NREC, rs_l.FANOUT_LOG2);
    fill_src(rs_src, rs_seed32, rs_l.NREC * rs_l.RD);
    fill_srcT(rs_srcT, rs_seed32, rs_l.NREC, rs_l.RD);
    bench_fill_zero(rs_dst, sizeof(rs_dst));

#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    // Start timer
    timer = benchmark_get_cycle();

    scatter_vsoxei(rs_dst, rs_srcT, rs_slot, rs_l.NREC);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(rs_dst, rs_l.NREC, rs_l.RD, rs_l.FANOUT_LOG2);

#ifdef PRINT_RESULT
    printf("radix-scatter vsoxei nrec=%u fanout=%u cache=%d: took %u cycles "
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
