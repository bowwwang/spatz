// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// radix-scatter: DB radix-partitioning record scatter (Polychroniou,
// Raghavan & Ross, SIGMOD'15): fixed 16-B records scattered to their
// partition slots. Slots come from an actual radix partition of
// synthetic keys (FANOUT buckets, histogram + prefix sum, computed
// UNTIMED and identically for both arms): within a bucket writes are
// sequential — the real locality structure (4 records per 64-B line),
// unlike a random permutation. Software today emulates block scatter
// with cache-line buffers + non-temporal stores; vsxblk is the direct
// expression.
//
// Arms (VARIANT): 1 = VSXBLK block scatter (64 records per chunk)
//                 0 = vsuxei32 baseline, column-decomposed: e32 byte
//                     offsets (slot*16 + 4d), one strided source load +
//                     one element scatter per record column d — the
//                     dictdecode-style strong baseline (u16 byte offsets
//                     cannot address the 1 MiB destination)
// Pure data movement: no FLOPs; report bandwidth. Check: exact image.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

// Native VLXBLK mnemonics (LLVM 14 + MC-layer patch); x-register
// form keeps the numeric rs1n interface, so call sites are unchanged.
#define VSXBLKEI16_V(vs3, rs1n, vs2) "vsxblkei16.v v" #vs3 ", (x" #rs1n "), v" #vs2 "\n"
#define VSETBLKLEN(rs1n)             "vsetblklen x" #rs1n "\n"

#ifndef NREC
#define NREC 65536 // records; 16 B each (u16 slot-id ceiling)
#endif
#ifndef VARIANT
#define VARIANT 1
#endif
#ifndef FANOUT_LOG2
#define FANOUT_LOG2 8 // 256 partitions (8 radix bits), SIGMOD'15 regime
#endif

#define RD 4 // record elements (e32) -> 16-B records
#define FANOUT (1u << FANOUT_LOG2)
#if (NREC & (NREC - 1)) != 0 || FANOUT > NREC
#error "NREC must be a power of two and >= FANOUT (closed-form slots)"
#endif

static uint32_t rs_src[NREC * RD] __attribute__((section(".data"), aligned(128)));
static uint32_t rs_dst[NREC * RD] __attribute__((section(".data"), aligned(128)));
static uint16_t rs_slot[NREC + 16] __attribute__((section(".data"), aligned(64)));
static uint16_t rs_seed16[128] __attribute__((section(".data"), aligned(64)));
static uint32_t rs_seed32[128] __attribute__((section(".data"), aligned(128)));
#if VARIANT == 0
// Column-major staging of the SAME payload for the baseline's source
// loads (built untimed): strided vector loads corrupt one lane under
// cache misses on this port (erratum #2, rec=18560 got=elem1-value;
// identical for vsuxei32 and vsoxei32 -> the scatter is innocent).
static uint32_t rs_srcT[NREC * RD] __attribute__((section(".data"), aligned(128)));
#endif

#if VARIANT == 1
static void scatter_vsxblk(uint32_t *dst, const uint32_t *src,
                           const uint16_t *slot, unsigned int n) {
  register uint32_t bl asm("t0") = RD;       // x5
  register uint32_t *dp asm("t1") = dst;     // x6
  asm volatile(VSETBLKLEN(5) :: "r"(bl), "r"(dp));
  for (unsigned int r = 0; r < n; r += 64) { // 64 rec x 4 = 256 = vlmax
    asm volatile("vsetvli zero, %[gr], e16, m1, ta, ma\n"
                 "vle16.v v2, (%[i0])\n"
                 "vsetvli zero, %[ec], e32, m8, ta, ma\n"
                 "vle32.v v8, (%[s0])\n"
                 VSXBLKEI16_V(8, 6, 2)
                 :
                 : [gr] "r"(64u), [ec] "r"(256u), [i0] "r"(slot + r),
                   [s0] "r"(src + r * RD), [dst] "r"(dp)
                 : "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                   "v15", "memory");
  }
}
#else
// Baseline: column decomposition. Per 64-record chunk: widen the u16
// slots once to e32 record byte offsets (vwmulu by 16), then per record
// column d (0..3): offsets+4d, strided source load (stride 16 B), and
// one vsuxei32 element scatter. u16 byte offsets cannot address the
// 1 MiB destination, so e32 offsets are required (2x index bandwidth).
static void scatter_vsuxei(uint32_t *dst, const uint32_t *src,
                           const uint16_t *slot, unsigned int n) {
  for (unsigned int r = 0; r < n; r += 64) {
    asm volatile("vsetvli zero, %[gr], e16, m2, ta, ma\n"
                 "vle16.v v2, (%[i0])\n"
                 "vwmulu.vx v4, v2, %[sixteen]\n" // e32 offsets = slot * 16
                 :
                 : [gr] "r"(64u), [i0] "r"(slot + r), [sixteen] "r"(16u)
                 : "v2", "v3", "v4", "v5", "v6", "v7", "memory");
    for (unsigned int d = 0; d < RD; ++d) {
      asm volatile("vsetvli zero, %[gr], e32, m4, ta, ma\n"
                   "vadd.vx v8, v4, %[eoff]\n"
                   "vle32.v v16, (%[s0])\n" // unit-stride from the column staging
                   "vsoxei32.v v16, (%[d0]), v8\n"
                   :
                   : [gr] "r"(64u), [eoff] "r"(4u * d),
                     [s0] "r"(rs_srcT + d * NREC + r), [d0] "r"(dst)
                   : "v8", "v9", "v10", "v11", "v16", "v17", "v18", "v19",
                     "memory");
    }
  }
}
#endif

int main(void) {
  const unsigned int cid = snrt_cluster_core_idx();
#if USE_CACHE == 1
  uint32_t spm_size = 16;
#else
  uint32_t spm_size = 120;
#endif
  if (cid == 0)
    l1d_init(spm_size);
  snrt_cluster_hw_barrier();

  int fails = 0;
  if (cid == 0) {
    // UNTIMED slot computation, identical for both arms: synthetic keys
    // assign records round-robin to FANOUT partitions (bucket = r mod F,
    // the canonical partition interleaving), so the prefix-sum offsets
    // collapse to the closed form slot(r) = bucket*(NREC/F) + r/F —
    // a permutation with per-bucket sequential runs, recomputable by
    // the checker. Generated vectorized (scalar datagen at this scale
    // dominates sim wall-clock).
    const uint32_t nrec_log2 = 31u - (uint32_t)__builtin_clz((uint32_t)NREC);
    for (unsigned int i = 0; i < 128; ++i) {
      rs_seed16[i] = (uint16_t)i;
      rs_seed32[i] = i;
    }
    // NOTE m4: flamingo is VLEN=1024, but e16 m1 VLMAX is 64 — an m1
    // request of 128 silently clamps and leaves half of each chunk
    // zero (the rec=0 CHECK-FAIL of 2026-09-07).
    for (unsigned int c = 0; c < NREC; c += 128) {
      asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                   "vle16.v v8, (%[seed])\n"
                   "vadd.vx v8, v8, %[c]\n"     // r
                   "vand.vx v12, v8, %[fm]\n"   // bucket = r & (F-1)
                   "vsll.vx v12, v12, %[wsh]\n" // bucket * (NREC/F)
                   "vsrl.vx v8, v8, %[fsh]\n"   // r / F
                   "vor.vv v8, v8, v12\n"
                   "vse16.v v8, (%[dst])\n"
                   :
                   : [n] "r"(128u), [seed] "r"(rs_seed16), [c] "r"(c),
                     [fm] "r"(FANOUT - 1u), [wsh] "r"(nrec_log2 - FANOUT_LOG2),
                     [fsh] "r"((uint32_t)FANOUT_LOG2), [dst] "r"(rs_slot + c)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
    }
    // src records: rs_src[i] = i (unique words), vectorized; dst zeroed.
    for (unsigned int c = 0; c < NREC * RD; c += 128) {
      asm volatile("vsetvli zero, %[n], e32, m8, ta, ma\n"
                   "vle32.v v8, (%[seed])\n"
                   "vadd.vx v8, v8, %[c]\n"
                   "vse32.v v8, (%[dst])\n"
                   :
                   : [n] "r"(128u), [seed] "r"(rs_seed32), [c] "r"(c),
                     [dst] "r"(rs_src + c)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "memory");
    }
#if VARIANT == 0
    for (unsigned int d = 0; d < RD; ++d)
      for (unsigned int c = 0; c < NREC; c += 128) {
        asm volatile("vsetvli zero, %[n], e32, m8, ta, ma\n"
                     "vle32.v v8, (%[seed])\n"
                     "vadd.vx v8, v8, %[c]\n"
                     "vsll.vi v8, v8, 2\n"
                     "vadd.vx v8, v8, %[d]\n" // value = 4r + d = src[r*4+d]
                     "vse32.v v8, (%[dst])\n"
                     :
                     : [n] "r"(128u), [seed] "r"(rs_seed32), [c] "r"(c),
                       [d] "r"(d), [dst] "r"(rs_srcT + d * NREC + c)
                     : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                       "memory");
      }
#endif
    bench_fill_zero(rs_dst, sizeof(rs_dst));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    scatter_vsxblk(rs_dst, rs_src, rs_slot, NREC);
#else
    scatter_vsuxei(rs_dst, rs_src, rs_slot, NREC);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // SAMPLED check (every 4th record + the last): exact record image at
    // the slot the generator assigned. Sampling bounds the scalar-core
    // reference cost; a structural scatter bug hits sampled records too.
    for (unsigned int s = 0; s <= NREC / 4 && fails == 0; ++s) {
      const unsigned int r = (s == NREC / 4) ? (NREC - 1) : (s * 4);
      for (unsigned int d = 0; d < RD; ++d)
        if (rs_dst[(uint32_t)rs_slot[r] * RD + d] != rs_src[r * RD + d]) {
          printf("FAILED rec=%d elem=%d got=0x%x exp=0x%x\n", r, d,
                 (unsigned)rs_dst[(uint32_t)rs_slot[r] * RD + d],
                 (unsigned)rs_src[r * RD + d]);
          fails = 1;
        }
    }

    printf("radix-scatter variant=%d nrec=%d fanout=%u cache=%d: took %u "
           "cycles %s (bytes=%u)\n", VARIANT, NREC, (unsigned)FANOUT,
           USE_CACHE, cycles, fails ? "CHECK-FAILED" : "CHECK-OK",
           (unsigned)(NREC * RD * 4));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
