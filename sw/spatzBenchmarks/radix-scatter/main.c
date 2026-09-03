// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// radix-scatter: DB radix-partitioning record scatter (Polychroniou,
// Raghavan & Ross, SIGMOD'15): fixed 16-B records scattered to unique
// destination slots (a permutation, as in partitioning with precomputed
// prefix-sum offsets). Software today emulates this with cache-line
// buffers + non-temporal stores; vsxblk is the direct expression.
//
// Arms (VARIANT): 1 = VSXBLK block scatter (64 records per chunk)
//                 0 = vsuxei16 baseline with per-element offset expansion
//                     (16-bit byte offsets => destination <= 64 KiB)
// Pure data movement: no FLOPs; report bandwidth. Check: exact image.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#define VLXBLK_WORD(f7, f3, vd, rs1n, vs2) \
  ".word ((" #f7 ")<<25)|((" #vs2 ")<<20)|((" #rs1n ")<<15)|((" #f3 ")<<12)|((" #vd ")<<7)|0x2B\n"
#define VSXBLKEI16_V(vs3, rs1n, vs2) VLXBLK_WORD(0x0D, 0x5, vs3, rs1n, vs2)
#define VSETBLKLEN(rs1n)             VLXBLK_WORD(0x0F, 0x0, 0, rs1n, 0)

#ifndef NREC
#define NREC 4096 // records; 16 B each
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define RD 4 // record elements (e32) -> 16-B records
#if VARIANT == 0 && (NREC * RD * 4) > 65536
#error "vsuxei16 baseline limited to 64 KiB destinations (e16 byte offsets)"
#endif

static uint32_t rs_src[NREC * RD] __attribute__((section(".data"), aligned(128)));
static uint32_t rs_dst[NREC * RD] __attribute__((section(".data"), aligned(128)));
static uint16_t rs_slot[NREC + 16] __attribute__((section(".data"), aligned(64)));

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
// Baseline: expand record slots to per-element e16 byte offsets (doubling
// expansion, D=4 -> even parity -> result in v4), then vsuxei16.
static void scatter_vsuxei(uint32_t *dst, const uint32_t *src,
                           const uint16_t *slot, unsigned int n) {
  for (unsigned int r = 0; r < n; r += 64) {
    asm volatile("vsetvli zero, %[gr], e16, m2, ta, ma\n"
                 "vle16.v v4, (%[i0])\n"
                 "vsll.vi v4, v4, 4\n" // slot -> byte offset (16-B records)
                 :
                 : [gr] "r"(64u), [i0] "r"(slot + r)
                 : "v4", "v5", "memory");
    size_t len = 64;
    unsigned long delta = (RD * 4) >> 1;
    for (unsigned int t = 0; t < 2; ++t) { // log2(RD) = 2 steps
      const unsigned long dhi = delta << 16;
      if (!(t & 1))
        asm volatile("vsetvli zero, %[l], e16, m2, ta, ma\n"
                     "vwaddu.vx v24, v4, zero\n"
                     "vsetvli zero, %[l], e32, m4, ta, ma\n"
                     "vsll.vi v28, v24, 16\n"
                     "vadd.vv v24, v24, v28\n"
                     "vadd.vx v24, v24, %[dh]\n"
                     :: [l] "r"(len), [dh] "r"(dhi)
                     : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
      else
        asm volatile("vsetvli zero, %[l], e16, m2, ta, ma\n"
                     "vwaddu.vx v4, v24, zero\n"
                     "vsetvli zero, %[l], e32, m4, ta, ma\n"
                     "vsll.vi v28, v4, 16\n"
                     "vadd.vv v4, v4, v28\n"
                     "vadd.vx v4, v4, %[dh]\n"
                     :: [l] "r"(len), [dh] "r"(dhi)
                     : "v4", "v5", "v6", "v7", "v28", "v29", "v30", "v31");
      len <<= 1;
      delta >>= 1;
    }
    // D=4: 2 steps, even -> expanded offsets end in v4 (e16, 256 elems, m4)
    asm volatile("vsetvli zero, %[ec], e32, m8, ta, ma\n"
                 "vle32.v v8, (%[s0])\n"
                 "vsuxei16.v v8, (%[d0]), v4\n"
                 :
                 : [ec] "r"(256u), [s0] "r"(src + r * RD), [d0] "r"(dst)
                 : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                   "memory");
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
    // permutation of slots (3 * 2^k + ... : use odd multiplier mod NREC)
    for (unsigned int i = 0; i < NREC; ++i)
      rs_slot[i] = (uint16_t)(((uint32_t)i * 40503u + 11u) % NREC);
    for (unsigned int i = 0; i < NREC * RD; ++i) {
      rs_src[i] = 0xE0000000u + i;
      rs_dst[i] = 0;
    }
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

    for (unsigned int r = 0; r < NREC && fails == 0; ++r)
      for (unsigned int d = 0; d < RD; ++d)
        if (rs_dst[(uint32_t)rs_slot[r] * RD + d] != rs_src[r * RD + d]) {
          printf("FAILED rec=%d elem=%d\n", r, d);
          fails = 1;
        }

    printf("radix-scatter variant=%d nrec=%d cache=%d: took %u cycles %s "
           "(bytes=%u)\n", VARIANT, NREC, USE_CACHE, cycles,
           fails ? "CHECK-FAILED" : "CHECK-OK", (unsigned)(NREC * RD * 4));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
