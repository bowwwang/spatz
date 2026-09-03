// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// pr-gather: PageRank pull iteration (GAP/Graph500-class graph analytics):
// rank_new[v] = base + damp * sum_{u in N(v)} contrib[u], contributions
// gathered by neighbor id (fp64, D=1: 8-B "blocks" = one port word).
// This is the fine-granularity regime point: the win is INDEX SEMANTICS -
// vlxblk consumes u16 vertex ids directly (hardware scales by 8), while
// vluxei64 needs the ids widened u16->u32->u64 and shifted by 3 in
// software, with an 8-byte-per-element index stream.
//
// Arms (VARIANT): 1 = vlxblkei16 with blk_len=1 (e64 data)
//                 0 = vluxei64 + software index widening/scaling
// Check: exact fp64 (ordered reduction vfredosum in both arms).
// Synthetic uniform-degree graph (deg=64), stated simplification.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#define VLXBLK_WORD(f7, f3, vd, rs1n, vs2) \
  ".word ((" #f7 ")<<25)|((" #vs2 ")<<20)|((" #rs1n ")<<15)|((" #f3 ")<<12)|((" #vd ")<<7)|0x2B\n"
#define VLXBLKEI16_V(vd, rs1n, vs2) VLXBLK_WORD(0x0C, 0x5, vd, rs1n, vs2)
#define VSETBLKLEN(rs1n)            VLXBLK_WORD(0x0F, 0x0, 0, rs1n, 0)

#ifndef NV
#define NV 8192 // vertices (contrib table NV * 8 B)
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define DEG 64      // in-degree (uniform, synthetic)
#define NACT 2048   // destination vertices processed

static double pr_contrib[NV] __attribute__((section(".data"), aligned(128)));
static uint16_t pr_nbr[NACT * DEG] __attribute__((section(".data"), aligned(64)));
static double pr_out[NACT] __attribute__((section(".data"), aligned(64)));

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
    for (unsigned int i = 0; i < NV; ++i)
      pr_contrib[i] = 0.001 + 0.000001 * (double)(i % 4093);
    for (unsigned int i = 0; i < NACT * DEG; ++i)
      pr_nbr[i] = (uint16_t)((i * 2654435761u) % NV);
    memset(pr_out, 0, sizeof(pr_out));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    const double base = 0.15 / (double)NV;
    const double damp = 0.85;

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    {
      register uint32_t bl asm("t0") = 1;                 // x5: blk_len = 1
      register const double *cp asm("t1") = pr_contrib;   // x6
      asm volatile(VSETBLKLEN(5) :: "r"(bl), "r"(cp));
      for (unsigned int v = 0; v < NACT; ++v) {
        double s;
        asm volatile("vsetvli zero, %[dg], e16, m1, ta, ma\n"
                     "vle16.v v2, (%[i0])\n"
                     "vsetvli zero, %[dg], e64, m4, ta, ma\n"
                     VLXBLKEI16_V(8, 6, 2)
                     "vmv.s.x v16, zero\n"
                     "vfredosum.vs v16, v8, v16\n"
                     "vfmv.f.s %[s], v16\n"
                     : [s] "=f"(s)
                     : [dg] "r"(DEG), [i0] "r"(pr_nbr + v * DEG),
                       [dict] "r"(cp)
                     : "v2", "v8", "v9", "v10", "v11", "v16", "memory");
        pr_out[v] = base + damp * s;
      }
    }
#else
    for (unsigned int v = 0; v < NACT; ++v) {
      double s;
      // index translation: u16 ids -> u64 byte offsets (two widening steps
      // + shift by 3): the software cost vlxblk eliminates.
      asm volatile("vsetvli zero, %[dg], e16, m1, ta, ma\n"
                   "vle16.v v2, (%[i0])\n"
                   "vwaddu.vx v4, v2, zero\n"
                   "vsetvli zero, %[dg], e32, m2, ta, ma\n"
                   "vwaddu.vx v8, v4, zero\n"
                   "vsetvli zero, %[dg], e64, m4, ta, ma\n"
                   "vsll.vi v8, v8, 3\n"
                   "vluxei64.v v12, (%[c0]), v8\n"
                   "vmv.s.x v16, zero\n"
                   "vfredosum.vs v16, v12, v16\n"
                   "vfmv.f.s %[s], v16\n"
                   : [s] "=f"(s)
                   : [dg] "r"(DEG), [i0] "r"(pr_nbr + v * DEG),
                     [c0] "r"(pr_contrib)
                   : "v2", "v4", "v5", "v8", "v9", "v10", "v11", "v12",
                     "v13", "v14", "v15", "v16", "memory");
      pr_out[v] = base + damp * s;
    }
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // Exact check: ordered reduction matches sequential CPU sum.
    for (unsigned int v = 0; v < NACT && fails == 0; ++v) {
      double s = 0.0;
      for (unsigned int e = 0; e < DEG; ++e)
        s += pr_contrib[pr_nbr[v * DEG + e]];
      if (pr_out[v] != base + damp * s) {
        printf("FAILED v=%d\n", v);
        fails = 1;
      }
    }

    printf("pr-gather variant=%d nv=%d cache=%d: took %u cycles %s "
           "(adds=%u)\n", VARIANT, NV, USE_CACHE, cycles,
           fails ? "CHECK-FAILED" : "CHECK-OK", (unsigned)(NACT * DEG));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
