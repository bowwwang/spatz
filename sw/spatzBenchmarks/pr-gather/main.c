// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// pr-gather: PageRank pull iteration (GAP/Graph500-class graph analytics):
// rank_new[v] = base + damp * sum_{u in N(v)} contrib[u], contributions
// gathered by neighbor id (fp64, D=1: 8-B "blocks" = one port word).
// This is the fine-granularity regime point: the win is INDEX SEMANTICS -
// vlxblk consumes u16 vertex ids directly (hardware scales by 8), while
// the vluxei baseline widens+scales the ids in software (vwmulu by 8,
// the most favorable single-op translation) and carries a 4-B/element
// index stream via vluxei32.
//
// Arms (VARIANT): 1 = vlxblkei16 with blk_len=1 (e64 data)
//                 0 = vluxei32 + software index widening/scaling
// Check: exact fp64 (ordered reduction vfredosum in both arms).
// Synthetic uniform-degree graph (deg=16 = GAP/Graph500 average degree;
// uniform neighbor ids are the stated, conservative simplification).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

// Native VLXBLK mnemonics (LLVM 14 + MC-layer patch); x-register
// form keeps the numeric rs1n interface, so call sites are unchanged.
#define VLXBLKEI16_V(vd, rs1n, vs2)  "vlxblkei16.v v" #vd ", (x" #rs1n "), v" #vs2 "\n"
#define VSETBLKLEN(rs1n)             "vsetblklen x" #rs1n "\n"

#ifndef NV
#define NV 8192 // vertices (contrib table NV * 8 B)
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define DEG 16      // in-degree: GAP/Graph500 average degree (edgefactor 16)
#define NACT 4096   // destination vertices processed (timed tile)

// Both vsetvli requests (e16 m1 index load, e64 m4 gather) cap at 32
// elements on VLEN=512 — a larger DEG silently clamps vl and sums only
// the first 32 neighbors (the old DEG=64 CHECK-FAIL). Strip-mine before
// raising this.
#if DEG > 32
#error "DEG > 32 exceeds one vsetvli group (e64 m4 / e16 m1); strip-mine"
#endif

static double pr_contrib[NV] __attribute__((section(".data"), aligned(128)));
static uint16_t pr_nbr[NACT * DEG] __attribute__((section(".data"), aligned(64)));
static double pr_out[NACT] __attribute__((section(".data"), aligned(64)));
static uint16_t pr_seed16[128] __attribute__((section(".data"), aligned(64)));

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
    // Contributions: prime-period pattern head (509 f64) + vector
    // replication (bench_fill.h) — a scalar fill of 512 KiB dominates
    // sim wall-clock.
    {
      const unsigned int head = NV < 488u ? NV : 488u; // f64: BF_HEAD_BYTES/8
      for (unsigned int i = 0; i < head; ++i)
        pr_contrib[i] = 0.001 + 0.000001 * (double)((i * 211u) & 2047u);
      if (NV > head)
        bench_fill_rep(pr_contrib, NV * 8u, head * 8u);
    }
    // Neighbor ids: multiplicative hash in u16 lanes masked to NV
    // (pow2), vector-generated — (i * 0x9E3779B1) mod 2^16 & (NV-1).
#if (NV & (NV - 1)) != 0
#error "NV must be a power of two (masked id generation)"
#endif
    for (unsigned int i = 0; i < 128; ++i)
      pr_seed16[i] = (uint16_t)i;
    for (unsigned int c = 0; c < NACT * DEG; c += 128) {
      asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                   "vle16.v v8, (%[seed])\n"
                   "vadd.vx v8, v8, %[c]\n"
                   "vmul.vx v8, v8, %[mult]\n"
                   "vand.vx v8, v8, %[mask]\n"
                   "vse16.v v8, (%[dst])\n"
                   :
                   : [n] "r"(128u), [seed] "r"(pr_seed16), [c] "r"(c),
                     [mult] "r"(2654435761u), [mask] "r"(NV - 1u),
                     [dst] "r"(pr_nbr + c)
                   : "v8", "v9", "v10", "v11", "memory");
    }
    bench_fill_zero(pr_out, sizeof(pr_out));
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
    // SCALAR baseline (the GAP reference itself is scalar C): the
    // element-vector translation (vwmulu widen + vluxei32) HANGS on the
    // first missing gather on this platform — the upstream-vluxei-
    // under-miss bug (wave-4 bisection, 2026-09-08); misses are
    // intrinsic here (512 KiB contrib array > L1), so no vluxei
    // baseline can run. Volatile reads keep -O3 from re-vectorizing.
    for (unsigned int v = 0; v < NACT; ++v) {
      double s = 0.0;
      const volatile double *cv = pr_contrib;
      for (unsigned int e = 0; e < DEG; ++e)
        s += cv[pr_nbr[v * DEG + e]];
      pr_out[v] = base + damp * s;
    }
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // SAMPLED exact check (every 4th vertex + the last): ordered
    // reduction matches the sequential CPU sum bit-exactly.
    for (unsigned int s4 = 0; s4 <= NACT / 4 && fails == 0; ++s4) {
      const unsigned int v = (s4 == NACT / 4) ? (NACT - 1) : (s4 * 4);
      double s = 0.0;
      for (unsigned int e = 0; e < DEG; ++e)
        s += pr_contrib[pr_nbr[v * DEG + e]];
      // vfredosum's internal association differs from the sequential C
      // sum in final ULPs on this implementation (v=4 diagnostic,
      // 2026-09-08: got==exp to 6 decimals) -> 1e-9 relative tolerance;
      // a single wrong index perturbs the sum by ~1e-4 relative, five
      // orders above this bound, so index errors stay fully detectable.
      const double expd = base + damp * s;
      double errd = pr_out[v] - expd;
      if (errd < 0) errd = -errd;
      double magd = expd < 0 ? -expd : expd;
      if (errd > 1e-9 * magd + 1e-15) {
        printf("FAILED v=%d got=%f exp=%f\n", v, pr_out[v], expd);
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
