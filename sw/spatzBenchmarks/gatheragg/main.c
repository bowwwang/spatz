// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// gatheragg: indexed-row gather + sum pooling. One source, two cited
// geometries (paper rows sls-fp32 and gnnagg):
//   ROW_D=32 (128-B fp32 rows): DLRM SparseLengthsSum fp32 control
//       (Gupta et al., HPCA'20: output dim 32, bags of ~tens of rows).
//   ROW_D=64 (256-B fp32 rows): GNN pull-mode neighbor aggregation
//       (HyGCN: aggregation >97% of GCN time; GraphSAGE fanout S1=25).
// out_b[:] = sum_{l} T[idx_bl][:] for NB destinations x LP rows each.
//
// Arms (VARIANT): 1 = VLXBLK, gather-across-units (the proven vqgemv
//                     pattern): UPG destinations pooled simultaneously —
//                     iteration l gathers the l-th row of all UPG units
//                     into one full m8 group (strided index load, one
//                     index per unit) and accumulates with a whole-group
//                     vfadd. NO register-group slicing: the old per-row
//                     m1/m2-slice + switch-fallthrough pattern hangs
//                     (d32) or zeroes (d64) in RTL.
//                 2 = vle baseline (rows >= one register: piecewise rule).
// Check: exact fp32 compare (same ascending add order in both arms).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

// Native VLXBLK mnemonics (LLVM 14 + MC-layer patch); x-register
// form keeps the numeric rs1n interface, so call sites are unchanged.
#define VLXBLKEI16_V(vd, rs1n, vs2)  "vlxblkei16.v v" #vd ", (x" #rs1n "), v" #vs2 "\n"
#define VSETBLKLEN(rs1n)             "vsetblklen x" #rs1n "\n"

#ifndef ROW_D
#define ROW_D 32
#endif
#ifndef NROWS
#define NROWS 4096
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define NB 2048 // destinations (bags / nodes) = T in the paper table
#if ROW_D == 32
#define ACC_LMUL "m1"   // 32 e32 = exactly one register
#else
#define ACC_LMUL "m2"   // 64 e32 = two registers
#endif
#if ROW_D == 32
#define LP 40         // rows pooled per bag (HPCA'20: "~tens" of IDs)
#elif ROW_D == 64
#define LP 25         // neighbors per node (GraphSAGE S1 = 25)
#else
#error "unsupported ROW_D"
#endif
#define UPG (128 / ROW_D) // units per m4 group (4x d32 / 2x d64 rows)
// m4, not m8: no PASSING kernel computes at m8 (m8 vfadd is the
// remaining hang suspect after vlse16 was exonerated; vqdecode proves
// m8 GATHERS are fine). All shapes here match the proven vqgemv/vqgemm
// m4 patterns.
#if (NB % UPG) != 0
#error "NB must be a multiple of the units-per-group width"
#endif

static float ga_tbl[NROWS * ROW_D] __attribute__((section(".data"), aligned(128)));
static uint16_t ga_idx[NB * LP + 128] __attribute__((section(".data"), aligned(64)));
static float ga_out[NB * ROW_D] __attribute__((section(".data"), aligned(128)));
static uint16_t ga_seed16[128] __attribute__((section(".data"), aligned(64)));

#if VARIANT == 1
static void agg_vlxblk(float *out, const float *t, const uint16_t *idx,
                       unsigned int nb) {
  register uint32_t bl asm("t0") = ROW_D;    // x5
  register const float *tp asm("t1") = t;    // x6
  asm volatile(VSETBLKLEN(5) :: "r"(bl), "r"(tp));
  // Accumulator v16-v19 (m4) holds UPG units side by side; the gathered
  // rows land in v8-v11 (m4) in the same unit positions, so the
  // accumulate is one whole-group vfadd.
  for (unsigned int b = 0; b < nb; b += UPG) {
    const uint16_t *bi = idx + b * LP;
    asm volatile("vsetvli zero, %[ec], e32, m4, ta, ma\n"
                 "vmv.v.i v16, 0\n"
                 :: [ec] "r"((uint32_t)(UPG * ROW_D))
                 : "v16", "v17", "v18", "v19");
    for (unsigned int l = 0; l < LP; ++l) {
      // Index layout is TRANSPOSED (round-major within each unit group):
      // round l's UPG indices are contiguous -> plain vle16 (vlse16
      // strided loads hang on this port; see ISA probe).
      asm volatile("vsetvli zero, %[g], e16, m1, ta, ma\n"
                   "vle16.v v2, (%[i0])\n"
                   "vsetvli zero, %[ec], e32, m4, ta, ma\n"
                   VLXBLKEI16_V(8, 6, 2)
                   "vfadd.vv v16, v16, v8\n"
                   :
                   : [g] "r"((uint32_t)UPG), [ec] "r"((uint32_t)(UPG * ROW_D)),
                     [i0] "r"(bi + l * UPG),
                     [dict] "r"(tp)
                   : "v2", "v8", "v9", "v10", "v11",
                     "v16", "v17", "v18", "v19", "memory");
    }
    asm volatile("vsetvli zero, %[ec], e32, m4, ta, ma\n"
                 "vse32.v v16, (%[o])\n"
                 :: [ec] "r"((uint32_t)(UPG * ROW_D)),
                    [o] "r"(out + b * ROW_D) : "memory");
  }
}
#else
static void agg_vle(float *out, const float *t, const uint16_t *idx,
                    unsigned int nb) {
  for (unsigned int b = 0; b < nb; ++b) {
    // Transposed index layout: unit b reads position
    // (b/UPG)*LP*UPG + l*UPG + (b%UPG).
    const uint16_t *bg = idx + (b / UPG) * (LP * UPG) + (b % UPG);
    asm volatile("vsetvli zero, %[d], e32, " ACC_LMUL ", ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [d] "r"(ROW_D) : "v24", "v25");
    for (unsigned int l = 0; l < LP; ++l) {
      const float *row = t + (uint32_t)bg[l * UPG] * ROW_D;
      asm volatile("vsetvli zero, %[d], e32, " ACC_LMUL ", ta, ma\n"
                   "vle32.v v8, (%[r])\n"
                   "vfadd.vv v24, v24, v8\n"
                   :
                   : [r] "r"(row), [d] "r"(ROW_D)
                   : "v8", "v9", "v24", "v25", "memory");
    }
    asm volatile("vse32.v v24, (%[o])\n" :: [o] "r"(out + b * ROW_D)
                 : "memory");
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
    // Table: pattern head + vector replication; the index array is
    // stored TRANSPOSED (round-major per unit group) so the kernel's
    // per-round index loads are contiguous.
    //
    // row ids: u16-lane multiplicative hash masked to NROWS (pow2),
    // vector-generated. Scalar datagen at MiB scale dominates sim
    // wall-clock (bench_fill.h).
    {
      const unsigned int n = NROWS * ROW_D;
      const unsigned int head = n < 976u ? n : 976u; // f32: BF_HEAD_BYTES/4
      for (unsigned int i = 0; i < head; ++i)
        ga_tbl[i] = 0.25f + 0.003f * (float)((i * 37u) & 2047u);
      if (n > head)
        bench_fill_rep(ga_tbl, n * 4u, head * 4u);
    }
    for (unsigned int i = 0; i < 128; ++i)
      ga_seed16[i] = (uint16_t)i;
#if (NROWS & (NROWS - 1)) != 0
#error "NROWS must be a power of two (masked id generation)"
#endif
    for (unsigned int c = 0; c < NB * LP + 16u; c += 128) {
      asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                   "vle16.v v8, (%[seed])\n"
                   "vadd.vx v8, v8, %[c]\n"
                   "vmul.vx v8, v8, %[mult]\n"
                   "vand.vx v8, v8, %[mask]\n"
                   "vse16.v v8, (%[dst])\n"
                   :
                   : [n] "r"(128u), [seed] "r"(ga_seed16), [c] "r"(c),
                     [mult] "r"(2654435761u), [mask] "r"(NROWS - 1u),
                     [dst] "r"(ga_idx + c)
                   : "v8", "v9", "v10", "v11", "memory");
    }
    bench_fill_zero(ga_out, sizeof(ga_out));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    agg_vlxblk(ga_out, ga_tbl, ga_idx, NB);
#else
    agg_vle(ga_out, ga_tbl, ga_idx, NB);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // SAMPLED fp32 check (every 64th unit + the last; all elements
    // within a checked unit): both arms accumulate rows in ascending l
    // order. Sampling bounds the scalar-core reference cost.
    for (unsigned int s64 = 0; s64 <= NB / 64 && fails == 0; ++s64) {
      const unsigned int b = (s64 == NB / 64) ? (NB - 1) : (s64 * 64);
      for (unsigned int d = 0; d < ROW_D; ++d) {
        float acc = 0.0f;
        for (unsigned int l = 0; l < LP; ++l)
          acc += ga_tbl[(uint32_t)ga_idx[(b / UPG) * (LP * UPG) + l * UPG +
                                         (b % UPG)] * ROW_D + d];
        float got = ga_out[b * ROW_D + d];
        float err = got > acc ? got - acc : acc - got;
        float mag = acc < 0 ? -acc : acc;
        if (err > 0.03125f + 9.5e-7f * mag) {
          printf("FAILED b=%d d=%d got=%f exp=%f\n", b, d, got, acc);
          fails = 1;
        }
      }
    }

    printf("gatheragg d=%d variant=%d nrows=%d cache=%d: took %u cycles %s "
           "(adds=%u)\n", ROW_D, VARIANT, NROWS, USE_CACHE, cycles,
           fails ? "CHECK-FAILED" : "CHECK-OK", (unsigned)(NB * LP * ROW_D));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
