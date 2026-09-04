// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// gatheragg: indexed-row gather + sum pooling. One source, two cited
// geometries:
//   ROW_D=32 (128-B fp32 rows): DLRM SparseLengthsSum / embedding-bag
//       (Gupta et al., HPCA'20: rows 32-256 fp32, bags of tens of rows).
//   ROW_D=64 (256-B fp32 rows): GNN neighbor-feature aggregation
//       (HyGCN: aggregation >97% of GCN time; hidden dim 64).
// out_b[:] = sum_{l} T[idx_bl][:] for NB destinations x LP rows each.
//
// Arms (VARIANT): 1 = VLXBLK group gather (m8 = 8x d32 rows / 4x d64 rows)
//                     + per-row register-slice vfadd accumulation.
//                 2 = vle baseline (rows >= one register: piecewise rule).
// Check: exact fp32 compare (identical per-lane add order on CPU).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

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

#define NB 256 // destinations (bags / nodes)
#if ROW_D == 32
#define ACC_LMUL "m1"   // 32 e32 = exactly one register
#else
#define ACC_LMUL "m2"   // 64 e32 = two registers
#endif
#if ROW_D == 32
#define LP 40         // rows pooled per bag (DLRM-class pooling factor)
#define RPG 8         // rows per m8 gather (8 x 32 e32 = 256 = vlmax)
#elif ROW_D == 64
#define LP 16         // neighbors per node
#define RPG 4         // 4 x 64 e32 = 256 = vlmax
#else
#error "unsupported ROW_D"
#endif

static float ga_tbl[NROWS * ROW_D] __attribute__((section(".data"), aligned(128)));
static uint16_t ga_idx[NB * LP + 16] __attribute__((section(".data"), aligned(64)));
static float ga_out[NB * ROW_D] __attribute__((section(".data"), aligned(128)));

#if VARIANT == 1
static void agg_vlxblk(float *out, const float *t, const uint16_t *idx,
                       unsigned int nb) {
  register uint32_t bl asm("t0") = ROW_D;    // x5
  register const float *tp asm("t1") = t;    // x6
  asm volatile(VSETBLKLEN(5) :: "r"(bl), "r"(tp));
  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bi = idx + b * LP;
    asm volatile("vsetvli zero, %[d], e32, " ACC_LMUL ", ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [d] "r"(ROW_D) : "v24", "v25");
    for (unsigned int l = 0; l < LP; l += RPG) {
      unsigned int r = (LP - l) < RPG ? (LP - l) : RPG;
      asm volatile("vsetvli zero, %[gr], e16, m1, ta, ma\n"
                   "vle16.v v2, (%[i0])\n"
                   "vsetvli zero, %[ec], e32, m8, ta, ma\n"
                   VLXBLKEI16_V(8, 6, 2)
                   :
                   : [gr] "r"(r), [ec] "r"(r * ROW_D), [i0] "r"(bi + l),
                     [dict] "r"(tp)
                   : "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                     "v15", "memory");
#if ROW_D == 32
      // rows are single registers v8..v15 (32 e32 each)
      asm volatile("vsetvli zero, %[d], e32, m1, ta, ma\n" :: [d] "r"(ROW_D));
      switch (r) {
      case 8: asm volatile("vfadd.vv v24, v24, v15" ::: "v24");
      case 7: asm volatile("vfadd.vv v24, v24, v14" ::: "v24");
      case 6: asm volatile("vfadd.vv v24, v24, v13" ::: "v24");
      case 5: asm volatile("vfadd.vv v24, v24, v12" ::: "v24");
      case 4: asm volatile("vfadd.vv v24, v24, v11" ::: "v24");
      case 3: asm volatile("vfadd.vv v24, v24, v10" ::: "v24");
      case 2: asm volatile("vfadd.vv v24, v24, v9" ::: "v24");
      case 1: asm volatile("vfadd.vv v24, v24, v8" ::: "v24");
      }
#else
      // rows are m2 slices v8,v10,v12,v14 (64 e32 each)
      asm volatile("vsetvli zero, %[d], e32, m2, ta, ma\n" :: [d] "r"(ROW_D));
      switch (r) {
      case 4: asm volatile("vfadd.vv v24, v24, v14" ::: "v24", "v25");
      case 3: asm volatile("vfadd.vv v24, v24, v12" ::: "v24", "v25");
      case 2: asm volatile("vfadd.vv v24, v24, v10" ::: "v24", "v25");
      case 1: asm volatile("vfadd.vv v24, v24, v8" ::: "v24", "v25");
      }
#endif
    }
    asm volatile("vsetvli zero, %[d], e32, " ACC_LMUL ", ta, ma\n"
                 "vse32.v v24, (%[o])\n" :: [d] "r"(ROW_D),
                 [o] "r"(out + b * ROW_D) : "memory");
  }
}
#else
static void agg_vle(float *out, const float *t, const uint16_t *idx,
                    unsigned int nb) {
  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bi = idx + b * LP;
    asm volatile("vsetvli zero, %[d], e32, " ACC_LMUL ", ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [d] "r"(ROW_D) : "v24", "v25");
    for (unsigned int l = 0; l < LP; ++l) {
      const float *row = t + (uint32_t)bi[l] * ROW_D;
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
    for (unsigned int i = 0; i < NROWS * ROW_D; ++i)
      ga_tbl[i] = 0.25f + 0.0625f * (float)(i % 61);
    for (unsigned int i = 0; i < NB * LP + 16; ++i)
      ga_idx[i] = (uint16_t)((i * 2654435761u) % NROWS);
    memset(ga_out, 0, sizeof(ga_out));
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

    // Exact fp32 check. NOTE the vlxblk arm sums rows in DESCENDING group
    // order (case fallthrough); reproduce the same order on the CPU.
    for (unsigned int b = 0; b < NB && fails == 0; ++b)
      for (unsigned int d = 0; d < ROW_D; ++d) {
        float acc = 0.0f;
#if VARIANT == 1
        for (unsigned int l0 = 0; l0 < LP; l0 += RPG) {
          unsigned int r = (LP - l0) < RPG ? (LP - l0) : RPG;
          for (int j = (int)r - 1; j >= 0; --j)
            acc += ga_tbl[(uint32_t)ga_idx[b * LP + l0 + j] * ROW_D + d];
        }
#else
        for (unsigned int l = 0; l < LP; ++l)
          acc += ga_tbl[(uint32_t)ga_idx[b * LP + l] * ROW_D + d];
#endif
        float got = ga_out[b * ROW_D + d];
        float err = got > acc ? got - acc : acc - got;
        float mag = acc < 0 ? -acc : acc;
        if (err > 0.03125f + 9.5e-7f * mag) {
          printf("FAILED b=%d d=%d got=%f exp=%f\n", b, d, got, acc);
          fails = 1;
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
