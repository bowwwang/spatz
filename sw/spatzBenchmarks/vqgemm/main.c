// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// vqgemm: vector-quantized GEMM with codebook decode FUSED into the
// matrix product (AQLM/VPTQ-class, 2-codebook residual VQ):
// C[M,N] += A[M,P] x W[P,N], W rows decoded on the fly as in vqgemv.
// Tiling: 4 output rows live in the VRF (4 x m4 fp16 accumulators);
// the decoded row w_k is produced ONCE per (tile, k) and consumed by 4
// vfmacc.vf — the VRF-capacity tiling any fused kernel needs (decode is
// re-done M/4 times; stated in the paper).
//
// Arms (VARIANT): 1 = VLXBLK: w_k = vlxblk(cb0) + vlxblk(cb1), scaled;
//                 0 = RVV baseline: same tiling, w_k decoded group-by-
//                     group (scalar-addressed vle16) into a scratch row,
//                     reloaded as one m4 vector for the MACs.
// Configs as vqgemv: AQLM 2x8 (CB_D=8, CBN=256, u8 idx) / VPTQ v16
// (CB_D=16, CBN=4096, u16 idx). Golden-checked (sampled) on-core.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

static inline float f16f(const __fp16 *p) {
  float v;
  asm("flh ft0, 0(%1)\n\t"
      "fcvt.s.h %0, ft0"
      : "=f"(v)
      : "r"(p)
      : "ft0");
  return v;
}

#ifndef CB_D
#define CB_D 8
#endif
#ifndef MDIM
#define MDIM 128
#endif
#ifndef KDIM
#define KDIM 128 // reduction dimension (P in the paper table)
#endif
#ifndef NDIM
#define NDIM 128
#endif
#ifndef VARIANT
#define VARIANT 1
#endif
#ifndef CBN
#define CBN 256
#endif

#define GROUPS (NDIM / CB_D)

#if CBN > 256
typedef uint16_t idx_t;
#define VLE_IDX    "vle16.v"
#define VLXBLK_IDX "vlxblkei16.v"
#define IDX_SEW    "e16"
#else
typedef uint8_t idx_t;
#define VLE_IDX    "vle8.v"
#define VLXBLK_IDX "vlxblkei8.v"
#define IDX_SEW    "e8"
#endif

typedef __fp16 f16;

static f16 a_mat[MDIM * KDIM] __attribute__((section(".data"), aligned(64)));
static f16 cb0[CBN * CB_D] __attribute__((section(".data"), aligned(64)));
static f16 cb1[CBN * CB_D] __attribute__((section(".data"), aligned(64)));
static idx_t idx0[KDIM * GROUPS] __attribute__((section(".data"), aligned(64)));
static idx_t idx1[KDIM * GROUPS] __attribute__((section(".data"), aligned(64)));
static f16 scales[KDIM] __attribute__((section(".data"), aligned(64)));
static f16 c_out[MDIM * NDIM] __attribute__((section(".data"), aligned(64)));
#if VARIANT == 0
// +16: decode stores are padded to 32 B (16-B vector stores hang the
// cache port; see isaprobe); ascending g overwrites the padding lanes.
static f16 wrow[NDIM + 16] __attribute__((section(".data"), aligned(64)));
#endif

#if VARIANT == 1
// Register map (e16): v2/v3 index vectors (m1); v8/v12/v16/v20 the four
// m4 row accumulators; v24-v27 + v28-v31 gather targets, v24 becomes w.
static void vqgemm_vlxblk(f16 *c, const f16 *a, const f16 *b0, const f16 *b1,
                          const idx_t *bi0, const idx_t *bi1, const f16 *sc,
                          unsigned int M, unsigned int K, unsigned int N) {
  asm volatile("vsetblklen %0" ::"r"((uint32_t)CB_D));
  const unsigned int groups = N / CB_D;
  for (unsigned int m = 0; m < M; m += 4) {
    asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                 "vmv.v.i v8, 0\n"
                 "vmv.v.i v12, 0\n"
                 "vmv.v.i v16, 0\n"
                 "vmv.v.i v20, 0\n"
                 :: [n] "r"((uint32_t)N)
                 : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                   "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    for (unsigned int k = 0; k < K; ++k) {
      float scale, a0, a1, a2, a3;
      asm volatile("flh %[s], 0(%[sc])" : [s] "=f"(scale) : [sc] "r"(sc + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a0) : [p] "r"(a + (m + 0) * K + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a1) : [p] "r"(a + (m + 1) * K + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a2) : [p] "r"(a + (m + 2) * K + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a3) : [p] "r"(a + (m + 3) * K + k));
      asm volatile("vsetvli zero, %[g], " IDX_SEW ", m1, ta, ma\n"
                   VLE_IDX " v2, (%[i0])\n"
                   VLE_IDX " v3, (%[i1])\n"
                   "vsetvli zero, %[n], e16, m4, ta, ma\n"
                   VLXBLK_IDX " v24, (%[cb0]), v2\n"
                   VLXBLK_IDX " v28, (%[cb1]), v3\n"
                   "vfadd.vv v24, v24, v28\n"
                   "vfmul.vf v24, v24, %[scale]\n"
                   "vfmacc.vf v8,  %[a0], v24\n"
                   "vfmacc.vf v12, %[a1], v24\n"
                   "vfmacc.vf v16, %[a2], v24\n"
                   "vfmacc.vf v20, %[a3], v24\n"
                   :
                   : [g] "r"(groups), [n] "r"((uint32_t)N),
                     [i0] "r"(bi0 + k * groups), [i1] "r"(bi1 + k * groups),
                     [cb0] "r"(b0), [cb1] "r"(b1), [scale] "f"(scale),
                     [a0] "f"(a0), [a1] "f"(a1), [a2] "f"(a2), [a3] "f"(a3)
                   : "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13",
                     "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                     "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                     "v30", "v31", "memory");
    }
    asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                 "vse16.v v8, (%[c0])\n"
                 "vse16.v v12, (%[c1])\n"
                 "vse16.v v16, (%[c2])\n"
                 "vse16.v v20, (%[c3])\n"
                 :: [n] "r"((uint32_t)N), [c0] "r"(c + (m + 0) * N),
                    [c1] "r"(c + (m + 1) * N), [c2] "r"(c + (m + 2) * N),
                    [c3] "r"(c + (m + 3) * N) : "memory");
  }
}
#else
static void vqgemm_rvv(f16 *c, const f16 *a, const f16 *b0, const f16 *b1,
                       const idx_t *bi0, const idx_t *bi1, const f16 *sc,
                       unsigned int M, unsigned int K, unsigned int N) {
  const unsigned int groups = N / CB_D;
  for (unsigned int m = 0; m < M; m += 4) {
    asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                 "vmv.v.i v8, 0\n"
                 "vmv.v.i v12, 0\n"
                 "vmv.v.i v16, 0\n"
                 "vmv.v.i v20, 0\n"
                 :: [n] "r"((uint32_t)N)
                 : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                   "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    for (unsigned int k = 0; k < K; ++k) {
      float scale, a0, a1, a2, a3;
      asm volatile("flh %[s], 0(%[sc])" : [s] "=f"(scale) : [sc] "r"(sc + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a0) : [p] "r"(a + (m + 0) * K + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a1) : [p] "r"(a + (m + 1) * K + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a2) : [p] "r"(a + (m + 2) * K + k));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(a3) : [p] "r"(a + (m + 3) * K + k));
      // Decode w_k group-by-group into the scratch row (natural RVV).
      asm volatile("vsetvli zero, %[d], e16, m1, ta, ma" ::[d] "r"((uint32_t)CB_D));
      for (unsigned int g = 0; g < groups; ++g) {
        const unsigned int ix = k * groups + g;
        const f16 *p0 = b0 + ((uint32_t)bi0[ix] * CB_D);
        const f16 *p1 = b1 + ((uint32_t)bi1[ix] * CB_D);
        asm volatile("vle16.v v24, (%[p0])\n"
                     "vle16.v v28, (%[p1])\n"
                     "vfadd.vv v24, v24, v28\n"
                     "vfmul.vf v24, v24, %[scale]\n"
#if CB_D < 16
                     // pad the store to 32 B (sub-32-B vse hangs)
                     "vsetvli zero, %[pv], e16, m1, ta, ma\n"
                     "vse16.v v24, (%[w])\n"
                     "vsetvli zero, %[d], e16, m1, ta, ma\n"
#else
                     "vse16.v v24, (%[w])\n"
#endif
                     :
                     : [p0] "r"(p0), [p1] "r"(p1), [scale] "f"(scale),
                       [w] "r"(wrow + g * CB_D), [pv] "r"(16u),
                       [d] "r"((uint32_t)CB_D)
                     : "v24", "v25", "v28", "v29", "memory");
      }
      asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                   "vle16.v v24, (%[w])\n"
                   "vfmacc.vf v8,  %[a0], v24\n"
                   "vfmacc.vf v12, %[a1], v24\n"
                   "vfmacc.vf v16, %[a2], v24\n"
                   "vfmacc.vf v20, %[a3], v24\n"
                   :
                   : [n] "r"((uint32_t)N), [w] "r"(wrow), [a0] "f"(a0),
                     [a1] "f"(a1), [a2] "f"(a2), [a3] "f"(a3)
                   : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                     "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                     "v24", "v25", "v26", "v27", "memory");
    }
    asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                 "vse16.v v8, (%[c0])\n"
                 "vse16.v v12, (%[c1])\n"
                 "vse16.v v16, (%[c2])\n"
                 "vse16.v v20, (%[c3])\n"
                 :: [n] "r"((uint32_t)N), [c0] "r"(c + (m + 0) * N),
                    [c1] "r"(c + (m + 1) * N), [c2] "r"(c + (m + 2) * N),
                    [c3] "r"(c + (m + 3) * N) : "memory");
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
    for (unsigned int k = 0; k < KDIM; ++k)
      scales[k] = (f16)(0.5f + 0.01f * (float)(k & 7u));
    for (unsigned int i = 0; i < MDIM * KDIM; ++i)
      a_mat[i] = (f16)(0.02f + 0.0005f * (float)((i * 41u) & 63u));
    {
      const unsigned int cbe = CBN * CB_D;
      const unsigned int head = cbe < 1952u ? cbe : 1952u; // f16: BF_HEAD_BYTES/2
      for (unsigned int i = 0; i < head; ++i) {
        cb0[i] = (f16)(-0.5f + 0.001f * (float)((i * 37u) & 1023u));
        cb1[i] = (f16)(-0.25f + 0.0008f * (float)((i * 53u) & 1023u));
      }
      if (cbe > head) {
        bench_fill_rep(cb0, cbe * 2u, head * 2u);
        bench_fill_rep(cb1, cbe * 2u, head * 2u);
      }
    }
    for (unsigned int i = 0; i < KDIM * GROUPS; ++i) {
      idx0[i] = (idx_t)((i * 179u + 3u) % CBN);
      idx1[i] = (idx_t)((i * 83u + 17u) % CBN);
    }
    bench_fill_zero(c_out, sizeof(c_out));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    vqgemm_vlxblk(c_out, a_mat, cb0, cb1, idx0, idx1, scales, MDIM, KDIM, NDIM);
#else
    vqgemm_rvv(c_out, a_mat, cb0, cb1, idx0, idx1, scales, MDIM, KDIM, NDIM);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // SAMPLED golden check: m in {0,31,63,95,127}, all groups, d in
    // {0, CB_D-1}; per-(m,k) activation-scale product recomputed, cb
    // reads via hardware fp16 loads. fp16 accumulate -> tolerance.
    const unsigned int msel[5] = {0u, 31u, 63u, 95u, MDIM - 1u};
    const unsigned int dsel[2] = {0u, CB_D - 1u};
    static float ref_s[KDIM];
    for (unsigned int k = 0; k < KDIM; ++k)
      ref_s[k] = f16f(&scales[k]);
    for (unsigned int ms = 0; ms < 5 && fails == 0; ++ms) {
      const unsigned int m = msel[ms];
      for (unsigned int g = 0; g < GROUPS && fails == 0; ++g)
        for (unsigned int ds = 0; ds < 2 && fails == 0; ++ds) {
          const unsigned int d = dsel[ds];
          float acc = 0.0f;
          for (unsigned int k = 0; k < KDIM; ++k) {
            const unsigned int ix = k * GROUPS + g;
            float w = f16f(&cb0[(uint32_t)idx0[ix] * CB_D + d]) +
                      f16f(&cb1[(uint32_t)idx1[ix] * CB_D + d]);
            acc += f16f(&a_mat[m * KDIM + k]) * (ref_s[k] * w);
          }
          const float got = f16f(&c_out[m * NDIM + g * CB_D + d]);
          float err = got > acc ? got - acc : acc - got;
          float mag = acc < 0 ? -acc : acc;
          if (err > 0.25f + 0.015625f * mag) {
            printf("FAILED m=%d g=%d d=%d got=%f exp=%f\n", m, g, d, got, acc);
            fails = 1;
          }
        }
    }

    printf("vqgemm variant=%d M=%d K=%d N=%d cache=%d: took %u cycles %s "
           "(macs=%u)\n", VARIANT, MDIM, KDIM, NDIM, USE_CACHE, cycles,
           fails ? "CHECK-FAILED" : "CHECK-OK",
           (unsigned)(MDIM * KDIM * NDIM));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
