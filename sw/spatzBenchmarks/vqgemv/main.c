// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// vqgemv: vector-quantized GEMV with codebook decode FUSED into the
// matrix-vector product (AQLM/VPTQ-class, 2-codebook residual VQ).
// c[N] = sum_k a[k] * W[k,:], where each CB_D-element column group of W
// at (k,g) is decoded on the fly: W_group = scale[k] * (cb0[idx0[k,g]] +
// cb1[idx1[k,g]]). Codebooks: CBN x CB_D fp16 per table — AQLM 2x8 config
// is 2 x 4 KiB (L1-resident, u8 idx), VPTQ v16 config is 2 x 128 KiB
// (exceeds the 112 KiB L1, u16 idx). Inner loop is the vq-date-verified
// kernel (84/84 golden there).
//
// Arms (VARIANT): 1 = VLXBLK: one vlxblkei8 gathers a whole vl of decoded
//                     groups (vectorized ACROSS output groups), then
//                     vfadd/vfmul/vfmacc on the full m4 vector.
//                 0 = RVV baseline: one output group per iteration, each
//                     codebook entry loaded by scalar-computed address
//                     (vle16) - the natural direct RVV kernel.
// Real MACs: the vfmacc is K*N fp16 MACs (GEMV). Golden-checked on-core.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

// Safe __fp16 -> float. This custom LLVM-14 toolchain's software fp16->float
// conversion (a plain C cast) is broken (garbage/NaN); go through the
// hardware half load + fcvt.s.h instead. Used by the CPU reference.
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
#define CB_D 8       // entry length (elements): 8 = AQLM 2x8 (16 B), 16 = VPTQ v16 (32 B)
#endif
#ifndef KDIM
#define KDIM 512     // accumulation dimension
#endif
#ifndef NDIM
#define NDIM 2048    // output dimension
#endif
#ifndef VARIANT
#define VARIANT 1
#endif
#ifndef CBN
#define CBN 256      // codebook entries: 256 = AQLM (u8 idx), 4096 = VPTQ (u16 idx)
#endif

#define GROUPS (NDIM / CB_D)

// Index width follows the entry count: >256 entries need 16-bit indices.
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

static f16 a_vec[KDIM] __attribute__((section(".data"), aligned(64)));
static f16 cb0[CBN * CB_D] __attribute__((section(".data"), aligned(64)));
static f16 cb1[CBN * CB_D] __attribute__((section(".data"), aligned(64)));
static idx_t idx0[KDIM * GROUPS] __attribute__((section(".data"), aligned(64)));
static idx_t idx1[KDIM * GROUPS] __attribute__((section(".data"), aligned(64)));
static f16 scales[KDIM] __attribute__((section(".data"), aligned(64)));
// +16 elements: the rvv arm's per-group stores are padded to 32 B
// (vector stores narrower than 32 B hang the cache port; see isaprobe).
static f16 c_out[NDIM + 16] __attribute__((section(".data"), aligned(64)));

#if VARIANT == 1
static void vqgemv_vlxblk(f16 *c, const f16 *a, const f16 *b0, const f16 *b1,
                          const idx_t *bi0, const idx_t *bi1,
                          const f16 *sc, unsigned int K, unsigned int N) {
  const unsigned int groups = N / CB_D;
  asm volatile("vsetblklen %0" ::"r"((uint32_t)CB_D));
  for (unsigned int g = 0; g < groups;) {
    size_t gvl;
    asm volatile("vsetvli %[gvl], %[vl], e16, m4, ta, ma"
                 : [gvl] "=r"(gvl) : [vl] "r"((groups - g) * CB_D));
    const unsigned int group_vl = gvl / CB_D;
    asm volatile("vmv.v.x v0, zero" ::: "v0");
    for (unsigned int k = 0; k < K; ++k) {
      float av, scale;
      asm volatile("flh %[av], 0(%[a])" : [av] "=f"(av) : [a] "r"(a + k));
      asm volatile("flh %[s], 0(%[sc])" : [s] "=f"(scale) : [sc] "r"(sc + k));
      // cb0/cb1 MUST be live operands of this block (%[cb0]/%[cb1]) — a
      // pointer pinned to a fixed register before the loop is not guaranteed
      // to survive into a separate later asm statement.
      asm volatile("vsetvli zero, %[gv], " IDX_SEW ", m2, ta, ma\n"
                   VLE_IDX " v28, (%[i0])\n"
                   VLE_IDX " v30, (%[i1])\n"
                   "vsetvli zero, %[gvl], e16, m4, ta, ma\n"
                   VLXBLK_IDX " v16, (%[cb0]), v28\n"
                   VLXBLK_IDX " v20, (%[cb1]), v30\n"
                   "vfadd.vv v16, v16, v20\n"
                   "vfmul.vf v16, v16, %[scale]\n"
                   "vfmacc.vf v0, %[av], v16\n"
                   :
                   : [gv] "r"(group_vl), [gvl] "r"(gvl),
                     [i0] "r"(bi0 + k * groups + g),
                     [i1] "r"(bi1 + k * groups + g),
                     [cb0] "r"(b0), [cb1] "r"(b1),
                     [scale] "f"(scale), [av] "f"(av)
                   : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                     "v28", "v29", "v30", "v31", "memory");
    }
    asm volatile("vse16.v v0, (%0)" ::"r"(c + g * CB_D) : "memory");
    g += group_vl;
  }
}
#else
static void vqgemv_rvv(f16 *c, const f16 *a, const f16 *b0, const f16 *b1,
                       const idx_t *bi0, const idx_t *bi1, const f16 *sc,
                       unsigned int K, unsigned int N) {
  const unsigned int groups = N / CB_D;
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(CB_D));
  for (unsigned int g = 0; g < groups; ++g) {
    asm volatile("vmv.v.x v0, zero" ::: "v0");
    for (unsigned int k = 0; k < K; ++k) {
      const unsigned int idx = k * groups + g;
      const f16 *p0 = b0 + ((unsigned int)bi0[idx] * CB_D);
      const f16 *p1 = b1 + ((unsigned int)bi1[idx] * CB_D);
      float av, scale;
      asm volatile("flh %[av], 0(%[a])" : [av] "=f"(av) : [a] "r"(a + k));
      asm volatile("flh %[s], 0(%[sc])" : [s] "=f"(scale) : [sc] "r"(sc + k));
      asm volatile("vle16.v v16, (%[p0])\n"
                   "vle16.v v20, (%[p1])\n"
                   "vfadd.vv v16, v16, v20\n"
                   "vfmul.vf v16, v16, %[scale]\n"
                   "vfmacc.vf v0, %[av], v16\n"
                   :
                   : [p0] "r"(p0), [p1] "r"(p1),
                     [scale] "f"(scale), [av] "f"(av)
                   : "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
                     "memory");
    }
    // Store padded to >= 32 B (16-B vector stores hang the cache port;
    // ascending g means each group's slot is last written by its own
    // store, so upper-lane garbage is always overwritten; c_out is
    // padded for the final group).
#if CB_D < 16
    asm volatile("vsetvli zero, %[pv], e16, m1, ta, ma\n"
                 "vse16.v v0, (%[c])\n"
                 :: [pv] "r"(16u), [c] "r"(c + g * CB_D) : "memory");
    asm volatile("vsetvli zero, %[d], e16, m1, ta, ma" ::[d] "r"((uint32_t)CB_D));
#else
    asm volatile("vse16.v v0, (%0)" ::"r"(c + g * CB_D) : "memory");
#endif
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
    for (unsigned int k = 0; k < KDIM; ++k) {
      a_vec[k]  = (f16)(0.02f + 0.001f * (float)(k % 51));
      scales[k] = (f16)(0.5f + 0.01f * (float)(k % 7));
    }
    // Codebooks: prime-period pattern head (1021 f16, non-repeating
    // inside the head) + vector replication — scalar-filling the VPTQ
    // 128 KiB tables would dominate sim wall-clock (bench_fill.h).
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
    memset(c_out, 0, sizeof(c_out));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    vqgemv_vlxblk(c_out, a_vec, cb0, cb1, idx0, idx1, scales, KDIM, NDIM);
#else
    vqgemv_rvv(c_out, a_vec, cb0, cb1, idx0, idx1, scales, KDIM, NDIM);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // SAMPLED golden check (the scalar-core reference dominates sim
    // wall-clock at full coverage): all groups, d in {0, CB_D/2,
    // CB_D-1}; per-k factors hoisted once. fp16 accumulate in the
    // kernel -> tolerance below.
    static float ref_ask[KDIM];
    for (unsigned int k = 0; k < KDIM; ++k)
      ref_ask[k] = f16f(&a_vec[k]) * f16f(&scales[k]);
    const unsigned int dsel[3] = {0u, CB_D / 2u, CB_D - 1u};
    for (unsigned int g = 0; g < GROUPS && fails == 0; ++g)
      for (unsigned int di = 0; di < 3 && fails == 0; ++di) {
        const unsigned int d = dsel[di];
        float acc = 0.0f;
        for (unsigned int k = 0; k < KDIM; ++k) {
          unsigned int ix = k * GROUPS + g;
          float w = f16f(&cb0[(unsigned int)idx0[ix] * CB_D + d]) +
                    f16f(&cb1[(unsigned int)idx1[ix] * CB_D + d]);
          acc += ref_ask[k] * w;
        }
        float got = f16f(&c_out[g * CB_D + d]);
        float err = got > acc ? got - acc : acc - got;
        float mag = acc < 0 ? -acc : acc;
        if (err > 0.25f + 0.015625f * mag) {
          printf("FAILED g=%d d=%d got=%f exp=%f\n", g, d, got, acc);
          fails = 1;
        }
      }

    // GEMV MACs = K*N (the vfmacc); + K*N decode adds + K*N muls.
    printf("vqgemv variant=%d K=%d N=%d cache=%d: took %u cycles %s "
           "(macs=%u idxKiB=%u)\n", VARIANT, KDIM, NDIM, USE_CACHE, cycles,
           fails ? "CHECK-FAILED" : "CHECK-OK", (unsigned)(KDIM * NDIM),
           (unsigned)(2u * KDIM * GROUPS * sizeof(idx_t) / 1024));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
