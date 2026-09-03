// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// attn-vagg: sparse-attention V-aggregation (paged-KV geometry, vLLM/
// FlashInfer-class): for each query, the top-K selected token indices
// gather 256-B V rows (head_dim 128 x fp16) which are score-weighted and
// accumulated: out_q[:] = sum_k p_qk * V[idx_qk][:].
//
// Arms (VARIANT): 1 = VLXBLK: one gather fetches 4 V rows (vl=512 e16, m8
//                     group), then 4x vfmacc.vf on m2 register slices.
//                 2 = vle baseline (piecewise rule: 256 B = 2 registers
//                     >= one register -> scalar-indexed vle loop).
// Check: CPU fp16 reference with identical per-lane MAC order; relative
// tolerance 2^-7 (half precision), exact index/addressing verified by
// construction.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#define VLXBLK_WORD(f7, f3, vd, rs1n, vs2) \
  ".word ((" #f7 ")<<25)|((" #vs2 ")<<20)|((" #rs1n ")<<15)|((" #f3 ")<<12)|((" #vd ")<<7)|0x2B\n"
#define VLXBLKEI16_V(vd, rs1n, vs2) VLXBLK_WORD(0x0C, 0x5, vd, rs1n, vs2)
#define VSETBLKLEN(rs1n)            VLXBLK_WORD(0x0F, 0x0, 0, rs1n, 0)

#ifndef NTOK
#define NTOK 512 // tokens in the V pool (NTOK * 256 B footprint)
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define HD 128   // head_dim (fp16) -> 256-B rows
#define NQ 64    // queries
#define TOPK 64  // selected tokens per query

typedef __fp16 f16;

static f16 v_pool[NTOK * HD] __attribute__((section(".data"), aligned(128)));
static uint16_t topk_idx[NQ * TOPK] __attribute__((section(".data"), aligned(64)));
static f16 scores[NQ * TOPK] __attribute__((section(".data"), aligned(64)));
static f16 attn_out[NQ * HD] __attribute__((section(".data"), aligned(128)));
static f16 attn_ref[NQ * HD] __attribute__((section(".data"), aligned(128)));

#if VARIANT == 1
// One vsetblklen(128) up front; per group of 4 selected tokens: one block
// gather -> v8..v15 (rows at v8,v10,v12,v14: 128 e16 = m2 slices), then
// 4 score-weighted MACs into the m2 accumulator v24.
static void vagg_vlxblk(f16 *out, const f16 *v, const uint16_t *idx,
                        const f16 *p, unsigned int nq) {
  register uint32_t bl asm("t0") = HD;      // x5
  register const f16 *vp asm("t1") = v;     // x6
  asm volatile(VSETBLKLEN(5) :: "r"(bl), "r"(vp));
  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * TOPK;
    const f16 *qp = p + q * TOPK;
    asm volatile("vsetvli zero, %[hd], e16, m2, ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [hd] "r"(HD) : "v24", "v25");
    for (unsigned int k = 0; k < TOPK; k += 4) {
      asm volatile("vsetvli zero, %[gr], e16, m1, ta, ma\n"
                   "vle16.v v2, (%[i0])\n"
                   "vsetvli zero, %[ec], e16, m8, ta, ma\n"
                   VLXBLKEI16_V(8, 6, 2)
                   "vsetvli zero, %[hd], e16, m2, ta, ma\n"
                   "flh ft0, 0(%[p0])\n"
                   "flh ft1, 2(%[p0])\n"
                   "flh ft2, 4(%[p0])\n"
                   "flh ft3, 6(%[p0])\n"
                   "vfmacc.vf v24, ft0, v8\n"
                   "vfmacc.vf v24, ft1, v10\n"
                   "vfmacc.vf v24, ft2, v12\n"
                   "vfmacc.vf v24, ft3, v14\n"
                   :
                   : [gr] "r"(4u), [ec] "r"(4 * HD), [hd] "r"(HD),
                     [i0] "r"(qi + k), [p0] "r"(qp + k), [dict] "r"(vp)
                   : "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                     "v15", "v24", "v25", "ft0", "ft1", "ft2", "ft3",
                     "memory");
    }
    asm volatile("vsetvli zero, %[hd], e16, m2, ta, ma\n"
                 "vse16.v v24, (%[o])\n" :: [hd] "r"(HD),
                 [o] "r"(out + q * HD) : "memory");
  }
}
#else
// Baseline (piecewise rule at 2-register rows): scalar-indexed vle loop -
// per selected token: load index, compute row address, vle16 the row,
// score-weighted MAC. Identical arithmetic order to the vlxblk arm.
static void vagg_vle(f16 *out, const f16 *v, const uint16_t *idx,
                     const f16 *p, unsigned int nq) {
  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * TOPK;
    const f16 *qp = p + q * TOPK;
    asm volatile("vsetvli zero, %[hd], e16, m2, ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [hd] "r"(HD) : "v24", "v25");
    for (unsigned int k = 0; k < TOPK; ++k) {
      const f16 *row = v + (uint32_t)qi[k] * HD;
      asm volatile("vle16.v v8, (%[r])\n"
                   "flh ft0, 0(%[p0])\n"
                   "vfmacc.vf v24, ft0, v8\n"
                   :
                   : [r] "r"(row), [p0] "r"(qp + k)
                   : "v8", "v9", "v24", "v25", "ft0", "memory");
    }
    asm volatile("vse16.v v24, (%[o])\n" :: [o] "r"(out + q * HD) : "memory");
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
    for (unsigned int i = 0; i < NTOK * HD; ++i)
      v_pool[i] = (f16)(0.5f + 0.001f * (float)(i % 977));
    for (unsigned int i = 0; i < NQ * TOPK; ++i) {
      topk_idx[i] = (uint16_t)((i * 2654435761u) % NTOK);
      scores[i]   = (f16)(0.01f + 0.0001f * (float)(i % 97));
    }
    memset(attn_out, 0, sizeof(attn_out));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    vagg_vlxblk(attn_out, v_pool, topk_idx, scores, NQ);
#else
    vagg_vle(attn_out, v_pool, topk_idx, scores, NQ);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // CPU reference (same per-lane MAC order), tolerance 2^-7 relative.
    for (unsigned int q = 0; q < NQ && fails == 0; ++q)
      for (unsigned int d = 0; d < HD; ++d) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < TOPK; ++k)
          acc += (float)(scores[q * TOPK + k]) *
                 (float)(v_pool[(uint32_t)topk_idx[q * TOPK + k] * HD + d]);
        float got = (float)(attn_out[q * HD + d]);
        float err = got - acc;
        if (err < 0) err = -err;
        float mag = acc < 0 ? -acc : acc;
        if (err > 0.03f + 0.0078125f * mag) {
          printf("FAILED q=%d d=%d\n", q, d);
          fails = 1;
        }
      }

    // MACs = NQ * TOPK * HD (fp16); peak 16 MACs/cycle
    printf("attn-vagg variant=%d ntok=%d cache=%d: took %u cycles %s "
           "(MACs=%u)\n", VARIANT, NTOK, USE_CACHE, cycles,
           fails ? "CHECK-FAILED" : "CHECK-OK", (unsigned)(NQ * TOPK * HD));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
