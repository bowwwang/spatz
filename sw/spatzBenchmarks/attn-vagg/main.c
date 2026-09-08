// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// attn-vagg (paper: spattn): top-K sparse-attention V-projection
// (Quest-class selection, FlashInfer page_size=1 records): per query,
// the top-K token indices gather 256-B V rows (head_dim 128 x fp16),
// score-weighted and accumulated: out_q = sum_k p_qk * V[t_qk] = Y^T p.
// Config: 16K-token V cache (4 MiB), token budget TOPK=1024 (Quest's
// near-lossless point at this context class), NQ=16 decode steps.
//
// Arms (VARIANT): 1 = VLXBLK: per token one gather of the whole 256-B
//                     row — 128 e16 = EXACTLY m4 — in its natural
//                     position, then one vfmacc.vf with the score into
//                     the m4 accumulator. No register-group slicing
//                     (the old 4-rows-at-m8 + m2-slice variant both
//                     clamped vl at m2=64 < HD and mis-sliced the
//                     group: m8 @ e16 holds TWO 256-B rows, not four).
//                 2 = vle baseline (piecewise rule at 4-register rows:
//                     scalar-indexed vle loop), same m4 shape.
// Check: CPU reference via hardware fp16 loads (software (float)__fp16
// casts are broken on this toolchain); fp16 accumulation over 1024
// terms diverges from the float reference by a random walk of fp16
// roundings, so the tolerance is 2^-6 relative.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

// Native VLXBLK mnemonics (LLVM 14 + MC-layer patch); x-register
// form keeps the numeric rs1n interface, so call sites are unchanged.
#define VLXBLKEI16_V(vd, rs1n, vs2)  "vlxblkei16.v v" #vd ", (x" #rs1n "), v" #vs2 "\n"
#define VSETBLKLEN(rs1n)             "vsetblklen x" #rs1n "\n"

#ifndef NTOK
#define NTOK 16384 // tokens in the V pool (NTOK * 256 B footprint)
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define HD 128    // head_dim (fp16) -> 256-B rows = exactly m4 @ e16
#define NQ 16     // decode steps (T in the paper table)
#define TOPK 1024 // token budget per query (P in the paper table)

// Safe __fp16 -> float for the CPU reference (software conversion is
// broken on this toolchain; see vqgemv).
static inline float f16f(const __fp16 *p) {
  float x;
  asm("flh ft0, 0(%1)\n\t"
      "fcvt.s.h %0, ft0"
      : "=f"(x)
      : "r"(p)
      : "ft0");
  return x;
}

typedef __fp16 f16;

static f16 v_pool[NTOK * HD] __attribute__((section(".data"), aligned(128)));
static uint16_t topk_idx[NQ * TOPK] __attribute__((section(".data"), aligned(64)));
static f16 scores[NQ * TOPK] __attribute__((section(".data"), aligned(64)));
static f16 attn_out[NQ * HD] __attribute__((section(".data"), aligned(128)));
static f16 attn_ref[NQ * HD] __attribute__((section(".data"), aligned(128)));

#if VARIANT == 1
// One vsetblklen(128) up front; per selected token: one block gather of
// the whole 256-B row (m4 in its natural position), one score vfmacc.vf
// into the m4 accumulator v24-v27.
static void vagg_vlxblk(f16 *out, const f16 *v, const uint16_t *idx,
                        const f16 *p, unsigned int nq) {
  register uint32_t bl asm("t0") = HD;      // x5
  register const f16 *vp asm("t1") = v;     // x6
  asm volatile(VSETBLKLEN(5) :: "r"(bl), "r"(vp));
  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * TOPK;
    const f16 *qp = p + q * TOPK;
    asm volatile("vsetvli zero, %[hd], e16, m4, ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [hd] "r"(HD)
                 : "v24", "v25", "v26", "v27");
    for (unsigned int k = 0; k < TOPK; ++k) {
      asm volatile("vsetvli zero, %[one], e16, m1, ta, ma\n"
                   "vle16.v v2, (%[i0])\n"
                   "vsetvli zero, %[hd], e16, m4, ta, ma\n"
                   VLXBLKEI16_V(8, 6, 2)
                   "flh ft0, 0(%[p0])\n"
                   "vfmacc.vf v24, ft0, v8\n"
                   :
                   : [one] "r"(1u), [hd] "r"(HD), [i0] "r"(qi + k),
                     [p0] "r"(qp + k), [dict] "r"(vp)
                   : "v2", "v8", "v9", "v10", "v11", "v24", "v25", "v26",
                     "v27", "ft0", "memory");
    }
    asm volatile("vsetvli zero, %[hd], e16, m4, ta, ma\n"
                 "vse16.v v24, (%[o])\n" :: [hd] "r"(HD),
                 [o] "r"(out + q * HD) : "memory");
  }
}
#else
// Baseline (piecewise rule at 4-register rows): scalar-indexed vle loop -
// per selected token: load index, compute row address, vle16 the row,
// score-weighted MAC. Identical arithmetic order to the vlxblk arm.
static void vagg_vle(f16 *out, const f16 *v, const uint16_t *idx,
                     const f16 *p, unsigned int nq) {
  for (unsigned int q = 0; q < nq; ++q) {
    const uint16_t *qi = idx + q * TOPK;
    const f16 *qp = p + q * TOPK;
    asm volatile("vsetvli zero, %[hd], e16, m4, ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [hd] "r"(HD)
                 : "v24", "v25", "v26", "v27");
    for (unsigned int k = 0; k < TOPK; ++k) {
      const f16 *row = v + (uint32_t)qi[k] * HD;
      asm volatile("vle16.v v8, (%[r])\n"
                   "flh ft0, 0(%[p0])\n"
                   "vfmacc.vf v24, ft0, v8\n"
                   :
                   : [r] "r"(row), [p0] "r"(qp + k)
                   : "v8", "v9", "v10", "v11", "v24", "v25", "v26", "v27",
                     "ft0", "memory");
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
    // V pool: prime-period pattern head (1021 f16) + vector replication
    // (bench_fill.h) — a scalar fill of the 4 MiB pool dominates sim
    // wall-clock. Index/score arrays are small enough for scalar init
    // (masks, no divisions).
    {
      const unsigned int n = NTOK * HD;
      const unsigned int head = n < 1952u ? n : 1952u; // f16: BF_HEAD_BYTES/2
      for (unsigned int i = 0; i < head; ++i)
        v_pool[i] = (f16)(0.5f + 0.0005f * (float)((i * 37u) & 2047u));
      if (n > head)
        bench_fill_rep(v_pool, n * 2u, head * 2u);
    }
#if (NTOK & (NTOK - 1)) != 0
#error "NTOK must be a power of two (masked id generation)"
#endif
    for (unsigned int i = 0; i < NQ * TOPK; ++i) {
      topk_idx[i] = (uint16_t)((i * 2654435761u) & (NTOK - 1u));
      scores[i] = (f16)(0.01f + 0.0001f * (float)((i * 29u) & 127u));
    }
    bench_fill_zero(attn_out, sizeof(attn_out));
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

    // SAMPLED golden check (the scalar-core reference dominates sim
    // wall-clock at full coverage): queries {0,5,10,15}, d in
    // {0,32,64,96,HD-1}; per-q scores hoisted once. Kernel accumulates
    // in fp16 over TOPK=1024 terms vs the float reference: tolerance
    // 2^-6 relative for the rounding random walk.
    static float ref_sc[TOPK];
    const unsigned int qsel[4] = {0u, 5u, 10u, 15u};
    const unsigned int dsel[5] = {0u, 32u, 64u, 96u, HD - 1u};
    for (unsigned int qs = 0; qs < 4 && fails == 0; ++qs) {
      const unsigned int q = qsel[qs];
      for (unsigned int k = 0; k < TOPK; ++k)
        ref_sc[k] = f16f(&scores[q * TOPK + k]);
      for (unsigned int ds = 0; ds < 5 && fails == 0; ++ds) {
        const unsigned int d = dsel[ds];
        float acc = 0.0f;
        for (unsigned int k = 0; k < TOPK; ++k)
          acc += ref_sc[k] *
                 f16f(&v_pool[(uint32_t)topk_idx[q * TOPK + k] * HD + d]);
        float got = f16f(&attn_out[q * HD + d]);
        float err = got - acc;
        if (err < 0) err = -err;
        float mag = acc < 0 ? -acc : acc;
        if (err > 0.05f + 0.015625f * mag) {
          printf("FAILED q=%d d=%d got=%f exp=%f\n", q, d, got, acc);
          fails = 1;
        }
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
