// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// SCALAR OPERANDS WITHOUT flh (RTL erratum #5, 2026-09-08: an flh result
// consumed by a .vf instruction can be STALE under vector memory traffic —
// flh runs in Spatz's FPU sequencer and the dependency is not enforced).
// Each fp16 scalar is fetched with an integer lhu — the nine of a step are
// issued together right after the step's first macc, so their latency
// hides behind the vector work — and moved into the f-register with
// fmv.w.x at the point of use. The .vf path reads only the low 16 bits, so
// no NaN-boxing is needed (verified bit-exact). No Zfh dependency. Cost vs
// flh: +18 scalar instructions per step (220,100 vs 169,446 cycles, AQLM);
// switch back to flh once the RTL is fixed.
// vqgemm, VLXBLK arm: C[M,N] = A[M,K] x W[K,N] with the 2-codebook VQ
// decode of W fused into the product (AQLM/VPTQ-class residual VQ):
// w_k[N] = scale[k] * (cb0[idx0[k,g]] + cb1[idx1[k,g]]), group g of lane
// range [g*cb_d, (g+1)*cb_d) being entry idx[k,g], exactly as in vqgemv.
//
// 8-ROW BLOCKING (after hp-fmatmul's matmul_8xVL and the tuned
// hp-dqmatmul-blk32 kernel): 8 output rows live in the VRF as 8 m2
// accumulators (v0, v2, ..., v14; at VLEN = 1024 one m2 group is 128 fp16 =
// the whole N = 128 row), so every decoded row w_k feeds EIGHT vfmacc.vf.
// The decode is redone only M/8 times, and per step the VFU spends 8 of
// its 10 ops on useful MACs.
//
// SOFTWARE PIPELINE, single decode buffer alternating v16/v18: step s
// accumulates row s (buffer CUR) while decoding row s+1 (buffer NXT).
// In-step order matters and follows the tuned reference:
//   1. the two index loads and the cb0 gather for row s+1 go out FIRST;
//   2. the eight maccs of row s run, interleaved with the fmv.w.x that
//      delivers that row's next scalar (a[m+i][s+1]) — see the scalar-operand
//      note above;
//   3. the cb1 gather is slotted after the first macc;
//   4. the decode arithmetic for row s+1 (vfadd, then vfmul by scale[s+1])
//      comes LAST, so its wait on both gathers hides behind eight maccs of
//      unrelated FP work instead of stalling the issue pipe.
// The first step initializes the accumulators with vfmul.vf instead of
// vfmacc.vf (sp-fmatmul's `if (n == 1) vfmul else vfmacc`), so there is no
// separate zeroing and no WAW race at the tile boundary. Steps are unrolled
// in pairs so the CUR/NXT register names are static; with K even the K-2
// steps after the first split into (K-2)/2 pairs and the epilogue's buffer
// is statically v18.
//   prologue : decode(0) -> v16 ; t_i = a[m+i][0]
//   first    : macc(0) as vfmul into acc ; decode(1) -> v18
//   pairs    : STEP(18->16, s+1) STEP(16->18, s+2) for s = 1, 3, ..., K-3
//   epilogue : macc(K-1) from v18 ; 8 stores
// Requirements: M multiple of 8, K even and >= 4, N <= 128 (one m2 group).
//
// ONE vtype for the whole kernel (e16, m2, vl = N): the index loads carry
// their own EEW (vle8/vle16) and over-read N index elements past the
// current k-row (the data header pads the index arrays).
//
// Register map:
//   v0 v2 v4 v6 v8 v10 v12 v14   the eight row accumulators (m2)
//   v16 / v18                    decode buffers (alternating CUR/NXT, m2)
//   v20                          cb1 gather scratch (m2)
//   v28(-29) v30(-31)            idx0 / idx1 vectors

#include "vqgemm-vlxblk.h"

// ---------------
// u8 indices
// ---------------

void vqgemm_vlxblk_ei8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                       const __fp16 *cb1, const uint8_t *idx0,
                       const uint8_t *idx1, const __fp16 *scales,
                       const unsigned int M, const unsigned int N,
                       const unsigned int K, const unsigned int cb_d) {
  const unsigned int groups = N / cb_d;

  asm volatile("vsetblklen %0" ::"r"(cb_d));
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(N));

  for (unsigned int m = 0; m < M; m += 8) {
    const __fp16 *pa0 = a + (m + 0) * K;
    const __fp16 *pa1 = a + (m + 1) * K;
    const __fp16 *pa2 = a + (m + 2) * K;
    const __fp16 *pa3 = a + (m + 3) * K;
    const __fp16 *pa4 = a + (m + 4) * K;
    const __fp16 *pa5 = a + (m + 5) * K;
    const __fp16 *pa6 = a + (m + 6) * K;
    const __fp16 *pa7 = a + (m + 7) * K;
    const __fp16 *ps = scales;
    const uint8_t *p0 = idx0; // k-row pointers of the row being decoded
    const uint8_t *p1 = idx1;

    float sc, t0, t1, t2, t3, t4, t5, t6, t7;
    unsigned int sc_b, t0_b, t1_b, t2_b, t3_b, t4_b, t5_b, t6_b, t7_b;

    // ---- prologue: decode(0) -> v16, t_i = a[m+i][0] ----
    asm volatile("vle8.v v28, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle8.v v30, (%0)" ::"r"(p1) : "memory");
    asm volatile("vlxblkei8.v v16, (%0), v28" ::"r"(cb0) : "memory");
    asm volatile("vlxblkei8.v v20, (%0), v30" ::"r"(cb1) : "memory");
    asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
    asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
    asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
    asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
    asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
    asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
    asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
    asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
    asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
    asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
    asm volatile("vfadd.vv v16, v16, v20");
    asm volatile("vfmul.vf v16, v16, %0" ::"f"(sc));
    p0 += groups;
    p1 += groups;
    ps += 1;
    pa0 += 1;
    pa1 += 1;
    pa2 += 1;
    pa3 += 1;
    pa4 += 1;
    pa5 += 1;
    pa6 += 1;
    pa7 += 1;

    // ---- first step: macc(0) as vfmul (accumulator init) ; decode(1) -> v18 ----
    asm volatile("vle8.v v28, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle8.v v30, (%0)" ::"r"(p1) : "memory");
    asm volatile("vlxblkei8.v v18, (%0), v28" ::"r"(cb0) : "memory");
    asm volatile("vfmul.vf v0, v16, %0" ::"f"(t0));
    asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
    asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
    asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
    asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
    asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
    asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
    asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
    asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
    asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
    asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
    asm volatile("vlxblkei8.v v20, (%0), v30" ::"r"(cb1) : "memory");
    asm volatile("vfmul.vf v2, v16, %0" ::"f"(t1));
    asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
    asm volatile("vfmul.vf v4, v16, %0" ::"f"(t2));
    asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
    asm volatile("vfmul.vf v6, v16, %0" ::"f"(t3));
    asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
    asm volatile("vfmul.vf v8, v16, %0" ::"f"(t4));
    asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
    asm volatile("vfmul.vf v10, v16, %0" ::"f"(t5));
    asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
    asm volatile("vfmul.vf v12, v16, %0" ::"f"(t6));
    asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
    asm volatile("vfmul.vf v14, v16, %0" ::"f"(t7));
    asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
    asm volatile("vfadd.vv v18, v18, v20");
    asm volatile("vfmul.vf v18, v18, %0" ::"f"(sc));
    p0 += groups;
    p1 += groups;
    ps += 1;
    pa0 += 1;
    pa1 += 1;
    pa2 += 1;
    pa3 += 1;
    pa4 += 1;
    pa5 += 1;
    pa6 += 1;
    pa7 += 1;

    // ---- pairs: STEP(18 -> 16, s+1) ; STEP(16 -> 18, s+2), s = 1, 3, ..., K-3 ----
    for (unsigned int s = 1; s < K - 1; s += 2) {
      // STEP: macc row s from v18 ; decode row s+1 -> v16
      asm volatile("vle8.v v28, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle8.v v30, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei8.v v16, (%0), v28" ::"r"(cb0) : "memory");
      asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
      asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
      asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
      asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
      asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
      asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
      asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
      asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
      asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
      asm volatile("vlxblkei8.v v20, (%0), v30" ::"r"(cb1) : "memory");
      asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
      asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
      asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
      asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
      asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
      asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
      asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
      asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
      asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
      asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
      asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
      asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
      asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
      asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
      asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
      asm volatile("vfadd.vv v16, v16, v20");
      asm volatile("vfmul.vf v16, v16, %0" ::"f"(sc));
      p0 += groups;
      p1 += groups;
      ps += 1;
      pa0 += 1;
      pa1 += 1;
      pa2 += 1;
      pa3 += 1;
      pa4 += 1;
      pa5 += 1;
      pa6 += 1;
      pa7 += 1;

      // STEP: macc row s+1 from v16 ; decode row s+2 -> v18
      asm volatile("vle8.v v28, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle8.v v30, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei8.v v18, (%0), v28" ::"r"(cb0) : "memory");
      asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
      asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
      asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
      asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
      asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
      asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
      asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
      asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
      asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
      asm volatile("vlxblkei8.v v20, (%0), v30" ::"r"(cb1) : "memory");
      asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t1));
      asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
      asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t2));
      asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
      asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t3));
      asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
      asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t4));
      asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
      asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t5));
      asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
      asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t6));
      asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
      asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t7));
      asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
      asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
      asm volatile("vfadd.vv v18, v18, v20");
      asm volatile("vfmul.vf v18, v18, %0" ::"f"(sc));
      p0 += groups;
      p1 += groups;
      ps += 1;
      pa0 += 1;
      pa1 += 1;
      pa2 += 1;
      pa3 += 1;
      pa4 += 1;
      pa5 += 1;
      pa6 += 1;
      pa7 += 1;
    }

    // ---- epilogue: macc row K-1 from v18 ; store the 8 rows ----
    asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
    asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
    asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
    asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
    asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
    asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
    asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
    asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
    asm volatile("vse16.v v0, (%0)" ::"r"(c + (m + 0) * N) : "memory");
    asm volatile("vse16.v v2, (%0)" ::"r"(c + (m + 1) * N) : "memory");
    asm volatile("vse16.v v4, (%0)" ::"r"(c + (m + 2) * N) : "memory");
    asm volatile("vse16.v v6, (%0)" ::"r"(c + (m + 3) * N) : "memory");
    asm volatile("vse16.v v8, (%0)" ::"r"(c + (m + 4) * N) : "memory");
    asm volatile("vse16.v v10, (%0)" ::"r"(c + (m + 5) * N) : "memory");
    asm volatile("vse16.v v12, (%0)" ::"r"(c + (m + 6) * N) : "memory");
    asm volatile("vse16.v v14, (%0)" ::"r"(c + (m + 7) * N) : "memory");
  }
}

// ---------------
// u16 indices
// ---------------

void vqgemm_vlxblk_ei16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                        const __fp16 *cb1, const uint16_t *idx0,
                        const uint16_t *idx1, const __fp16 *scales,
                        const unsigned int M, const unsigned int N,
                        const unsigned int K, const unsigned int cb_d) {
  const unsigned int groups = N / cb_d;

  asm volatile("vsetblklen %0" ::"r"(cb_d));
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(N));

  for (unsigned int m = 0; m < M; m += 8) {
    const __fp16 *pa0 = a + (m + 0) * K;
    const __fp16 *pa1 = a + (m + 1) * K;
    const __fp16 *pa2 = a + (m + 2) * K;
    const __fp16 *pa3 = a + (m + 3) * K;
    const __fp16 *pa4 = a + (m + 4) * K;
    const __fp16 *pa5 = a + (m + 5) * K;
    const __fp16 *pa6 = a + (m + 6) * K;
    const __fp16 *pa7 = a + (m + 7) * K;
    const __fp16 *ps = scales;
    const uint16_t *p0 = idx0;
    const uint16_t *p1 = idx1;

    float sc, t0, t1, t2, t3, t4, t5, t6, t7;
    unsigned int sc_b, t0_b, t1_b, t2_b, t3_b, t4_b, t5_b, t6_b, t7_b;

    // ---- prologue: decode(0) -> v16, t_i = a[m+i][0] ----
    asm volatile("vle16.v v28, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle16.v v30, (%0)" ::"r"(p1) : "memory");
    asm volatile("vlxblkei16.v v16, (%0), v28" ::"r"(cb0) : "memory");
    asm volatile("vlxblkei16.v v20, (%0), v30" ::"r"(cb1) : "memory");
    asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
    asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
    asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
    asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
    asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
    asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
    asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
    asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
    asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
    asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
    asm volatile("vfadd.vv v16, v16, v20");
    asm volatile("vfmul.vf v16, v16, %0" ::"f"(sc));
    p0 += groups;
    p1 += groups;
    ps += 1;
    pa0 += 1;
    pa1 += 1;
    pa2 += 1;
    pa3 += 1;
    pa4 += 1;
    pa5 += 1;
    pa6 += 1;
    pa7 += 1;

    // ---- first step: macc(0) as vfmul (accumulator init) ; decode(1) -> v18 ----
    asm volatile("vle16.v v28, (%0)" ::"r"(p0) : "memory");
    asm volatile("vle16.v v30, (%0)" ::"r"(p1) : "memory");
    asm volatile("vlxblkei16.v v18, (%0), v28" ::"r"(cb0) : "memory");
    asm volatile("vfmul.vf v0, v16, %0" ::"f"(t0));
    asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
    asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
    asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
    asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
    asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
    asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
    asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
    asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
    asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
    asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
    asm volatile("vlxblkei16.v v20, (%0), v30" ::"r"(cb1) : "memory");
    asm volatile("vfmul.vf v2, v16, %0" ::"f"(t1));
    asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
    asm volatile("vfmul.vf v4, v16, %0" ::"f"(t2));
    asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
    asm volatile("vfmul.vf v6, v16, %0" ::"f"(t3));
    asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
    asm volatile("vfmul.vf v8, v16, %0" ::"f"(t4));
    asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
    asm volatile("vfmul.vf v10, v16, %0" ::"f"(t5));
    asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
    asm volatile("vfmul.vf v12, v16, %0" ::"f"(t6));
    asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
    asm volatile("vfmul.vf v14, v16, %0" ::"f"(t7));
    asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
    asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
    asm volatile("vfadd.vv v18, v18, v20");
    asm volatile("vfmul.vf v18, v18, %0" ::"f"(sc));
    p0 += groups;
    p1 += groups;
    ps += 1;
    pa0 += 1;
    pa1 += 1;
    pa2 += 1;
    pa3 += 1;
    pa4 += 1;
    pa5 += 1;
    pa6 += 1;
    pa7 += 1;

    // ---- pairs: STEP(18 -> 16, s+1) ; STEP(16 -> 18, s+2), s = 1, 3, ..., K-3 ----
    for (unsigned int s = 1; s < K - 1; s += 2) {
      asm volatile("vle16.v v28, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v30, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei16.v v16, (%0), v28" ::"r"(cb0) : "memory");
      asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
      asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
      asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
      asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
      asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
      asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
      asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
      asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
      asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
      asm volatile("vlxblkei16.v v20, (%0), v30" ::"r"(cb1) : "memory");
      asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
      asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
      asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
      asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
      asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
      asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
      asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
      asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
      asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
      asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
      asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
      asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
      asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
      asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
      asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
      asm volatile("vfadd.vv v16, v16, v20");
      asm volatile("vfmul.vf v16, v16, %0" ::"f"(sc));
      p0 += groups;
      p1 += groups;
      ps += 1;
      pa0 += 1;
      pa1 += 1;
      pa2 += 1;
      pa3 += 1;
      pa4 += 1;
      pa5 += 1;
      pa6 += 1;
      pa7 += 1;

      asm volatile("vle16.v v28, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v30, (%0)" ::"r"(p1) : "memory");
      asm volatile("vlxblkei16.v v18, (%0), v28" ::"r"(cb0) : "memory");
      asm volatile("vfmacc.vf v0, %0, v16" ::"f"(t0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t0_b) : "r"(pa0));
      asm volatile("lhu %0, 0(%1)" : "=r"(t1_b) : "r"(pa1));
      asm volatile("lhu %0, 0(%1)" : "=r"(t2_b) : "r"(pa2));
      asm volatile("lhu %0, 0(%1)" : "=r"(t3_b) : "r"(pa3));
      asm volatile("lhu %0, 0(%1)" : "=r"(t4_b) : "r"(pa4));
      asm volatile("lhu %0, 0(%1)" : "=r"(t5_b) : "r"(pa5));
      asm volatile("lhu %0, 0(%1)" : "=r"(t6_b) : "r"(pa6));
      asm volatile("lhu %0, 0(%1)" : "=r"(t7_b) : "r"(pa7));
      asm volatile("lhu %0, 0(%1)" : "=r"(sc_b) : "r"(ps));
      asm volatile("fmv.w.x %0, %1" : "=f"(t0) : "r"(t0_b));
      asm volatile("vlxblkei16.v v20, (%0), v30" ::"r"(cb1) : "memory");
      asm volatile("vfmacc.vf v2, %0, v16" ::"f"(t1));
      asm volatile("fmv.w.x %0, %1" : "=f"(t1) : "r"(t1_b));
      asm volatile("vfmacc.vf v4, %0, v16" ::"f"(t2));
      asm volatile("fmv.w.x %0, %1" : "=f"(t2) : "r"(t2_b));
      asm volatile("vfmacc.vf v6, %0, v16" ::"f"(t3));
      asm volatile("fmv.w.x %0, %1" : "=f"(t3) : "r"(t3_b));
      asm volatile("vfmacc.vf v8, %0, v16" ::"f"(t4));
      asm volatile("fmv.w.x %0, %1" : "=f"(t4) : "r"(t4_b));
      asm volatile("vfmacc.vf v10, %0, v16" ::"f"(t5));
      asm volatile("fmv.w.x %0, %1" : "=f"(t5) : "r"(t5_b));
      asm volatile("vfmacc.vf v12, %0, v16" ::"f"(t6));
      asm volatile("fmv.w.x %0, %1" : "=f"(t6) : "r"(t6_b));
      asm volatile("vfmacc.vf v14, %0, v16" ::"f"(t7));
      asm volatile("fmv.w.x %0, %1" : "=f"(t7) : "r"(t7_b));
      asm volatile("fmv.w.x %0, %1" : "=f"(sc) : "r"(sc_b));
      asm volatile("vfadd.vv v18, v18, v20");
      asm volatile("vfmul.vf v18, v18, %0" ::"f"(sc));
      p0 += groups;
      p1 += groups;
      ps += 1;
      pa0 += 1;
      pa1 += 1;
      pa2 += 1;
      pa3 += 1;
      pa4 += 1;
      pa5 += 1;
      pa6 += 1;
      pa7 += 1;
    }

    // ---- epilogue: macc row K-1 from v18 ; store the 8 rows ----
    asm volatile("vfmacc.vf v0, %0, v18" ::"f"(t0));
    asm volatile("vfmacc.vf v2, %0, v18" ::"f"(t1));
    asm volatile("vfmacc.vf v4, %0, v18" ::"f"(t2));
    asm volatile("vfmacc.vf v6, %0, v18" ::"f"(t3));
    asm volatile("vfmacc.vf v8, %0, v18" ::"f"(t4));
    asm volatile("vfmacc.vf v10, %0, v18" ::"f"(t5));
    asm volatile("vfmacc.vf v12, %0, v18" ::"f"(t6));
    asm volatile("vfmacc.vf v14, %0, v18" ::"f"(t7));
    asm volatile("vse16.v v0, (%0)" ::"r"(c + (m + 0) * N) : "memory");
    asm volatile("vse16.v v2, (%0)" ::"r"(c + (m + 1) * N) : "memory");
    asm volatile("vse16.v v4, (%0)" ::"r"(c + (m + 2) * N) : "memory");
    asm volatile("vse16.v v6, (%0)" ::"r"(c + (m + 3) * N) : "memory");
    asm volatile("vse16.v v8, (%0)" ::"r"(c + (m + 4) * N) : "memory");
    asm volatile("vse16.v v10, (%0)" ::"r"(c + (m + 5) * N) : "memory");
    asm volatile("vse16.v v12, (%0)" ::"r"(c + (m + 6) * N) : "memory");
    asm volatile("vse16.v v14, (%0)" ::"r"(c + (m + 7) * N) : "memory");
  }
}
