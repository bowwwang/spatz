// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, VLXBLK arm: GROMACS-style 4x4 cluster-pair non-bonded force
// kernel (Pall & Hess 2013; Pall et al. 2020), fp32. The domain is
// resident as four parallel X4 field arrays (x, y, z, q; 4 atoms x fp32 =
// one 16-B block per cluster), bare u16 cluster ids as entry numbers.
//
// Chunk = 8 j-clusters = 32 fp32 = one e32 m1 register per field: one
// 64-B index load (32 ids, the first 8 used) drives four 128-B block
// gathers (x, y, z, q) that share the index vector; the gathered j-data
// is reused by all 4 i-atoms. Single vtype e32 m1 vl=32 throughout (the
// vle16 index load uses its own EEW). Every load group is consumed at
// the LMUL it was loaded with: Spatz tracks register hazards per base
// register only (erratum #7), so m2 gathers read as m1 halves deadlock.
//
// Pair force (simplified, FMA-only, 17 FLOP/pair) per i-atom a, 13 ops:
//   d = ri - rj; r2 = dx^2+dy^2+dz^2; w = max(cut2 - r2, 0); s = w^2*qj;
//   f_a += s * d, accumulated LANE-WISE (vfmacc) in 12 vector accumulators
// (4 atoms x 3 components, v20-31); the first chunk of an i-cluster
// initializes them with vfmul (no zeroing). The 12 cross-lane reductions
// (seeded by a zeroed temp) and the 12 scalar read-backs happen once per
// i-cluster, batched: a reduction is ~50 cycles on Spatz.
//
// Two-round software pipeline over the flat chunk sequence: set A (fields
// v8-11) and set B (v12-15) alternate, index vector v4 shared (its next
// load waits in order behind the previous gathers). The NEXT chunk's five
// memory instructions are INTERLEAVED into the current chunk's four atom
// bodies (index + x before atom 0, y / z / q before atoms 1 / 2 / 3): the
// VLSU queue holds one pending instruction, so back-to-back memory
// instructions would block the in-order dispatch of the arithmetic.
// Prefetch runs across i-cluster boundaries (the pair list carries 64 ids
// of padding). Temps v16-19 (dx, dy, dz, s). i-atom coordinates and cut2
// reach the FPRs through integer loads + fmv.w.x (no FPU loads under
// vector traffic, erratum #5). Requirements: list a multiple of 16.
#include "nbforce-vlxblk.h"
static inline void nbf_chunk_a_first(const uint16_t *nids, const float *nb_x, const float *nb_y, const float *nb_z, const float *nb_q,
    const unsigned int lanes2, const unsigned int lanes1, const float xi0, const float yi0, const float zi0, const float xi1,
    const float yi1, const float zi1, const float xi2, const float yi2, const float zi2, const float xi3, const float yi3, const float zi3,
    const float cut2, const float fzero, const float fone) {
  // (m1 gathers: no vtype change)
  asm volatile("vle16.v v4, (%0)" ::"r"(nids) : "memory");
  asm volatile("vlxblkei16.v v12, (%0), v4" ::"r"(nb_x) : "memory");
  // (m1)
  // i-atom 0, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi0));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi0));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi0));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmul.vv v20, v19, v16");
  asm volatile("vfmul.vv v21, v19, v17");
  asm volatile("vfmul.vv v22, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v13, (%0), v4" ::"r"(nb_y) : "memory");
  // (m1)
  // i-atom 1, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi1));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi1));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi1));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmul.vv v23, v19, v16");
  asm volatile("vfmul.vv v24, v19, v17");
  asm volatile("vfmul.vv v25, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v14, (%0), v4" ::"r"(nb_z) : "memory");
  // (m1)
  // i-atom 2, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi2));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi2));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi2));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmul.vv v26, v19, v16");
  asm volatile("vfmul.vv v27, v19, v17");
  asm volatile("vfmul.vv v28, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v15, (%0), v4" ::"r"(nb_q) : "memory");
  // (m1)
  // i-atom 3, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi3));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi3));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi3));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmul.vv v29, v19, v16");
  asm volatile("vfmul.vv v30, v19, v17");
  asm volatile("vfmul.vv v31, v19, v18");
}
static inline void nbf_chunk_a(const uint16_t *nids, const float *nb_x, const float *nb_y, const float *nb_z, const float *nb_q,
    const unsigned int lanes2, const unsigned int lanes1, const float xi0, const float yi0, const float zi0, const float xi1,
    const float yi1, const float zi1, const float xi2, const float yi2, const float zi2, const float xi3, const float yi3, const float zi3,
    const float cut2, const float fzero, const float fone) {
  // (m1 gathers: no vtype change)
  asm volatile("vle16.v v4, (%0)" ::"r"(nids) : "memory");
  asm volatile("vlxblkei16.v v12, (%0), v4" ::"r"(nb_x) : "memory");
  // (m1)
  // i-atom 0, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi0));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi0));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi0));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmacc.vv v20, v19, v16");
  asm volatile("vfmacc.vv v21, v19, v17");
  asm volatile("vfmacc.vv v22, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v13, (%0), v4" ::"r"(nb_y) : "memory");
  // (m1)
  // i-atom 1, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi1));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi1));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi1));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmacc.vv v23, v19, v16");
  asm volatile("vfmacc.vv v24, v19, v17");
  asm volatile("vfmacc.vv v25, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v14, (%0), v4" ::"r"(nb_z) : "memory");
  // (m1)
  // i-atom 2, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi2));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi2));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi2));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmacc.vv v26, v19, v16");
  asm volatile("vfmacc.vv v27, v19, v17");
  asm volatile("vfmacc.vv v28, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v15, (%0), v4" ::"r"(nb_q) : "memory");
  // (m1)
  // i-atom 3, half 0
  asm volatile("vfrsub.vf v16, v8, %0" ::"f"(xi3));
  asm volatile("vfrsub.vf v17, v9, %0" ::"f"(yi3));
  asm volatile("vfrsub.vf v18, v10, %0" ::"f"(zi3));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v11");
  asm volatile("vfmacc.vv v29, v19, v16");
  asm volatile("vfmacc.vv v30, v19, v17");
  asm volatile("vfmacc.vv v31, v19, v18");
}
static inline void nbf_chunk_b(const uint16_t *nids, const float *nb_x, const float *nb_y, const float *nb_z, const float *nb_q,
    const unsigned int lanes2, const unsigned int lanes1, const float xi0, const float yi0, const float zi0, const float xi1,
    const float yi1, const float zi1, const float xi2, const float yi2, const float zi2, const float xi3, const float yi3, const float zi3,
    const float cut2, const float fzero, const float fone) {
  // (m1 gathers: no vtype change)
  asm volatile("vle16.v v4, (%0)" ::"r"(nids) : "memory");
  asm volatile("vlxblkei16.v v8, (%0), v4" ::"r"(nb_x) : "memory");
  // (m1)
  // i-atom 0, half 0
  asm volatile("vfrsub.vf v16, v12, %0" ::"f"(xi0));
  asm volatile("vfrsub.vf v17, v13, %0" ::"f"(yi0));
  asm volatile("vfrsub.vf v18, v14, %0" ::"f"(zi0));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v15");
  asm volatile("vfmacc.vv v20, v19, v16");
  asm volatile("vfmacc.vv v21, v19, v17");
  asm volatile("vfmacc.vv v22, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v9, (%0), v4" ::"r"(nb_y) : "memory");
  // (m1)
  // i-atom 1, half 0
  asm volatile("vfrsub.vf v16, v12, %0" ::"f"(xi1));
  asm volatile("vfrsub.vf v17, v13, %0" ::"f"(yi1));
  asm volatile("vfrsub.vf v18, v14, %0" ::"f"(zi1));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v15");
  asm volatile("vfmacc.vv v23, v19, v16");
  asm volatile("vfmacc.vv v24, v19, v17");
  asm volatile("vfmacc.vv v25, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v10, (%0), v4" ::"r"(nb_z) : "memory");
  // (m1)
  // i-atom 2, half 0
  asm volatile("vfrsub.vf v16, v12, %0" ::"f"(xi2));
  asm volatile("vfrsub.vf v17, v13, %0" ::"f"(yi2));
  asm volatile("vfrsub.vf v18, v14, %0" ::"f"(zi2));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v15");
  asm volatile("vfmacc.vv v26, v19, v16");
  asm volatile("vfmacc.vv v27, v19, v17");
  asm volatile("vfmacc.vv v28, v19, v18");
  // (m1 gathers: no vtype change)
  asm volatile("vlxblkei16.v v11, (%0), v4" ::"r"(nb_q) : "memory");
  // (m1)
  // i-atom 3, half 0
  asm volatile("vfrsub.vf v16, v12, %0" ::"f"(xi3));
  asm volatile("vfrsub.vf v17, v13, %0" ::"f"(yi3));
  asm volatile("vfrsub.vf v18, v14, %0" ::"f"(zi3));
  asm volatile("vfmul.vv v19, v16, v16");
  asm volatile("vfmacc.vv v19, v17, v17");
  asm volatile("vfmacc.vv v19, v18, v18");
  asm volatile("vfrsub.vf v19, v19, %0" ::"f"(cut2));
  asm volatile("vfmax.vf v19, v19, %0" ::"f"(fzero));
  asm volatile("vfmul.vv v19, v19, v19");
  asm volatile("vfmul.vv v19, v19, v15");
  asm volatile("vfmacc.vv v29, v19, v16");
  asm volatile("vfmacc.vv v30, v19, v17");
  asm volatile("vfmacc.vv v31, v19, v18");
}
void nbforce_vlxblk(float *fo, const float *nb_x, const float *nb_y, const float *nb_z, const float *nb_q,
                    const uint16_t *pairlist, const unsigned int nc, const unsigned int list, const uint32_t cut2_bits) {
  const unsigned int cl = 4u, lanes2 = 64u, lanes1 = 32u, chunk = 8u;
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(lanes1)); (void)lanes2;
  const uint32_t *xw = (const uint32_t *)nb_x; const uint32_t *yw = (const uint32_t *)nb_y; const uint32_t *zw = (const uint32_t *)nb_z;
  const uint16_t *ids = pairlist; float *f = fo; float cut2, fzero;
  float xi0, yi0, zi0, xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3;
  float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
  asm volatile("fmv.w.x %0, %1" : "=f"(cut2) : "r"(cut2_bits));
  asm volatile("fmv.w.x %0, zero" : "=f"(fzero));
  float fone; asm volatile("fmv.w.x %0, %1" : "=f"(fone) : "r"(0x3f800000u));
  asm volatile("vsetblklen %0" ::"r"(cl));
  // (m1 gathers: no vtype change)
  asm volatile("vle16.v v4, (%0)" ::"r"(ids) : "memory");
  asm volatile("vlxblkei16.v v8, (%0), v4" ::"r"(nb_x) : "memory");
  asm volatile("vlxblkei16.v v9, (%0), v4" ::"r"(nb_y) : "memory");
  asm volatile("vlxblkei16.v v10, (%0), v4" ::"r"(nb_z) : "memory");
  asm volatile("vlxblkei16.v v11, (%0), v4" ::"r"(nb_q) : "memory");
  // (m1)
  for (unsigned int c = 0; c < nc; ++c) {
    asm volatile("fmv.w.x %0, %1" : "=f"(xi0) : "r"(xw[c * cl + 0]));
    asm volatile("fmv.w.x %0, %1" : "=f"(xi1) : "r"(xw[c * cl + 1]));
    asm volatile("fmv.w.x %0, %1" : "=f"(xi2) : "r"(xw[c * cl + 2]));
    asm volatile("fmv.w.x %0, %1" : "=f"(xi3) : "r"(xw[c * cl + 3]));
    asm volatile("fmv.w.x %0, %1" : "=f"(yi0) : "r"(yw[c * cl + 0]));
    asm volatile("fmv.w.x %0, %1" : "=f"(yi1) : "r"(yw[c * cl + 1]));
    asm volatile("fmv.w.x %0, %1" : "=f"(yi2) : "r"(yw[c * cl + 2]));
    asm volatile("fmv.w.x %0, %1" : "=f"(yi3) : "r"(yw[c * cl + 3]));
    asm volatile("fmv.w.x %0, %1" : "=f"(zi0) : "r"(zw[c * cl + 0]));
    asm volatile("fmv.w.x %0, %1" : "=f"(zi1) : "r"(zw[c * cl + 1]));
    asm volatile("fmv.w.x %0, %1" : "=f"(zi2) : "r"(zw[c * cl + 2]));
    asm volatile("fmv.w.x %0, %1" : "=f"(zi3) : "r"(zw[c * cl + 3]));
    nbf_chunk_a_first(ids + chunk, nb_x, nb_y, nb_z, nb_q, lanes2, lanes1, xi0, yi0, zi0, xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3, cut2, fzero, fone);
    nbf_chunk_b(ids + 2 * chunk, nb_x, nb_y, nb_z, nb_q, lanes2, lanes1, xi0, yi0, zi0, xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3, cut2, fzero, fone);
    ids += 2 * chunk;
    for (unsigned int j = 2 * chunk; j < list; j += 2 * chunk) {
      nbf_chunk_a(ids + chunk, nb_x, nb_y, nb_z, nb_q, lanes2, lanes1, xi0, yi0, zi0, xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3, cut2, fzero, fone);
      nbf_chunk_b(ids + 2 * chunk, nb_x, nb_y, nb_z, nb_q, lanes2, lanes1, xi0, yi0, zi0, xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3, cut2, fzero, fone);
      ids += 2 * chunk;
    }
    asm volatile("vmv.v.i v16, 0");
    asm volatile("vfredusum.vs v20, v20, v16");
    asm volatile("vfredusum.vs v21, v21, v16");
    asm volatile("vfredusum.vs v22, v22, v16");
    asm volatile("vfredusum.vs v23, v23, v16");
    asm volatile("vfredusum.vs v24, v24, v16");
    asm volatile("vfredusum.vs v25, v25, v16");
    asm volatile("vfredusum.vs v26, v26, v16");
    asm volatile("vfredusum.vs v27, v27, v16");
    asm volatile("vfredusum.vs v28, v28, v16");
    asm volatile("vfredusum.vs v29, v29, v16");
    asm volatile("vfredusum.vs v30, v30, v16");
    asm volatile("vfredusum.vs v31, v31, v16");
    asm volatile("vfmv.f.s %0, v20" : "=f"(s0));
    asm volatile("vfmv.f.s %0, v21" : "=f"(s1));
    asm volatile("vfmv.f.s %0, v22" : "=f"(s2));
    asm volatile("vfmv.f.s %0, v23" : "=f"(s3));
    asm volatile("vfmv.f.s %0, v24" : "=f"(s4));
    asm volatile("vfmv.f.s %0, v25" : "=f"(s5));
    asm volatile("vfmv.f.s %0, v26" : "=f"(s6));
    asm volatile("vfmv.f.s %0, v27" : "=f"(s7));
    asm volatile("vfmv.f.s %0, v28" : "=f"(s8));
    asm volatile("vfmv.f.s %0, v29" : "=f"(s9));
    asm volatile("vfmv.f.s %0, v30" : "=f"(s10));
    asm volatile("vfmv.f.s %0, v31" : "=f"(s11));
    f[0] = s0;
    f[1] = s1;
    f[2] = s2;
    f[3] = s3;
    f[4] = s4;
    f[5] = s5;
    f[6] = s6;
    f[7] = s7;
    f[8] = s8;
    f[9] = s9;
    f[10] = s10;
    f[11] = s11;
    f += 12;
  }
}
