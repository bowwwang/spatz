// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, plain-RVV baseline: the SAME kernel as the VLXBLK arm with the
// only difference being how a chunk of j-data is fetched. Where the
// extension issues one 32-B index load and four 128-B block gathers, plain
// RVV must expand every j-cluster id into four per-ATOM element indices
// (precomputed in the data header, u32 - u16 cannot address the 1.45-MiB
// domain in bytes), load 32 of them (128 B, 4x the index traffic of the
// extension), scale them to byte offsets and issue four element gathers.
// That index-traffic and per-element-transaction cost IS the measurement.
//
// Chunk = 8 j-clusters = 32 fp32 = one e32 m1 register per field; LIST = 96
// -> 12 chunks per i-cluster (even, so set A always starts an i-cluster).
// Pair force per i-atom a, 13 ops (identical to the VLXBLK arm):
//   d = ri - rj; r2 = dx^2+dy^2+dz^2; w = max(cut2 - r2, 0); s = w^2 * qj;
//   f_a += s * d, accumulated LANE-WISE (vfmacc) in 12 vector accumulators
// (4 atoms x 3 components, v20-31); the first chunk of an i-cluster
// initializes them with vfmul (no zeroing). The 12 cross-lane reductions
// (seeded by the zero register v0) and the 12 scalar read-backs happen ONCE
// per i-cluster, batched - a reduction is ~50 cycles on Spatz, so per-chunk
// reductions would dominate (the v1 mistake: 811 vs 322 cycles per chunk).
//
// Two-round software pipeline: set A (fields v8-11, indices v4) and set B
// (v12-15, v5) alternate, and the next chunk's SIX memory instructions are
// interleaved into the current chunk's four atom bodies (index+shift+x
// before atom 0, y / z / q before atoms 1 / 2 / 3): the VLSU queue holds one
// pending instruction, so back-to-back memory instructions would block the
// in-order dispatch of the arithmetic. Prefetch runs across i-cluster
// boundaries, so nb_list_exp carries 256 elements of padding. Temps v16-19.
// i-atom coordinates and cut2 reach the FPRs through integer loads +
// fmv.w.x (no FPU loads under vector traffic, erratum #5). Every register
// group is produced and consumed at e32 m1 through its base register
// (the whole VRF is zeroed at entry, see the note in the body)
// (erratum #7). Requirements: list a multiple of 16, EEW of the indices ==
// SEW of the data (32).

#include "nbforce-rvv.h"

void nbforce_rvv(float *fo, const float *nb_x, const float *nb_y,
                 const float *nb_z, const float *nb_q,
                 const uint32_t *pairlist_exp, const unsigned int nc,
                 const unsigned int list, const float cut2) {
  const unsigned int cl = 4u;      // atoms per cluster
  const unsigned int lanes = 32u;  // e32 m1 at VLEN=1024 = 8 j-clusters
  const unsigned int chunk = 8u;   // j-clusters per chunk
  const unsigned int estride = 32u; // index elements per chunk (8 cl x 4)
  const uint32_t *xw = (const uint32_t *)nb_x;
  const uint32_t *yw = (const uint32_t *)nb_y;
  const uint32_t *zw = (const uint32_t *)nb_z;
  const uint32_t *ex = pairlist_exp;
  float *f = fo;
  float fzero;
  float xi0, yi0, zi0, xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3;
  float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
  (void)chunk;

  asm volatile("fmv.w.x %0, zero" : "=f"(fzero));

  // Zero the WHOLE vector register file before use (user advice 2026-09-10):
  // four e32 m8 moves cover v0-v31. An uninitialized VRF is X in RTL
  // simulation, and an X that reaches an address or a valid bit stalls the
  // VLSU forever - the element-gather baseline hung even with an L1-resident
  // domain, which no miss-path explanation covers.
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(4u * lanes));
  asm volatile("vmv.v.i v0, 0");
  asm volatile("vmv.v.i v8, 0");
  asm volatile("vmv.v.i v16, 0");
  asm volatile("vmv.v.i v24, 0");

  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(lanes));

  // prologue: chunk 0 of i-cluster 0 -> set A
  asm volatile("vle32.v v4, (%0)" ::"r"(ex) : "memory");
  asm volatile("vsll.vi v4, v4, 2");   // element index -> byte offset
  asm volatile("vluxei32.v v8, (%0), v4" ::"r"(nb_x) : "memory");
  asm volatile("vluxei32.v v9, (%0), v4" ::"r"(nb_y) : "memory");
  asm volatile("vluxei32.v v10, (%0), v4" ::"r"(nb_z) : "memory");
  asm volatile("vluxei32.v v11, (%0), v4" ::"r"(nb_q) : "memory");

  for (unsigned int c = 0; c < nc; ++c) {
    // i-cluster coordinates: integer loads + fmv.w.x
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

    for (unsigned int j = 0; j < list; j += 2 * chunk) {
      // ---- arithmetic of chunk (A) with the next chunk (B) loading ----
      if (j == 0) {
        asm volatile("vle32.v v5, (%0)" ::"r"(ex + 1 * estride) : "memory");
        asm volatile("vsll.vi v5, v5, 2");
        asm volatile("vluxei32.v v12, (%0), v5" ::"r"(nb_x) : "memory");
        // i-atom 0
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
        asm volatile("vluxei32.v v13, (%0), v5" ::"r"(nb_y) : "memory");
        // i-atom 1
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
        asm volatile("vluxei32.v v14, (%0), v5" ::"r"(nb_z) : "memory");
        // i-atom 2
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
        asm volatile("vluxei32.v v15, (%0), v5" ::"r"(nb_q) : "memory");
        // i-atom 3
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
      } else {
        asm volatile("vle32.v v5, (%0)" ::"r"(ex + 1 * estride) : "memory");
        asm volatile("vsll.vi v5, v5, 2");
        asm volatile("vluxei32.v v12, (%0), v5" ::"r"(nb_x) : "memory");
        // i-atom 0
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
        asm volatile("vluxei32.v v13, (%0), v5" ::"r"(nb_y) : "memory");
        // i-atom 1
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
        asm volatile("vluxei32.v v14, (%0), v5" ::"r"(nb_z) : "memory");
        // i-atom 2
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
        asm volatile("vluxei32.v v15, (%0), v5" ::"r"(nb_q) : "memory");
        // i-atom 3
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
      // ---- arithmetic of chunk (B) with the chunk after next (A) loading ----
      // (prefetches into the next i-cluster; nb_list_exp is padded)
      asm volatile("vle32.v v4, (%0)" ::"r"(ex + 2 * estride) : "memory");
      asm volatile("vsll.vi v4, v4, 2");
      asm volatile("vluxei32.v v8, (%0), v4" ::"r"(nb_x) : "memory");
      // i-atom 0
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
      asm volatile("vluxei32.v v9, (%0), v4" ::"r"(nb_y) : "memory");
      // i-atom 1
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
      asm volatile("vluxei32.v v10, (%0), v4" ::"r"(nb_z) : "memory");
      // i-atom 2
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
      asm volatile("vluxei32.v v11, (%0), v4" ::"r"(nb_q) : "memory");
      // i-atom 3
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
      ex += 2 * estride;
    }

    // cross-lane reductions (seed v0 = 0), then 12 read-backs and stores
    asm volatile("vfredusum.vs v20, v20, v0");
    asm volatile("vfredusum.vs v21, v21, v0");
    asm volatile("vfredusum.vs v22, v22, v0");
    asm volatile("vfredusum.vs v23, v23, v0");
    asm volatile("vfredusum.vs v24, v24, v0");
    asm volatile("vfredusum.vs v25, v25, v0");
    asm volatile("vfredusum.vs v26, v26, v0");
    asm volatile("vfredusum.vs v27, v27, v0");
    asm volatile("vfredusum.vs v28, v28, v0");
    asm volatile("vfredusum.vs v29, v29, v0");
    asm volatile("vfredusum.vs v30, v30, v0");
    asm volatile("vfredusum.vs v31, v31, v0");
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
