// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, plain-RVV baseline, MATERIALISATION form (2026-09-10).
//
// The element-gather form (vluxei32 over the 1.45-MiB domain) HANGS on this
// RTL - erratum #3, the same wall pagerank hit - and the hang survived a
// full rewrite, an L1-resident domain and a warm cache. So this baseline
// does what a programmer without block gather actually has to do: copy the
// chunk j-atoms into a contiguous scratch row with SCALAR loads and stores,
// then read the scratch row back with one unit-stride vector load. Every
// vector access is then 128 B and unit-stride; the gather cost appears as
// scalar work, which is exactly the cost the extension removes. Precedent:
// the vqgemm rvv arm materialises its decoded row the same way.
//
// Chunk = 8 j-clusters = 32 fp32 per field = one e32 m1 register at
// VLEN = 1024. Per chunk and field: 32 scalar loads/stores + one vle32.
// The arithmetic, the 12 lane-wise accumulators and the once-per-i-cluster
// reductions are identical to the VLXBLK arm.

#include "nbforce-rvv.h"

void nbforce_rvv(float *fo, const float *nb_x, const float *nb_y,
                 const float *nb_z, const float *nb_q,
                 const uint32_t *pairlist_exp, const unsigned int nc,
                 const unsigned int list, const float cut2) {
  const unsigned int cl = 4u, lanes = 32u, chunk = 8u;
  const uint32_t *xw = (const uint32_t *)nb_x;
  const uint32_t *yw = (const uint32_t *)nb_y;
  const uint32_t *zw = (const uint32_t *)nb_z;
  const uint32_t *qw = (const uint32_t *)nb_q;
  const uint32_t *ex = pairlist_exp;   // element index = cluster*4 + lane
  float *f = fo;
  float fzero;
  float xi0, yi0, zi0, xi1, yi1, zi1, xi2, yi2, zi2, xi3, yi3, zi3;
  float s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
  static uint32_t scratch[4][32] __attribute__((aligned(64), section(".data")));

  asm volatile("fmv.w.x %0, zero" : "=f"(fzero));
  asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(lanes));
  asm volatile("vmv.v.i v0, 0"); // reduction seed

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

    for (unsigned int j = 0; j < list; j += chunk) {
      const uint32_t *ei = ex + j * cl;
      for (unsigned int e = 0; e < 32u; ++e) {   // materialise the chunk
        const uint32_t k = ei[e];
        scratch[0][e] = xw[k];
        scratch[1][e] = yw[k];
        scratch[2][e] = zw[k];
        scratch[3][e] = qw[k];
      }
      asm volatile("vle32.v v8, (%0)" ::"r"(scratch[0]) : "memory");
      asm volatile("vle32.v v9, (%0)" ::"r"(scratch[1]) : "memory");
      asm volatile("vle32.v v10, (%0)" ::"r"(scratch[2]) : "memory");
      asm volatile("vle32.v v11, (%0)" ::"r"(scratch[3]) : "memory");
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
      if (j == 0) {
        asm volatile("vfmul.vv v20, v19, v16");
        asm volatile("vfmul.vv v21, v19, v17");
        asm volatile("vfmul.vv v22, v19, v18");
      } else {
        asm volatile("vfmacc.vv v20, v19, v16");
        asm volatile("vfmacc.vv v21, v19, v17");
        asm volatile("vfmacc.vv v22, v19, v18");
      }
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
      if (j == 0) {
        asm volatile("vfmul.vv v23, v19, v16");
        asm volatile("vfmul.vv v24, v19, v17");
        asm volatile("vfmul.vv v25, v19, v18");
      } else {
        asm volatile("vfmacc.vv v23, v19, v16");
        asm volatile("vfmacc.vv v24, v19, v17");
        asm volatile("vfmacc.vv v25, v19, v18");
      }
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
      if (j == 0) {
        asm volatile("vfmul.vv v26, v19, v16");
        asm volatile("vfmul.vv v27, v19, v17");
        asm volatile("vfmul.vv v28, v19, v18");
      } else {
        asm volatile("vfmacc.vv v26, v19, v16");
        asm volatile("vfmacc.vv v27, v19, v17");
        asm volatile("vfmacc.vv v28, v19, v18");
      }
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
      if (j == 0) {
        asm volatile("vfmul.vv v29, v19, v16");
        asm volatile("vfmul.vv v30, v19, v17");
        asm volatile("vfmul.vv v31, v19, v18");
      } else {
        asm volatile("vfmacc.vv v29, v19, v16");
        asm volatile("vfmacc.vv v30, v19, v17");
        asm volatile("vfmacc.vv v31, v19, v18");
      }
    }
    ex += list * cl;

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
