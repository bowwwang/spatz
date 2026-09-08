// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, plain-RVV baseline: canonical element-granular vluxei32 with
// a PRECOMPUTED expanded per-element u32 index array (no vid.v/vrgather
// on Spatz), loaded inside the timed region — its 4x index traffic is
// charged to the baseline fairly. Same pair-force math and same i-cluster
// scalar hoisting as the VLXBLK arm.

#include "nbforce-rvv.h"
#include <stddef.h>

static inline unsigned int mdg_vlmax_e32m4(void) {
  unsigned int vlmax;
  asm volatile("vsetvli %0, zero, e32, m4, ta, ma" : "=r"(vlmax));
  return vlmax;
}

// One i-atom against the gathered j-data of the current chunk (vtype
// e32/m4, vl = gvl). Reads v8/v12/v16/v20 (xj/yj/zj/qj), clobbers
// v0-7/v24-31, accumulates reduced (fx,fy,fz) into scalar accumulators.
// Ported verbatim from the vq-DATE sp-mdgather kernel (golden there);
// the serialized seeded-vfredosum read-back sequence is the proven
// pattern — do not batch reductions (RTL FSM deadlock).
static inline void mdg_iatom_body(const float xi, const float yi,
                                  const float zi, const float cut2,
                                  float *fx, float *fy, float *fz) {
  float ax = *fx, ay = *fy, az = *fz;
  const float fzero = 0.0f;

  asm volatile("vfrsub.vf v24, v8, %0" ::"f"(xi));   // dx = xi - xj
  asm volatile("vfrsub.vf v28, v12, %0" ::"f"(yi));  // dy = yi - yj
  asm volatile("vfrsub.vf v0, v16, %0" ::"f"(zi));   // dz = zi - zj
  asm volatile("vfmul.vv v4, v24, v24");
  asm volatile("vfmacc.vv v4, v28, v28");
  asm volatile("vfmacc.vv v4, v0, v0");             // r2
  asm volatile("vfrsub.vf v4, v4, %0" ::"f"(cut2));  // w = cut2 - r2
  asm volatile("vfmax.vf v4, v4, %0" ::"f"(fzero));  // w = max(w, 0)
  asm volatile("vfmul.vv v4, v4, v4");              // w^2
  asm volatile("vfmul.vv v4, v4, v20");             // s = w^2 * qj
  asm volatile("vfmul.vv v24, v24, v4");            // s * dx
  asm volatile("vfmul.vv v28, v28, v4");            // s * dy
  asm volatile("vfmul.vv v0, v0, v4");              // s * dz

  asm volatile("vfmv.s.f v4, %0" ::"f"(ax));
  asm volatile("vfredusum.vs v4, v24, v4");
  asm volatile("vfmv.f.s %0, v4" : "=f"(ax));
  asm volatile("vfmv.s.f v4, %0" ::"f"(ay));
  asm volatile("vfredusum.vs v4, v28, v4");
  asm volatile("vfmv.f.s %0, v4" : "=f"(ay));
  asm volatile("vfmv.s.f v4, %0" ::"f"(az));
  asm volatile("vfredusum.vs v4, v0, v4");
  asm volatile("vfmv.f.s %0, v4" : "=f"(az));

  *fx = ax;
  *fy = ay;
  *fz = az;
}

// Baseline: element-granular vluxei32 from a precomputed expanded u32
// element-index array (value = cluster*4 + lane; <<2 = byte offset).
void nbforce_rvv(float *fo, const float *nb_x, const float *nb_y,
                 const float *nb_z, const float *nb_q,
                 const uint32_t *pairlist_exp, const unsigned int nc,
                 const unsigned int list, const float cut2) {
  const unsigned int cl = 4u; // atoms per cluster (4 x fp32 = 16-B block)
  const unsigned int vlmax = mdg_vlmax_e32m4();
  const unsigned int list_el = list * cl;

  for (unsigned int c = 0; c < nc; ++c) {
    const uint32_t *lp = pairlist_exp + c * list_el;
    float fx[4] = {0, 0, 0, 0}, fy[4] = {0, 0, 0, 0}, fz[4] = {0, 0, 0, 0};
    // Hoist ALL i-cluster scalar reads ahead of the gather loop (see
    // the VLXBLK arm): volatile forces the loads to complete here.
    float cix[4], ciy[4], ciz[4];
    {
      const volatile float *vx = nb_x, *vy = nb_y, *vz = nb_z;
      for (unsigned int a = 0; a < 4; ++a) {
        cix[a] = vx[c * cl + a];
        ciy[a] = vy[c * cl + a];
        ciz[a] = vz[c * cl + a];
      }
    }

    for (unsigned int e = 0; e < list_el;) {
      const unsigned int gvl = ((list_el - e) < vlmax) ? (list_el - e) : vlmax;

      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(gvl));
      asm volatile("vle32.v v4, (%0)" ::"r"(lp + e) : "memory");
      asm volatile("vsll.vi v4, v4, 2"); // byte offs = cluster*16 + lane*4
      asm volatile("vluxei32.v v8, (%0), v4" ::"r"(nb_x) : "memory");
      asm volatile("vluxei32.v v12, (%0), v4" ::"r"(nb_y) : "memory");
      asm volatile("vluxei32.v v16, (%0), v4" ::"r"(nb_z) : "memory");
      asm volatile("vluxei32.v v20, (%0), v4" ::"r"(nb_q) : "memory");

      for (unsigned int a = 0; a < 4; ++a)
        mdg_iatom_body(cix[a], ciy[a], ciz[a], cut2, &fx[a], &fy[a], &fz[a]);
      e += gvl;
    }

    float *f = fo + c * 12;
    for (unsigned int a = 0; a < 4; ++a) {
      f[3 * a + 0] = fx[a];
      f[3 * a + 1] = fy[a];
      f[3 * a + 2] = fz[a];
    }
  }
}
