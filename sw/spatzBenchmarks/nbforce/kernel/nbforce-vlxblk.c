// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, VLXBLK arm: GROMACS-style cluster-pair non-bonded force kernel
// (Pall & Hess 2013; Pall et al. 2020), fp32. The domain is resident as
// four parallel X4 field arrays (x, y, z, q; 4 atoms x fp32 = 16-B
// blocks), bare cluster ids as entry numbers. Per chunk one u16 index
// vector drives 4 field-block gathers (16 j-clusters per chunk at
// VLEN = 512), and the gathered j-data is reused by all 4 i-atoms.
// Simplified FMA-only pair force, 17 FLOP/pair (see mdg_iatom_body).

#include "nbforce-vlxblk.h"
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

// Four field gathers per chunk share one u16 cluster-id vector (bare
// entry numbers, blk_len = 4 fp32).
void nbforce_vlxblk(float *fo, const float *nb_x, const float *nb_y,
                    const float *nb_z, const float *nb_q,
                    const uint16_t *pairlist, const unsigned int nc,
                    const unsigned int list, const float cut2) {
  const unsigned int cl = 4u; // atoms per cluster (4 x fp32 = 16-B block)
  const unsigned int vlmax = mdg_vlmax_e32m4();
  const unsigned int chunk_cl_max = vlmax / cl;

  asm volatile("vsetblklen %0" ::"r"(cl));

  for (unsigned int c = 0; c < nc; ++c) {
    const uint16_t *lp = pairlist + c * list;
    float fx[4] = {0, 0, 0, 0}, fy[4] = {0, 0, 0, 0}, fz[4] = {0, 0, 0, 0};
    // Hoist ALL i-cluster scalar reads ahead of the gather loop: a
    // scalar load-miss outstanding while vector gathers miss deadlocks
    // the port (instant-hang family, wave-5 diagnosis). volatile forces
    // the loads to complete here.
    float cix[4], ciy[4], ciz[4];
    {
      const volatile float *vx = nb_x, *vy = nb_y, *vz = nb_z;
      for (unsigned int a = 0; a < 4; ++a) {
        cix[a] = vx[c * cl + a];
        ciy[a] = vy[c * cl + a];
        ciz[a] = vz[c * cl + a];
      }
    }

    for (unsigned int j = 0; j < list;) {
      const unsigned int n_cl =
          ((list - j) < chunk_cl_max) ? (list - j) : chunk_cl_max;
      const unsigned int gvl = n_cl * cl;

      // index vector: n_cl u16 cluster ids
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(n_cl));
      asm volatile("vle16.v v4, (%0)" ::"r"(lp + j) : "memory");

      // Gathers issued 2+2 with a scalar read-back barrier between the
      // pairs: four concurrent MISSING gathers deadlock the port
      // (wave-4 bisection: first-chunk hang; 2 concurrent gathers are
      // the proven level, and P11's 4x pass only L1-resident).
      float sync;
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(gvl));
      asm volatile("vlxblkei16.v v8, (%0), v4" ::"r"(nb_x) : "memory");
      asm volatile("vlxblkei16.v v12, (%0), v4" ::"r"(nb_y) : "memory");
      asm volatile("vfmv.f.s %0, v12" : "=f"(sync));
      (void)sync;
      asm volatile("vlxblkei16.v v16, (%0), v4" ::"r"(nb_z) : "memory");
      asm volatile("vlxblkei16.v v20, (%0), v4" ::"r"(nb_q) : "memory");

      for (unsigned int a = 0; a < 4; ++a)
        mdg_iatom_body(cix[a], ciy[a], ciz[a], cut2, &fx[a], &fy[a], &fz[a]);
      j += n_cl;
    }

    float *f = fo + c * 12;
    for (unsigned int a = 0; a < 4; ++a) {
      f[3 * a + 0] = fx[a];
      f[3 * a + 1] = fy[a];
      f[3 * a + 2] = fz[a];
    }
  }
}
