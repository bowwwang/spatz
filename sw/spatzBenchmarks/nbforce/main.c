// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// nbforce: GROMACS-style cluster-pair non-bonded force kernel (Pall &
// Hess 2013; Pall et al. 2020), fp32, ADH-sized domain: NC_DOM = 23,750
// j-clusters (95k atoms, the paper's Fig. 8 benchmark system) resident
// as four parallel X4 field arrays (x, y, z, q; 4 atoms x fp32 = 16-B
// blocks, bare cluster ids as entry numbers — at this domain size the
// packed-record field-slice trick would overflow u16 entry numbers:
// EI32 headroom evidence). Timed region: a 256-i-cluster tile in
// spatial order, LIST = 96 j-clusters per i-cluster (paper-derived),
// pair list drawn from a +-256-id locality window (spatial-sort model).
// Simplified FMA-only pair force, 17 FLOP/pair (see mdg_iatom_body).
//
// Arms (VARIANT): 1 = VLXBLK: one u16 index vector per chunk drives 4
//                     field-block gathers (16 j-clusters per chunk),
//                     j-data reused by all 4 i-atoms.
//                 0 = RVV baseline: canonical element-granular
//                     vluxei32 with a PRECOMPUTED expanded per-element
//                     u32 index array (no vid.v/vrgather on Spatz),
//                     loaded inside the timed region — its 4x index
//                     traffic is charged to the baseline fairly.
// Check: SAMPLED (every 32nd i-cluster + the last), 1% tolerance
// (fp32 reassociation across chunked accumulation).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

#define VLXBLKEI16_V(vd, rs1n, vs2)  "vlxblkei16.v v" #vd ", (x" #rs1n "), v" #vs2 "\n"
#define VSETBLKLEN(rs1n)             "vsetblklen x" #rs1n "\n"

#ifndef NC_DOM
#define NC_DOM 23750 // domain j-clusters (ADH: 95,000 atoms / 4)
#endif
#ifndef NC_TILE
#define NC_TILE 256 // timed i-clusters (T in the paper table)
#endif
#ifndef LIST
#define LIST 96 // j-clusters per i-cluster (P; Pall'13-derived)
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define MDG_CL 4      // atoms per cluster (4 x fp32 = 16-B field block)
#define WINDOW 256    // pair-list locality window (+-ids)
#define CUT2 2.0f

static float nb_x[NC_DOM * MDG_CL] __attribute__((section(".data"), aligned(128)));
static float nb_y[NC_DOM * MDG_CL] __attribute__((section(".data"), aligned(128)));
static float nb_z[NC_DOM * MDG_CL] __attribute__((section(".data"), aligned(128)));
static float nb_q[NC_DOM * MDG_CL] __attribute__((section(".data"), aligned(128)));
static uint16_t nb_list[NC_TILE * LIST] __attribute__((section(".data"), aligned(64)));
#if VARIANT == 0
static uint32_t nb_list_exp[NC_TILE * LIST * MDG_CL] __attribute__((section(".data"), aligned(128)));
#endif
static float nb_f[NC_TILE * 12] __attribute__((section(".data"), aligned(64)));

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

  asm volatile(
      "vfrsub.vf v24, v8,  %[xi]\n"
      "vfrsub.vf v28, v12, %[yi]\n"
      "vfrsub.vf v0,  v16, %[zi]\n"
      "vfmul.vv  v4,  v24, v24\n"
      "vfmacc.vv v4,  v28, v28\n"
      "vfmacc.vv v4,  v0,  v0\n"
      "vfrsub.vf v4,  v4,  %[c2]\n"
      "vfmax.vf  v4,  v4,  %[z]\n"
      "vfmul.vv  v4,  v4,  v4\n"
      "vfmul.vv  v4,  v4,  v20\n"
      "vfmul.vv  v24, v24, v4\n"
      "vfmul.vv  v28, v28, v4\n"
      "vfmul.vv  v0,  v0,  v4\n"
      :
      : [xi] "f"(xi), [yi] "f"(yi), [zi] "f"(zi), [c2] "f"(cut2),
        [z] "f"(fzero)
      : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");

  asm volatile("vfmv.s.f v4, %[ax]\n"
               "vfredusum.vs v4, v24, v4\n" ::[ax] "f"(ax) : "v4");
  asm volatile("vfmv.f.s %0, v4" : "=f"(ax));
  asm volatile("vfmv.s.f v4, %[ay]\n"
               "vfredusum.vs v4, v28, v4\n" ::[ay] "f"(ay) : "v4");
  asm volatile("vfmv.f.s %0, v4" : "=f"(ay));
  asm volatile("vfmv.s.f v4, %[az]\n"
               "vfredusum.vs v4, v0, v4\n" ::[az] "f"(az) : "v4");
  asm volatile("vfmv.f.s %0, v4" : "=f"(az));

  *fx = ax;
  *fy = ay;
  *fz = az;
}

#if VARIANT == 1
// Four field gathers per chunk share one u16 cluster-id vector (bare
// entry numbers, blk_len = 4 fp32).
static void nbforce_vlxblk(float *forces, const uint16_t *pairlist,
                           unsigned int nc, unsigned int list, float cut2) {
  const unsigned int vlmax = mdg_vlmax_e32m4();
  const unsigned int chunk_cl_max = vlmax / MDG_CL;
  register uint32_t bl asm("t0") = MDG_CL; // x5
  asm volatile(VSETBLKLEN(5) :: "r"(bl));

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
        cix[a] = vx[c * MDG_CL + a];
        ciy[a] = vy[c * MDG_CL + a];
        ciz[a] = vz[c * MDG_CL + a];
      }
    }

    for (unsigned int j = 0; j < list;) {
      const unsigned int n_cl =
          ((list - j) < chunk_cl_max) ? (list - j) : chunk_cl_max;
      const unsigned int gvl = n_cl * MDG_CL;
      register const float *bx asm("t1") = nb_x; // x6
      register const float *by asm("t2") = nb_y; // x7
      register const float *bz asm("t3") = nb_z; // x28
      register const float *bq asm("t4") = nb_q; // x29
      // Gathers issued 2+2 with a scalar read-back barrier between the
      // pairs: four concurrent MISSING gathers deadlock the port
      // (wave-4 bisection: first-chunk hang; 2 concurrent gathers are
      // the proven level, and P11's 4x pass only L1-resident).
      float sync;
      asm volatile("vsetvli zero, %[n_cl], e16, m1, ta, ma\n"
                   "vle16.v v4, (%[idxp])\n"
                   "vsetvli zero, %[gvl], e32, m4, ta, ma\n"
                   VLXBLKEI16_V(8, 6, 4)
                   VLXBLKEI16_V(12, 7, 4)
                   "vfmv.f.s %[sy], v12\n"
                   : [sy] "=f"(sync)
                   : [n_cl] "r"(n_cl), [gvl] "r"(gvl), [idxp] "r"(lp + j),
                     "r"(bx), "r"(by)
                   : "v4", "v5", "v8", "v9", "v10", "v11", "v12", "v13",
                     "v14", "v15", "memory");
      (void)sync;
      asm volatile(VLXBLKEI16_V(16, 28, 4)
                   VLXBLKEI16_V(20, 29, 4)
                   :
                   : "r"(bz), "r"(bq)
                   : "v16", "v17", "v18", "v19", "v20", "v21",
                     "v22", "v23", "memory");

      for (unsigned int a = 0; a < 4; ++a)
        mdg_iatom_body(cix[a], ciy[a], ciz[a], cut2, &fx[a], &fy[a], &fz[a]);
      j += n_cl;
    }
    float *fo = forces + c * 12;
    for (unsigned int a = 0; a < 4; ++a) {
      fo[3 * a + 0] = fx[a];
      fo[3 * a + 1] = fy[a];
      fo[3 * a + 2] = fz[a];
    }
  }
}
#else
// Baseline: element-granular vluxei32 from a precomputed expanded u32
// element-index array (value = cluster*4 + lane; <<2 = byte offset).
static void nbforce_rvv(float *forces, const uint32_t *pairlist_exp,
                        unsigned int nc, unsigned int list, float cut2) {
  const unsigned int vlmax = mdg_vlmax_e32m4();
  const unsigned int list_el = list * MDG_CL;

  for (unsigned int c = 0; c < nc; ++c) {
    const uint32_t *lp = pairlist_exp + c * list_el;
    float fx[4] = {0, 0, 0, 0}, fy[4] = {0, 0, 0, 0}, fz[4] = {0, 0, 0, 0};
    float cix[4], ciy[4], ciz[4];
    {
      const volatile float *vx = nb_x, *vy = nb_y, *vz = nb_z;
      for (unsigned int a = 0; a < 4; ++a) {
        cix[a] = vx[c * MDG_CL + a];
        ciy[a] = vy[c * MDG_CL + a];
        ciz[a] = vz[c * MDG_CL + a];
      }
    }

    for (unsigned int e = 0; e < list_el;) {
      const unsigned int gvl = ((list_el - e) < vlmax) ? (list_el - e) : vlmax;
      asm volatile("vsetvli zero, %[gvl], e32, m4, ta, ma\n"
                   "vle32.v v4, (%[idxp])\n"
                   "vsll.vi v4, v4, 2\n" // byte offs = cluster*16 + lane*4
                   "vluxei32.v v8,  (%[cx]), v4\n"
                   "vluxei32.v v12, (%[cy]), v4\n"
                   "vluxei32.v v16, (%[cz]), v4\n"
                   "vluxei32.v v20, (%[cq]), v4\n"
                   :
                   : [gvl] "r"(gvl), [idxp] "r"(lp + e), [cx] "r"(nb_x),
                     [cy] "r"(nb_y), [cz] "r"(nb_z), [cq] "r"(nb_q)
                   : "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
                     "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
                     "v21", "v22", "v23", "memory");

      for (unsigned int a = 0; a < 4; ++a)
        mdg_iatom_body(cix[a], ciy[a], ciz[a], cut2, &fx[a], &fy[a], &fz[a]);
      e += gvl;
    }
    float *fo = forces + c * 12;
    for (unsigned int a = 0; a < 4; ++a) {
      fo[3 * a + 0] = fx[a];
      fo[3 * a + 1] = fy[a];
      fo[3 * a + 2] = fz[a];
    }
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
    // Coordinates/charges in [0,4): prime-period pattern head + vector
    // replication per field array (bench_fill.h).
    {
      const unsigned int n = NC_DOM * MDG_CL;
      for (unsigned int i = 0; i < 976u; ++i) { // f32: BF_HEAD_BYTES/4
        nb_x[i] = 0.00390625f * (float)((i * 37u) & 1023u);
        nb_y[i] = 0.00390625f * (float)((i * 53u) & 1023u);
        nb_z[i] = 0.00390625f * (float)((i * 71u) & 1023u);
        nb_q[i] = 0.00390625f * (float)((i * 89u) & 1023u);
      }
      bench_fill_rep(nb_x, n * 4u, 976u * 4u);
      bench_fill_rep(nb_y, n * 4u, 976u * 4u);
      bench_fill_rep(nb_z, n * 4u, 976u * 4u);
      bench_fill_rep(nb_q, n * 4u, 976u * 4u);
    }
    // Pair list: LIST j-ids per i-cluster from a +-WINDOW id window
    // (spatial-sort model), wrapped at the domain edge, j != i.
    for (unsigned int c = 0; c < NC_TILE; ++c)
      for (unsigned int l = 0; l < LIST; ++l) {
        const uint32_t h = ((c * LIST + l) * 2654435761u) >> 23; // 9 bits
        int off = (int)(h & (2u * WINDOW - 1u)) - (int)WINDOW;
        if (off >= 0)
          off += 1; // off in [-256,-1] u [1,256]
        int j = (int)c + off;
        if (j < 0)
          j += NC_DOM;
        nb_list[c * LIST + l] = (uint16_t)j;
      }
#if VARIANT == 0
    for (unsigned int e = 0; e < NC_TILE * LIST; ++e) {
      const uint32_t base = (uint32_t)nb_list[e] * MDG_CL;
      nb_list_exp[4 * e + 0] = base + 0;
      nb_list_exp[4 * e + 1] = base + 1;
      nb_list_exp[4 * e + 2] = base + 2;
      nb_list_exp[4 * e + 3] = base + 3;
    }
#endif
    bench_fill_zero(nb_f, sizeof(nb_f));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    nbforce_vlxblk(nb_f, nb_list, NC_TILE, LIST, CUT2);
#else
    nbforce_rvv(nb_f, nb_list_exp, NC_TILE, LIST, CUT2);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // SAMPLED check (every 32nd i-cluster + the last): 1% + abs
    // tolerance for fp32 reassociation across chunk boundaries.
    for (unsigned int s32 = 0; s32 <= NC_TILE / 32 && fails == 0; ++s32) {
      const unsigned int c = (s32 == NC_TILE / 32) ? (NC_TILE - 1) : (s32 * 32);
      for (unsigned int a = 0; a < 4 && fails == 0; ++a) {
        float ex = 0.0f, ey = 0.0f, ez = 0.0f;
        const float xi = nb_x[c * MDG_CL + a], yi = nb_y[c * MDG_CL + a],
                    zi = nb_z[c * MDG_CL + a];
        for (unsigned int l = 0; l < LIST; ++l) {
          const uint32_t j = nb_list[c * LIST + l];
          for (unsigned int b = 0; b < 4; ++b) {
            const float dx = xi - nb_x[j * MDG_CL + b];
            const float dy = yi - nb_y[j * MDG_CL + b];
            const float dz = zi - nb_z[j * MDG_CL + b];
            float w = CUT2 - (dx * dx + dy * dy + dz * dz);
            if (w < 0.0f)
              w = 0.0f;
            const float s = nb_q[j * MDG_CL + b] * w * w;
            ex += s * dx;
            ey += s * dy;
            ez += s * dz;
          }
        }
        const float gx = nb_f[c * 12 + 3 * a + 0];
        const float gy = nb_f[c * 12 + 3 * a + 1];
        const float gz = nb_f[c * 12 + 3 * a + 2];
        const float e3[3] = {ex, ey, ez}, g3[3] = {gx, gy, gz};
        for (unsigned int d = 0; d < 3; ++d) {
          float err = g3[d] - e3[d];
          if (err < 0)
            err = -err;
          float mag = e3[d] < 0 ? -e3[d] : e3[d];
          if (err > 0.01f * mag + 0.01f) {
            printf("FAILED c=%d a=%d d=%d got=%f exp=%f\n", c, a, d, g3[d],
                   e3[d]);
            fails = 1;
          }
        }
      }
    }

    printf("nbforce variant=%d ncdom=%d tile=%d list=%d cache=%d: took %u "
           "cycles %s (flop=%u)\n", VARIANT, NC_DOM, NC_TILE, LIST, USE_CACHE,
           cycles, fails ? "CHECK-FAILED" : "CHECK-OK",
           (unsigned)(NC_TILE * 4 * LIST * 4 * 17));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
