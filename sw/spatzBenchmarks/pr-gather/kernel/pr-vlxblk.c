// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// pr-gather, VLXBLK arm: PageRank pull iteration (GAP/Graph500-class
// graph analytics), out[v] = base + damp * sum_{u in N(v)} contrib[u].
//
// Fine-granularity regime point (D=1, 8-B "blocks" = one port word): the
// win is INDEX SEMANTICS - vlxblk consumes the u16 vertex ids directly
// (vsetblklen 1, hardware scales by 8), where an element-indexed load
// would first need the ids widened and scaled to byte offsets.
//
// Step = 8 vertices: their 128 ids (256 B, one e16 m2 load) feed ONE e64
// m8 gather (128 blocks, 1 KiB); the group is scaled by damp in place
// (vfmul.vf), and each vertex's neighborhood - one e64 register of the
// group - is reduced with an ordered reduction seeded by v0 (lane 0 =
// base, others 0), so the result is complete: out = base + sum(damp*c).
// The 8 sums are read back into scalar registers and stored by the core.
//
// Two-round software pipeline: group A (ids v2-3, rows v8-15) and group B
// (ids v4-5, rows v16-23) alternate; the next group's id load and gather
// are issued before the current group's arithmetic so the gather (mostly
// L1 misses: the 512-KiB table exceeds L1) overlaps the VFU work and the
// scalar read-backs. Reductions land in v24-31.
//
// Requirements: deg == 16, nact a multiple of 16 (A/B pairs), seed[0] =
// base and seed[1..15] = 0.

#include "pr-vlxblk.h"

void pr_vlxblk(double *out, const double *contrib, const uint16_t *nbr,
               const double *seed, const unsigned int nact,
               const double damp) {
  const unsigned int idx_el = 128; // ids per step: 8 vertices x deg 16
  const unsigned int red_el = 16;  // one neighborhood = one e64 register
  const unsigned int blk_len = 1;  // one 8-B element per block
  const uint16_t *ni = nbr;
  double *o = out;
  double s0, s1, s2, s3, s4, s5, s6, s7;

  asm volatile("vsetblklen %0" ::"r"(blk_len));

  // reduction seed (one full-width 128-B load)
  asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(red_el));
  asm volatile("vle64.v v0, (%0)" ::"r"(seed) : "memory");

  // prologue: ids + gather of group A (vertices 0..7)
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(idx_el));
  asm volatile("vle16.v v2, (%0)" ::"r"(ni) : "memory");
  asm volatile("vsetvli zero, %0, e64, m8, ta, ma" ::"r"(idx_el));
  asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(contrib) : "memory");

  for (unsigned int v = 0; v < nact; v += 16) {
    // ---- load group B (vertices v+8..v+15) ----
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(idx_el));
    asm volatile("vle16.v v4, (%0)" ::"r"(ni + idx_el) : "memory");
    asm volatile("vsetvli zero, %0, e64, m8, ta, ma" ::"r"(idx_el));
    asm volatile("vlxblkei16.v v16, (%0), v4" ::"r"(contrib) : "memory");

    // ---- arithmetic group A (vertices v..v+7) ----
    asm volatile("vfmul.vf v8, v8, %0" ::"f"(damp));
    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(red_el));
    asm volatile("vfredosum.vs v24, v8, v0");
    asm volatile("vfredosum.vs v25, v9, v0");
    asm volatile("vfredosum.vs v26, v10, v0");
    asm volatile("vfredosum.vs v27, v11, v0");
    asm volatile("vfredosum.vs v28, v12, v0");
    asm volatile("vfredosum.vs v29, v13, v0");
    asm volatile("vfredosum.vs v30, v14, v0");
    asm volatile("vfredosum.vs v31, v15, v0");
    asm volatile("vfmv.f.s %0, v24" : "=f"(s0));
    asm volatile("vfmv.f.s %0, v25" : "=f"(s1));
    asm volatile("vfmv.f.s %0, v26" : "=f"(s2));
    asm volatile("vfmv.f.s %0, v27" : "=f"(s3));
    asm volatile("vfmv.f.s %0, v28" : "=f"(s4));
    asm volatile("vfmv.f.s %0, v29" : "=f"(s5));
    asm volatile("vfmv.f.s %0, v30" : "=f"(s6));
    asm volatile("vfmv.f.s %0, v31" : "=f"(s7));
    o[0] = s0;
    o[1] = s1;
    o[2] = s2;
    o[3] = s3;
    o[4] = s4;
    o[5] = s5;
    o[6] = s6;
    o[7] = s7;
    ni += idx_el;
    o += 8;

    // ---- load group A (vertices v+16..v+23), if any ----
    if (v + 16 < nact) {
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(idx_el));
      asm volatile("vle16.v v2, (%0)" ::"r"(ni + idx_el) : "memory");
      asm volatile("vsetvli zero, %0, e64, m8, ta, ma" ::"r"(idx_el));
      asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(contrib) : "memory");
    }

    // ---- arithmetic group B (vertices v+8..v+15) ----
    asm volatile("vsetvli zero, %0, e64, m8, ta, ma" ::"r"(idx_el));
    asm volatile("vfmul.vf v16, v16, %0" ::"f"(damp));
    asm volatile("vsetvli zero, %0, e64, m1, ta, ma" ::"r"(red_el));
    asm volatile("vfredosum.vs v24, v16, v0");
    asm volatile("vfredosum.vs v25, v17, v0");
    asm volatile("vfredosum.vs v26, v18, v0");
    asm volatile("vfredosum.vs v27, v19, v0");
    asm volatile("vfredosum.vs v28, v20, v0");
    asm volatile("vfredosum.vs v29, v21, v0");
    asm volatile("vfredosum.vs v30, v22, v0");
    asm volatile("vfredosum.vs v31, v23, v0");
    asm volatile("vfmv.f.s %0, v24" : "=f"(s0));
    asm volatile("vfmv.f.s %0, v25" : "=f"(s1));
    asm volatile("vfmv.f.s %0, v26" : "=f"(s2));
    asm volatile("vfmv.f.s %0, v27" : "=f"(s3));
    asm volatile("vfmv.f.s %0, v28" : "=f"(s4));
    asm volatile("vfmv.f.s %0, v29" : "=f"(s5));
    asm volatile("vfmv.f.s %0, v30" : "=f"(s6));
    asm volatile("vfmv.f.s %0, v31" : "=f"(s7));
    o[0] = s0;
    o[1] = s1;
    o[2] = s2;
    o[3] = s3;
    o[4] = s4;
    o[5] = s5;
    o[6] = s6;
    o[7] = s7;
    ni += idx_el;
    o += 8;
  }
}
