// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// pr-gather, VLXBLK arm: PageRank pull iteration (GAP/Graph500-class
// graph analytics), out[v] = base + damp * sum_{u in N(v)} contrib[u],
// contributions gathered by neighbor id (fp64, D=1: 8-B "blocks" = one
// port word). This is the fine-granularity regime point: the win is
// INDEX SEMANTICS - vlxblk consumes the u16 vertex ids directly
// (vsetblklen 1, hardware scales by 8), where an element-indexed load
// would first need the ids widened and scaled to byte offsets.
// The per-vertex sum is an ordered reduction (vfredosum) seeded with 0.

#include "pr-vlxblk.h"

void pr_vlxblk(double *out, const double *contrib, const uint16_t *nbr,
               const unsigned int nact, const unsigned int deg,
               const double base, const double damp) {
  const unsigned int blk_len = 1; // one 8-B element per block

  asm volatile("vsetblklen %0" ::"r"(blk_len));

  for (unsigned int v = 0; v < nact; ++v) {
    double s;

    // deg neighbor ids (u16)
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(deg));
    asm volatile("vle16.v v2, (%0)" ::"r"(nbr + v * deg) : "memory");

    // gather the deg contributions (e64) and reduce them in order
    asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(deg));
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(contrib) : "memory");
    asm volatile("vmv.s.x v16, zero");
    asm volatile("vfredosum.vs v16, v8, v16");
    asm volatile("vfmv.f.s %0, v16" : "=f"(s));

    out[v] = base + damp * s;
  }
}
