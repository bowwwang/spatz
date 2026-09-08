// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef PR_VLXBLK_H
#define PR_VLXBLK_H

#include <stdint.h>

// PageRank pull iteration, VLXBLK arm (fp64, D=1, u16 neighbor ids):
//   out[v] = base + damp * sum_{e < deg} contrib[nbr[v * deg + e]]
// Per vertex: one vle16 of deg ids, one vlxblkei16 gather of deg 8-B
// "blocks" (vsetblklen 1), one seeded ordered reduction (vfredosum).
// deg <= 32: both vsetvli requests (e16 m1 index load, e64 m4 gather)
// cap at 32 elements on VLEN=512; a larger deg silently clamps vl and
// sums only the first 32 neighbors (gen_data.py asserts DEG <= 32).
void pr_vlxblk(double *out, const double *contrib, const uint16_t *nbr,
               const unsigned int nact, const unsigned int deg,
               const double base, const double damp);

#endif
