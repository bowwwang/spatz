// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef PR_VLXBLK_H
#define PR_VLXBLK_H

#include <stdint.h>

// PageRank pull iteration, VLXBLK arm (fp64, D=1, u16 neighbor ids):
//   out[v] = seed[0] + sum_{e < 16} damp * contrib[nbr[v * 16 + e]]
// seed is a 16-lane fp64 vector (lane 0 = base = 0.15/NV, lanes 1..15 = 0)
// used to seed the per-vertex ordered reductions. Fixed deg = 16 (one
// neighborhood = one e64 register at VLEN=1024); nact must be a multiple
// of 16 (two 8-vertex pipeline rounds per loop iteration).
void pr_vlxblk(double *out, const double *contrib, const uint16_t *nbr,
               const double *seed, const unsigned int nact,
               const double damp);

#endif
