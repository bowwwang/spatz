// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef PR_SCALAR_H
#define PR_SCALAR_H

#include <stdint.h>

// PageRank pull iteration, SCALAR baseline (fp64, u16 neighbor ids):
//   out[v] = base + damp * sum_{e < deg} contrib[nbr[v * deg + e]]
// The GAP reference itself is scalar C; the element-vector translation
// (vwmulu + vluxei32) cannot run on this platform (erratum #3, see
// pr-scalar.c). Reads are volatile so -O3 does not re-vectorize.
void pr_scalar(double *out, const double *contrib, const uint16_t *nbr,
               const unsigned int nact, const unsigned int deg,
               const double base, const double damp);

#endif
