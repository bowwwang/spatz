// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// pr-gather, baseline arm: the SCALAR pull loop (the GAP reference
// itself is scalar C). The element-vector translation - widen + scale
// the u16 ids to 4-B byte offsets (vwmulu by 8, the most favorable
// single-op translation) and gather with vluxei32 - HANGS on the first
// missing gather on this platform: erratum #3, the upstream-vluxei-
// under-miss bug (wave-4 bisection, 2026-09-08). Misses are intrinsic
// here (the 512 KiB contrib array exceeds L1), so no vluxei baseline can
// run; the benchmark target keeps its historical `vluxei` name only.
// Volatile reads keep -O3 from re-vectorizing the gather.

#include "pr-scalar.h"

void pr_scalar(double *out, const double *contrib, const uint16_t *nbr,
               const unsigned int nact, const unsigned int deg,
               const double base, const double damp) {
  for (unsigned int v = 0; v < nact; ++v) {
    double s = 0.0;
    const volatile double *cv = contrib;
    for (unsigned int e = 0; e < deg; ++e)
      s += cv[nbr[v * deg + e]];
    out[v] = base + damp * s;
  }
}
