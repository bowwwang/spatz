// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GATHERAGG_VLXBLK_H
#define GATHERAGG_VLXBLK_H

#include <stdint.h>

// Indexed-row gather + sum pooling, VLXBLK arm:
//   out[b, :] = sum_l t[idx[b, l], :]   for nb destinations x lp rows of
// row_d f32 each. Gather-across-units: the 128 / row_d destinations of one
// m4 group are pooled simultaneously (one vlxblkei16 per round l, one
// whole-group vfadd). idx is TRANSPOSED (round-major within each group):
// round l of group g is the contiguous run idx[g*lp*upg + l*upg .. +upg).
// nb must be a multiple of 128 / row_d.
// dbg_every != 0 prints "DBGG <b>" every dbg_every destinations (bug-A
// hang locator for the waveform session; 0 in measurement configs).
void agg_vlxblk(float *out, const float *t, const uint16_t *idx,
                const unsigned int nb, const unsigned int row_d,
                const unsigned int lp, const unsigned int dbg_every);

#endif
