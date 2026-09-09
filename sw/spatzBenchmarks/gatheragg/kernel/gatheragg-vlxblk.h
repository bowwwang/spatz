// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GATHERAGG_VLXBLK_H
#define GATHERAGG_VLXBLK_H

#include <stdint.h>

// GNN pull-mode neighbour aggregation (paper row "gnnagg"), VLXBLK arm:
//   out[b, :] = sum_l t[idx[b, l], :]   for nb nodes x lp rows of row_d = 64
// fp32 (256-B rows). Gather-across-nodes: the 4 nodes of one e32 m8 group
// are pooled simultaneously (one vlxblkei16 of 4 rows per round l, one
// whole-group vfadd). idx is TRANSPOSED (round-major within each group of
// 4): round l of group g is the contiguous run idx[g*lp*4 + l*4 .. +4),
// and the array carries >= 16 ids of padding. nb must be a multiple of 4,
// lp >= 2.
void agg_vlxblk(float *out, const float *t, const uint16_t *idx,
                const unsigned int nb, const unsigned int row_d,
                const unsigned int lp);

#endif
