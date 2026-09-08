// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef GATHERAGG_VLE_H
#define GATHERAGG_VLE_H

#include <stdint.h>

// Indexed-row gather + sum pooling, vle baseline: one destination per
// iteration, each pooled row loaded by scalar-computed address (vle32)
// and accumulated with vfadd. idx is TRANSPOSED (round-major within each
// group of 128 / row_d destinations, the vlxblk arm's layout).
// Selects the per-LMUL kernel on row_d: 32 (one register, m1) or 64 (m2).
void agg_vle(float *out, const float *t, const uint16_t *idx,
             const unsigned int nb, const unsigned int row_d,
             const unsigned int lp);

// 32-f32 rows (128 B = exactly one register, m1): sls-fp32.
void agg_vle_d32(float *out, const float *t, const uint16_t *idx,
                 const unsigned int nb, const unsigned int lp);

// 64-f32 rows (256 B = two registers, m2): gnnagg.
void agg_vle_d64(float *out, const float *t, const uint16_t *idx,
                 const unsigned int nb, const unsigned int lp);

#endif
