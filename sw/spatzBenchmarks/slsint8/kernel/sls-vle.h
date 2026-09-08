// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SLS_VLE_H
#define SLS_VLE_H

#include <stdint.h>

// int8 embedding-bag, vle baseline: per lookup the row_d-byte u8 row is
// loaded by scalar-computed address (vle16, row_d/2 x e16), then the same
// widen x2 / vfcvt / vfmacc.vf dequant chain as the VLXBLK arm; the fp16
// row biases are summed as a scalar and added once per bag.
//   out[nb][row_d]  f32 bag sums
//   t[nrows][row_d] u8 table rows
//   sb[nrows][2]    fp16 (scale, bias) per row
//   idx[nb][lp]     u16 row ids
void sls_vle(float *out, const uint8_t *t, const __fp16 *sb,
             const uint16_t *idx, const unsigned int nb,
             const unsigned int lp, const unsigned int row_d);

#endif
