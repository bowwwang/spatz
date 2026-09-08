// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SLS_VLXBLK_H
#define SLS_VLXBLK_H

#include <stdint.h>

// int8 embedding-bag, VLXBLK arm: per lookup one vlxblkei16 gathers the
// row_d-byte u8 row (as row_d/2 x e16, blk_len row_d/2), widen x2, vfcvt,
// vfmacc.vf with the fp16 row scale; the fp16 row biases are summed as a
// scalar and added once per bag.
//   out[nb][row_d]  f32 bag sums
//   t[nrows][row_d] u8 table rows
//   sb[nrows][2]    fp16 (scale, bias) per row
//   idx[nb][lp]     u16 row ids
void sls_vlxblk(float *out, const uint8_t *t, const __fp16 *sb,
                const uint16_t *idx, const unsigned int nb,
                const unsigned int lp, const unsigned int row_d);

#endif
