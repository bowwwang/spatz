// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SLS32_VLXBLK_H
#define SLS32_VLXBLK_H

#include <stdint.h>

// SparseLengthsSum pooling over fp32 rows of row_d = 32 elements (128-B
// blocks, paper point sls-2), VLXBLK arm: 8 rows per m8 gather, one m8 fp32
// add per gather into 8 chunk-slot partial sums (out[b][8][32], 1 KiB per
// bag, no final reduction), consecutive gathers alternate two register
// groups. Requirements (not asserted): row_d = 32; lp a multiple of 8; nb
// even; idx carries >= 16 elements of tail padding (32-B index loads).
void sls32_vlxblk(float *out, const float *t, const uint16_t *idx,
                  const unsigned int nb, const unsigned int lp,
                  const unsigned int row_d, const unsigned int dbg_every);

#endif
