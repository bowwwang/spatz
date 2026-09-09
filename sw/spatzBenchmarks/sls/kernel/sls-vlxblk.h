// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SLS_VLXBLK_H
#define SLS_VLXBLK_H

#include <stdint.h>

// SparseLengthsSum (embedding-bag pooling) over fp16 rows of row_d = 16
// elements (32-B blocks), VLXBLK arm: 32 + 8 rows per bag in two gathers (m8 +
// m2), fp16 accumulate into 8 chunk-slot partial sums per bag
// (out[b][8][row_d], no final reduction), two bags in flight.
// Requirements (not asserted): row_d = 16; lp = 40 (32-row block + 8-row
// tail); nb even; idx carries >= 16 elements of tail padding (32-B index
// loads over-read).
void sls_vlxblk(__fp16 *out, const __fp16 *t, const uint16_t *idx,
                const unsigned int nb, const unsigned int lp,
                const unsigned int row_d, const unsigned int dbg_every);

#endif
