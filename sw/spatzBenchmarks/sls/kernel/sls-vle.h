// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SLS_VLE_H
#define SLS_VLE_H

#include <stdint.h>

// SparseLengthsSum baseline: each 32-B fp16 row loaded by scalar-computed
// address (vle16, 32 B) and added (fp16) into one of 8 slot accumulators
// (row l -> slot l % 8), two rows in flight; output = the 8 slot partials
// per bag like the VLXBLK arm. Requirements (not asserted): row_d = 16;
// lp a multiple of 8.
void sls_vle(__fp16 *out, const __fp16 *t, const uint16_t *idx,
             const unsigned int nb, const unsigned int lp,
             const unsigned int row_d, const unsigned int dbg_every);

#endif
