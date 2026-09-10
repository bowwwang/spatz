// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef SLS32_VLE_H
#define SLS32_VLE_H

#include <stdint.h>

// sls-2 plain-RVV baseline: one unit-stride 128-B vle32 per lookup at a
// scalar-computed address, accumulated into 8 slot accumulators (row l ->
// slot l % 8), one load in flight. row_d == 32, lp a multiple of 8.
void sls32_vle(float *out, const float *t, const uint16_t *idx,
               const unsigned int nb, const unsigned int lp,
               const unsigned int row_d);

#endif
