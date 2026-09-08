// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RADIX_VSOXEI_H
#define RADIX_VSOXEI_H

#include <stdint.h>

// Radix-partition record scatter, plain-RVV baseline (vsoxei32, column
// decomposition): n 16-B records to dst[slot[r]]. Reads the COLUMN-STAGED
// source srcT[d * n + r] = src[r * 4 + d] (see the erratum #2 note in
// radix-vsoxei.c), 64 records per chunk, one element scatter per record
// column d.
void scatter_vsoxei(uint32_t *dst, const uint32_t *srcT, const uint16_t *slot,
                    const unsigned int n);

#endif
