// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RADIX_VSXBLK_H
#define RADIX_VSXBLK_H

#include <stdint.h>

// Radix-partition record scatter, VSXBLK arm: n 16-B records (4 x e32)
// from row-major src to dst[slot[r]], 64 records per chunk, one
// vsxblkei16 block scatter per chunk (blk_len = 4 elements).
void scatter_vsxblk(uint32_t *dst, const uint32_t *src, const uint16_t *slot,
                    const unsigned int n);

#endif
