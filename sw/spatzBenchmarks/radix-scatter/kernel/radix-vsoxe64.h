// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef RADIX_VSOXE64_H
#define RADIX_VSOXE64_H

#include <stdint.h>

void scatter_vsoxe64(uint64_t *dst, const uint64_t *srcT64,
                     const uint16_t *slot, const unsigned int n);

#endif
