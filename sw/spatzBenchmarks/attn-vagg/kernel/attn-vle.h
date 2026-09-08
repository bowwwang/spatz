// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ATTN_VLE_H
#define ATTN_VLE_H

#include <stdint.h>

// Top-K sparse-attention V aggregation, plain-RVV baseline: per selected
// token the row address is computed by scalar code (idx * hd), the row is
// loaded with vle16 (same m4 shape) and accumulated with vfmacc.vf.
// Identical arithmetic order to the VLXBLK arm.
// out[q, :] = sum_k p[q, k] * pool[idx[q, k], :]
void attn_vle(__fp16 *out, const __fp16 *pool, const uint16_t *idx,
              const __fp16 *p, const unsigned int nq, const unsigned int topk,
              const unsigned int hd);

#endif
