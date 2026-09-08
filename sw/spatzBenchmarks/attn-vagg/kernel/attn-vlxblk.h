// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ATTN_VLXBLK_H
#define ATTN_VLXBLK_H

#include <stdint.h>

// Top-K sparse-attention V aggregation, VLXBLK arm: per selected token one
// block gather of the whole hd-element fp16 row (vlxblkei16, blk_len = hd)
// in its natural m4 position and one vfmacc.vf into the m4 accumulator.
// out[q, :] = sum_k p[q, k] * pool[idx[q, k], :]
void attn_vlxblk(__fp16 *out, const __fp16 *pool, const uint16_t *idx,
                 const __fp16 *p, const unsigned int nq,
                 const unsigned int topk, const unsigned int hd);

#endif
