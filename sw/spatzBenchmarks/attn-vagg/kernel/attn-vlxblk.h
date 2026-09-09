// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef ATTN_VLXBLK_H
#define ATTN_VLXBLK_H

#include <stdint.h>

// Top-K sparse-attention V aggregation (paper row "spattn"), VLXBLK arm v1:
//   out[q, :] = sum_k p[q, k] * pool[idx[q, k], :]   over 256-B fp16 rows.
// Two tokens per step: one e16 m8 block gather of two rows (blk_len = hd),
// scaled by a two-half score vector at m8 and folded into the m4
// accumulator; two-round pipeline. Scores are fp16, loaded with flh one
// step ahead (hp-fmatmul idiom). idx is row-major with >= 16 ids of
// padding; hd == 128, topk even >= 4.
void attn_vlxblk(__fp16 *out, const __fp16 *pool, const uint16_t *idx,
                 const __fp16 *p, const unsigned int nq,
                 const unsigned int topk, const unsigned int hd);

#endif
