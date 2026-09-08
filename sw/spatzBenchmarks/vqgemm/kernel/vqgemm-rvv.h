// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef VQGEMM_RVV_H
#define VQGEMM_RVV_H

#include <stdint.h>

// Fused VQ-GEMM, plain-RVV baseline, 8-element entries (AQLM 2x8, u8
// indices): same 4-row tiling as the VLXBLK arm, but w_k is decoded
// group-by-group (scalar-addressed vle16) into the scratch row wrow[N+16]
// (decode stores padded to 32 B) and reloaded as one m4 vector for the MACs.
void vqgemm_rvv_d8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                   const __fp16 *cb1, const uint8_t *idx0,
                   const uint8_t *idx1, const __fp16 *scales, __fp16 *wrow,
                   const unsigned int M, const unsigned int N,
                   const unsigned int K);

// Same baseline, 16-element entries (VPTQ v16, u16 indices): the natural
// 32-B decode store needs no padding.
void vqgemm_rvv_d16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                    const __fp16 *cb1, const uint16_t *idx0,
                    const uint16_t *idx1, const __fp16 *scales, __fp16 *wrow,
                    const unsigned int M, const unsigned int N,
                    const unsigned int K);

#endif
