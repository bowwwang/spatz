// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef VQGEMM_VLXBLK_H
#define VQGEMM_VLXBLK_H

#include <stdint.h>

// Fused VQ-GEMM, VLXBLK arm, u8 indices (AQLM 2x8: 256-entry codebooks).
// c[M,N] = a[M,K] x W[K,N], W decoded on the fly; 4 output rows per tile.
void vqgemm_vlxblk_ei8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                       const __fp16 *cb1, const uint8_t *idx0,
                       const uint8_t *idx1, const __fp16 *scales,
                       const unsigned int M, const unsigned int N,
                       const unsigned int K, const unsigned int cb_d);

// Fused VQ-GEMM, VLXBLK arm, u16 indices (VPTQ v16: 4096-entry codebooks).
void vqgemm_vlxblk_ei16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                        const __fp16 *cb1, const uint16_t *idx0,
                        const uint16_t *idx1, const __fp16 *scales,
                        const unsigned int M, const unsigned int N,
                        const unsigned int K, const unsigned int cb_d);

#endif
