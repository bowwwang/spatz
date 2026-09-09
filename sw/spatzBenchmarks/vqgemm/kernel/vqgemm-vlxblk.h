// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef VQGEMM_VLXBLK_H
#define VQGEMM_VLXBLK_H

#include <stdint.h>

// Fused VQ-GEMM, VLXBLK arm, 8-row-blocked and software-pipelined (see the
// .c header comment). c[M,N] = a[M,K] x W[K,N], W decoded on the fly; every
// decoded row feeds eight vfmacc. Requirements on the data (not asserted):
//   * M is a multiple of 8 (8-row blocking), K is EVEN and >= 4;
//   * scalar operands are fetched with lhu + fmv.w.x, not flh (RTL erratum
//     #5; see the .c header);
//   * idx0/idx1 carry at least N elements of tail padding: the index loads
//     run under the tile vtype (e16, m2, vl = N) and over-read N index
//     elements past the current k-row.

// u8 indices (AQLM 2x8: 256-entry codebooks).
void vqgemm_vlxblk_ei8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                       const __fp16 *cb1, const uint8_t *idx0,
                       const uint8_t *idx1, const __fp16 *scales,
                       const unsigned int M, const unsigned int N,
                       const unsigned int K, const unsigned int cb_d);

// u16 indices (VPTQ v16: 4096-entry codebooks).
void vqgemm_vlxblk_ei16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                        const __fp16 *cb1, const uint16_t *idx0,
                        const uint16_t *idx1, const __fp16 *scales,
                        const unsigned int M, const unsigned int N,
                        const unsigned int K, const unsigned int cb_d);

#endif
