// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef VQGEMM_RVV_H
#define VQGEMM_RVV_H

#include <stdint.h>

// Fused VQ-GEMM, plain-RVV baseline, software-pipelined over the per-group
// decode (see the .c header comment): same 8-row blocking as the VLXBLK arm,
// w_k decoded group-by-group (scalar-addressed vle16) into a scratch row
// and reloaded as one m2 vector for the 8 MACs.
// Requirements on the data (not asserted):
//   * M is a multiple of 8 (8-row blocking);
//   * groups = N/cb_d is EVEN (two groups per hot-loop iteration);
//   * K is EVEN (scratch-row parity);
//   * wrow holds TWO scratch rows of N+16 elements each (double-buffered
//     by k parity; +16 = slack for the padded 32-B decode stores).

// 8-element entries (AQLM 2x8, u8 indices); decode stores padded to 32 B.
void vqgemm_rvv_d8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                   const __fp16 *cb1, const uint8_t *idx0,
                   const uint8_t *idx1, const __fp16 *scales, __fp16 *wrow,
                   const unsigned int M, const unsigned int N,
                   const unsigned int K);

// 16-element entries (VPTQ v16, u16 indices): the natural 32-B decode
// store needs no padding.
void vqgemm_rvv_d16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                    const __fp16 *cb1, const uint16_t *idx0,
                    const uint16_t *idx1, const __fp16 *scales, __fp16 *wrow,
                    const unsigned int M, const unsigned int N,
                    const unsigned int K);

#endif
