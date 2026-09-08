// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef VQGEMV_VLXBLK_H
#define VQGEMV_VLXBLK_H

#include <stdint.h>

// Fused VQ-GEMV, VLXBLK arm, software-pipelined in two rounds (see the
// .c header comment). Requirements on the data (not asserted):
//   * K is EVEN (two k-rows per hot-loop iteration, no odd epilogue);
//   * idx0/idx1 carry at least VLMAX(e16,m2) = 256 elements of tail
//     padding: the index loads run under the group vtype and over-read
//     gvl index elements past the current k-row.

// u8 indices (AQLM 2x8: 256-entry codebooks).
void vqgemv_vlxblk_ei8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                       const __fp16 *cb1, const uint8_t *idx0,
                       const uint8_t *idx1, const __fp16 *scales,
                       const unsigned int K, const unsigned int N,
                       const unsigned int cb_d);

// u16 indices (VPTQ v16: 4096-entry codebooks).
void vqgemv_vlxblk_ei16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                        const __fp16 *cb1, const uint16_t *idx0,
                        const uint16_t *idx1, const __fp16 *scales,
                        const unsigned int K, const unsigned int N,
                        const unsigned int cb_d);

#endif
