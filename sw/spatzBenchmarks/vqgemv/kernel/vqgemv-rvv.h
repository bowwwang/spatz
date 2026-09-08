// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef VQGEMV_RVV_H
#define VQGEMV_RVV_H

#include <stdint.h>

// Fused VQ-GEMV, plain-RVV baseline, software-pipelined in two rounds
// exactly like the VLXBLK arm (see the .c header comment). One output
// group (one codebook entry width) per outer iteration; entries are loaded
// by scalar-computed address (idx * cb_d) with vle16. K must be EVEN (two
// k-rows per hot-loop iteration, no odd epilogue) — not asserted.

// 8-element entries (AQLM 2x8, u8 indices). Output stores padded to 32 B.
void vqgemv_rvv_d8(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                   const __fp16 *cb1, const uint8_t *idx0,
                   const uint8_t *idx1, const __fp16 *scales,
                   const unsigned int K, const unsigned int N);

// 16-element entries (VPTQ v16, u16 indices): the natural 32-B store
// needs no padding.
void vqgemv_rvv_d16(__fp16 *c, const __fp16 *a, const __fp16 *cb0,
                    const __fp16 *cb1, const uint16_t *idx0,
                    const uint16_t *idx1, const __fp16 *scales,
                    const unsigned int K, const unsigned int N);

#endif
