// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Bowen Wang <bowwang@iis.ee.ethz.ch>

#ifndef F32_SPMM_BASELINE_H
#define F32_SPMM_BASELINE_H

#include <stdint.h>

// Baseline RVV SpMM kernel using standard vluxei32 / vsuxei32 (gather-modify-
// scatter through L1) instead of the Ventaglio custom instructions.
//
//   res[M x P] = A[M x N] * W[N x P]   (W is n:m structured sparse, compact P_W)
//
// Dataflow: Gustavson's (row-by-row accumulation into dense output matrix).
// Loop order: N (sparse rows) outer, P_W tile middle, M (output rows) inner;
// M is unrolled by 2 so two independent gather/fmacc/scatter chains can
// pipeline in hardware. `byte_offsets` is a scratch buffer of size P_W,
// allocated by the caller, used for the SW N:M index translation.
void f32_spmm_baseline(float          *res,
                       const float    *a,
                       const float    *w,
                       const uint32_t *nm_index,
                       uint32_t       *byte_offsets,
                       uint32_t M,
                       uint32_t N,
                       uint32_t P,
                       uint32_t P_W,
                       uint32_t NM_INDEX_ROW_WORDS,
                       uint32_t idx_width,
                       uint32_t m_sparse,
                       uint32_t n_sparse);

#endif // F32_SPMM_BASELINE_H
