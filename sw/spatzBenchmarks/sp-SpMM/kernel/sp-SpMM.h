// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Bowen Wang <bowwang@iis.ee.ethz.ch>

#ifndef SP_SPMM_H
#define SP_SPMM_H

#include <stdint.h>

/**
 * Structured-sparse SpMM (n:m, fp32) using Ventaglio's vfx instructions.
 *
 *   res[M x P]  =  A[M x N]  *  W[N x P]      (W is n:m structured sparse)
 *
 * W is stored compactly: per sparse row of W, P_W = (P / m) * n nonzeros
 * + a per-row packed index buffer.
 *
 * Implementation:
 *  - Unrolled by 2 over the M dim (v16, v18 accumulators).
 *  - Software-pipelined inner loop over N with double-buffered weights/idx
 *    (v8/v20 ↔ v4/v24) to overlap loads with compute.
 *  - `vventclr` at the top of every (m, m+1) iteration clears the bank
 *    so vfxmul.vrf scatters into a clean slate.
 *
 * @param res                   output buffer (M * P fp32 elements)
 * @param a                     activation matrix (M * N fp32, row-major)
 * @param w                     compact weight matrix (N * P_W fp32, row-major)
 * @param nm_index              packed index buffer (NM_INDEX_WORDS u32s)
 * @param M                     batch dim
 * @param N                     reduction dim
 * @param P                     dense output column count
 * @param P_W                   compact column count per sparse row
 * @param NM_INDEX_ROW_WORDS    u32 words per row of indices
 * @param idx_width             bits per packed index (= log2(m_sparse))
 * @param m_sparse              m in n:m (block width)
 * @param n_sparse              n in n:m (nonzeros per block)
 */
void sp_spmm(float *res,
             const float *a,
             const float *w,
             const uint32_t *nm_index,
             uint32_t M,
             uint32_t N,
             uint32_t P,
             uint32_t P_W,
             uint32_t NM_INDEX_ROW_WORDS,
             uint32_t idx_width,
             uint32_t m_sparse,
             uint32_t n_sparse);

#endif // SP_SPMM_H
