// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Bowen Wang <bowwang@iis.ee.ethz.ch>

#ifndef SP_SPMV_H
#define SP_SPMV_H

#include <stdint.h>

/**
 * Structured-sparse SpMV (n:m, fp32) using Ventaglio's vfx instructions.
 *
 *   res[P]   =  W[N x P]^T * a[N]      (W is n:m structured sparse)
 *
 * W is stored compactly: per sparse row of W, P_W = (P / m) * n nonzeros
 * + a per-row packed index buffer (`idx_width` bits per index).
 *
 * Prologue uses vfxmul.vrf to scatter the first row's contribution into
 * a freshly-cleared Ventaglio bank (via `vventclr`). Subsequent rows
 * accumulate via vfxmacc.vrf.
 *
 * @param res                   output buffer (P fp32 elements)
 * @param a                     activation vector (N fp32 elements)
 * @param w                     compact weight matrix (N * P_W fp32, row-major)
 * @param nm_index              packed index buffer (NM_INDEX_WORDS u32s)
 * @param N                     reduction dim
 * @param P_W                   compact column count per sparse row
 * @param NM_INDEX_ROW_WORDS    u32 words per row of indices
 * @param idx_width             bits per packed index (= log2(M_SPARSE))
 * @param m_sparse              m in n:m (block width, 4 for now)
 * @param n_sparse              n in n:m (nonzeros per block)
 */
void sp_spmv(float *res,
             const float *a,
             const float *w,
             const uint32_t *nm_index,
             uint32_t N,
             uint32_t P_W,
             uint32_t NM_INDEX_ROW_WORDS,
             uint32_t idx_width,
             uint32_t m_sparse,
             uint32_t n_sparse);

#endif // SP_SPMV_H
