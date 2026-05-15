// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Bowen Wang <bowwang@iis.ee.ethz.ch>

#include <stdint.h>

#include "f32_SpMV-baseline.h"

// Unpack packed N:M indices for ONE sparse row into byte offsets that
// vluxei32/vsuxei32 can consume directly.
static inline void unpack_nm_to_byte_offsets(const uint32_t *packed,
                                             uint32_t       *offsets,
                                             uint32_t        pw,
                                             uint32_t        idx_width,
                                             uint32_t        m_sparse,
                                             uint32_t        n_sparse) {
  const uint32_t idx_mask = (1u << idx_width) - 1u;
  uint32_t bitpos = 0;

  for (uint32_t j = 0; j < pw; j++) {
    uint32_t widx = bitpos >> 5;
    uint32_t boff = bitpos & 0x1fu;
    uint32_t idx  = (packed[widx] >> boff) & idx_mask;

    uint32_t block     = j / n_sparse;
    uint32_t dense_pos = block * m_sparse + idx;
    offsets[j] = dense_pos * (uint32_t)sizeof(float);

    bitpos += idx_width;
  }
}

void f32_spmv_baseline(float          *res,
                       const float    *a,
                       const float    *w,
                       const uint32_t *nm_index,
                       uint32_t       *byte_offsets,
                       uint32_t N,
                       uint32_t P_W,
                       uint32_t NM_INDEX_ROW_WORDS,
                       uint32_t idx_width,
                       uint32_t m_sparse,
                       uint32_t n_sparse) {
  const float    *_w  = w;
  const uint32_t *_nm = nm_index;

  for (uint32_t n = 0; n < N; n++) {
    // Step 1: software N:M index translation for this sparse row.
    unpack_nm_to_byte_offsets(_nm, byte_offsets, P_W,
                              idx_width, m_sparse, n_sparse);

    // Step 2: load scalar activation a[n].
    float act;
    asm volatile("flw %0, (%1)" : "=f"(act) : "r"(a + n));

    // Step 3: tile across the compact dimension P_W.
    const float    *_w_tile  = _w;
    const uint32_t *_bo_tile = byte_offsets;
    uint32_t        avl      = P_W;

    do {
      uint32_t vl;
      asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(avl));

      asm volatile("vle32.v    v4,  (%0)"          :: "r"(_bo_tile) : "memory");
      asm volatile("vle32.v    v8,  (%0)"          :: "r"(_w_tile)  : "memory");

      if (n == 0) {
        // First sparse row: v12 has no initialized contents and Spatz has no
        // `vmv` to clear it, so we use vfmul (write-only) to seed v12 with
        // a[0] * w[0,:]. The vluxei + vfmacc path is skipped to avoid
        // accumulating into uninitialized v12.
        asm volatile("vfmul.vf v12, v8, %0"        :: "f"(act));
      } else {
        // Subsequent rows: gather running partial sums, accumulate, scatter.
        asm volatile("vluxei32.v v12, (%0), v4"    :: "r"(res)      : "memory");
        asm volatile("vfmacc.vf  v12, %0, v8"      :: "f"(act));
      }
      asm volatile("vsuxei32.v v12, (%0), v4"      :: "r"(res)      : "memory");

      _w_tile  += vl;
      _bo_tile += vl;
      avl      -= vl;
    } while (avl > 0);

    _w  += P_W;
    _nm += NM_INDEX_ROW_WORDS;
  }
}
