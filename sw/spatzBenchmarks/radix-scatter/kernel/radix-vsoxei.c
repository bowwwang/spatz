// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// radix-scatter, plain-RVV baseline: column decomposition, the
// dictdecode-style strong baseline. Per 64-record chunk: widen the u16
// slots once to e32 record byte offsets (vwmulu by 16), then per record
// column d (0..3): offsets + 4d, one unit-stride source load of column d
// and one vsoxei32 element scatter. u16 byte offsets cannot address the
// 1 MiB destination, so e32 offsets are required (2x index bandwidth).
//
// Erratum #2 workaround: the natural source access is a strided load
// (vlse32, stride 16 B) from the row-major records, but strided vector
// loads corrupt one lane under cache misses on this port (rec=18560
// got=elem1-value; identical for vsuxei32 and vsoxei32 -> the scatter is
// innocent). The baseline therefore reads a column-major staging of the
// SAME payload, srcT[d * n + r] = src[r * 4 + d], built untimed by main,
// with unit-stride vle32.

#include "radix-vsoxei.h"
#include <stddef.h>

void scatter_vsoxei(uint32_t *dst, const uint32_t *srcT, const uint16_t *slot,
                    const unsigned int n) {
  const unsigned int rd = 4; // e32 elements per record (16-B records)

  for (unsigned int r = 0; r < n; r += 64) {
    // e32 byte offsets = slot * 16 (v4..v7 from the e16 m2 slots in v2..v3)
    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(64u));
    asm volatile("vle16.v v2, (%0)" ::"r"(slot + r) : "memory");
    asm volatile("vwmulu.vx v4, v2, %0" ::"r"(16u));

    for (unsigned int d = 0; d < rd; ++d) {
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(64u));
      asm volatile("vadd.vx v8, v4, %0" ::"r"(4u * d));
      // unit-stride from the column staging (column stride = n records)
      asm volatile("vle32.v v16, (%0)" ::"r"(srcT + d * n + r) : "memory");
      asm volatile("vsoxei32.v v16, (%0), v8" ::"r"(dst) : "memory");
    }
  }
}
