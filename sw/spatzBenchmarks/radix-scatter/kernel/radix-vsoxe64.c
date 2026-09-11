// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// radix-scatter, plain-RVV baseline at e64 grain: column decomposition as
// in radix-vsoxei.c, but a record is two 64-bit halves, so a chunk takes
// TWO scatters of full-port-width (8-B) elements instead of four
// half-width vsoxei32 - half the element transactions of the e32 form.
// The instruction is vsoxei32 executed at vtype e64 (mixed EEW: u32 byte
// offsets, e64 data; ei64 indexed ops are reserved on RV32), so the same
// vwmulu-widened u16 -> u32 offsets serve directly and no index bytes are
// loaded beyond the slots. Reads the u64-grain column staging rs_srcT64
// of the same payload (a strided vle64 of the row-major records would
// hit erratum #2, like the e32 form's vlse32).

#include "radix-vsoxe64.h"
#include <stddef.h>

void scatter_vsoxe64(uint64_t *dst, const uint64_t *srcT64,
                     const uint16_t *slot, const unsigned int n) {
  const unsigned int hd = 2; // e64 elements per record (16-B records)

  for (unsigned int r = 0; r < n; r += 64) {
    // e32 byte offsets = slot * 16 (v4..v5 from the e16 m1 slots in v2)
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(64u));
    asm volatile("vle16.v v2, (%0)" ::"r"(slot + r) : "memory");
    asm volatile("vwmulu.vx v4, v2, %0" ::"r"(16u));

    for (unsigned int d = 0; d < hd; ++d) {
      asm volatile("vsetvli zero, %0, e32, m2, ta, ma" ::"r"(64u));
      asm volatile("vadd.vx v8, v4, %0" ::"r"(8u * d));
      // e64 data, EEW-32 index group (EMUL = m2 under vtype e64 m4)
      asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(64u));
      // unit-stride from the u64 column staging (column stride = n records)
      asm volatile("vle64.v v24, (%0)" ::"r"(srcT64 + d * n + r) : "memory");
      asm volatile("vsoxei32.v v24, (%0), v8" ::"r"(dst) : "memory");
    }
  }
}
