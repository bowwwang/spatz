// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// radix-scatter, VSXBLK arm (Polychroniou, Raghavan & Ross, SIGMOD'15):
// fixed 16-B records scattered to their partition slots. Software today
// emulates block scatter with cache-line buffers + non-temporal stores;
// vsxblk is the direct expression.
//
// Per 64-record chunk: the u16 slot ids are consumed as record numbers
// directly (vsetblklen = 4 elements per record), the 64 x 4 = 256 e32
// payload words fill exactly one e32 m8 register group (VLMAX), and one
// vsxblkei16 writes all 64 records to dst + slot * 16 B.

#include "radix-vsxblk.h"
#include <stddef.h>

void scatter_vsxblk(uint32_t *dst, const uint32_t *src, const uint16_t *slot,
                    const unsigned int n) {
  const unsigned int rd = 4; // e32 elements per record (16-B records)

  asm volatile("vsetblklen %0" ::"r"(rd));

  for (unsigned int r = 0; r < n; r += 64) { // 64 rec x 4 = 256 = vlmax
    // slot ids of the chunk: 64 x u16
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(64u));
    asm volatile("vle16.v v2, (%0)" ::"r"(slot + r) : "memory");

    // payload of the chunk: 256 x e32, then one block scatter
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(256u));
    asm volatile("vle32.v v8, (%0)" ::"r"(src + r * rd) : "memory");
    asm volatile("vsxblkei16.v v8, (%0), v2" ::"r"(dst) : "memory");
  }
}
