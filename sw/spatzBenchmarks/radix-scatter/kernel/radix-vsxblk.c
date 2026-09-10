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
//
// Two-round software pipeline: chunk A (slots v2, payload v8-15) and
// chunk B (slots v3, payload v16-23) alternate; the next chunk's slot and
// payload loads are issued before the current chunk's scatter so the
// VLSU queue never drains. n must be a multiple of 128 (two chunks per
// loop iteration).

#include "radix-vsxblk.h"

void scatter_vsxblk(uint32_t *dst, const uint32_t *src, const uint16_t *slot,
                    const unsigned int n) {
  const unsigned int rd = 4;        // e32 elements per record (16-B records)
  const unsigned int chunk = 64;    // records per chunk
  const unsigned int pay_el = 256;  // e32 payload words per chunk = VLMAX
  const uint16_t *si = slot;
  const uint32_t *pi = src;

  asm volatile("vsetblklen %0" ::"r"(rd));

  // prologue: chunk 0 -> A
  asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(chunk));
  asm volatile("vle16.v v2, (%0)" ::"r"(si) : "memory");
  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(pay_el));
  asm volatile("vle32.v v8, (%0)" ::"r"(pi) : "memory");

  for (unsigned int r = 0; r < n; r += 2 * chunk) {
    // ---- load chunk B (records r+64 .. r+127) ----
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(chunk));
    asm volatile("vle16.v v3, (%0)" ::"r"(si + chunk) : "memory");
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(pay_el));
    asm volatile("vle32.v v16, (%0)" ::"r"(pi + pay_el) : "memory");

    // ---- scatter chunk A (records r .. r+63) ----
    asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(dst) : "memory");

    // ---- load chunk A (records r+128 .. r+191), if any ----
    if (r + 2 * chunk < n) {
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(chunk));
      asm volatile("vle16.v v2, (%0)" ::"r"(si + 2 * chunk) : "memory");
      asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(pay_el));
      asm volatile("vle32.v v8, (%0)" ::"r"(pi + 2 * pay_el) : "memory");
    }

    // ---- scatter chunk B (records r+64 .. r+127) ----
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(pay_el));
    asm volatile("vlxblkei16.v v16, (%0), v3" ::"r"(dst) : "memory");

    si += 2 * chunk;
    pi += 2 * pay_el;
  }
}
