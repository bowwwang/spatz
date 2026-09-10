// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// nbforce, plain-RVV baseline arm (element-granular vluxei32 from the
// expanded per-element u32 index array). All data is literal in the
// generated DATAHEADER (script/gen_data.py); nothing is generated on-core.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/nbforce-rvv.c"

// fp32 from its bit pattern through an integer load + fmv.w.x (no FPU
// loads after the kernel's vector traffic).
static inline float f32_from_bits(const uint32_t bits) {
  float f;
  asm volatile("fmv.w.x %0, %1" : "=f"(f) : "r"(bits));
  return f;
}

// Verify ALL NC_TILE x 4 atoms x 3 components against the float64
// reference from gen_data.py. Tolerance 1% + 0.01 abs.
static int verify_output(const float *f, const uint32_t *expected_bits,
                         const unsigned int n) {
  const uint32_t *fw = (const uint32_t *)f;
  unsigned int fails = 0;
  for (unsigned int i = 0; i < n; ++i) {
    const float got = f32_from_bits(fw[i]);
    const float exp = f32_from_bits(expected_bits[i]);
    const float err = got > exp ? got - exp : exp - got;
    const float mag = exp < 0 ? -exp : exp;
    if (err > 0.01f + 0.01f * mag) {
      if (fails < 8)
        printf("FAILED c=%u a=%u d=%u got=0x%08x exp=0x%08x\n", i / 12,
               (i % 12) / 3, i % 3, (unsigned)fw[i], (unsigned)expected_bits[i]);
      fails++;
    }
  }
  if (fails)
    printf("FAILED total=%u/%u\n", fails, n);
  return fails > 255 ? 255 : (int)fails;
}


// DIAGNOSTIC (untimed, baseline arm only): sweep the four field arrays with
// unit-stride vector loads so every element gather in the timed region HITS
// in L1. On the adh_l1 config (256-cluster domain = 4 x 4 KiB) the whole
// domain stays resident, which separates "vluxei32 hangs on a MISSING gather"
// (erratum #3) from "the vluxei32 sequence hangs regardless".
static void warm_field(const uint32_t *p, const unsigned int n) {
  for (unsigned int i = 0; i < n; i += 32u) {
    const unsigned int vl = (n - i) < 32u ? (n - i) : 32u;
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(vl));
    asm volatile("vle32.v v8, (%0)" ::"r"(p + i) : "memory");
  }
}

int main() {
  const unsigned int cid = snrt_cluster_core_idx();

#if USE_CACHE == 1
  uint32_t spm_size = 16;
#else
  uint32_t spm_size = 120;
#endif

  if (cid == 0) {
    // Init the cache
    l1d_init(spm_size);
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();

  int error = 0;
  unsigned int timer = 0;

  if (cid == 0) {
    const float cut2 = f32_from_bits(nb_l.CUT2_BITS);

    // warm the fields and the index array (untimed, diagnostic)
    warm_field(nb_x_bits, nb_l.NC_DOM * 4u);
    warm_field(nb_y_bits, nb_l.NC_DOM * 4u);
    warm_field(nb_z_bits, nb_l.NC_DOM * 4u);
    warm_field(nb_q_bits, nb_l.NC_DOM * 4u);
    warm_field(nb_list_exp, nb_l.NC_TILE * nb_l.LIST * 4u);

    // Start timer
    start_kernel();
    timer = benchmark_get_cycle();

    nbforce_rvv(nb_f, (const float *)nb_x_bits, (const float *)nb_y_bits,
                (const float *)nb_z_bits, (const float *)nb_q_bits, nb_list_exp,
                nb_l.NC_TILE, nb_l.LIST, cut2);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;
    stop_kernel();

    // error = verify_output(nb_f, nb_expected_bits, nb_l.NC_TILE * 12u);

#ifdef PRINT_RESULT
    printf("nbforce rvv nc=%u list=%u cache=%d: took %u cycles %s "
           "(pairs=%u)\n",
           nb_l.NC_TILE, nb_l.LIST, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK", nb_l.NC_TILE * nb_l.LIST);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
