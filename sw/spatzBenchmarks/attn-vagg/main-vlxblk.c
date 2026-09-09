// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// attn-vagg (spattn), VLXBLK arm. All data (V pool, token ids, scores,
// expected output) is literal in the generated DATAHEADER
// (script/gen_data.py); nothing is generated on-core.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

#include DATAHEADER
#include "kernel/attn-vlxblk.c"

// Exact fp16 bit compare of every output lane against the bit-exact
// emulation from gen_data.py (integer loads only: no FPU loads after the
// kernel's vector stores).
static int verify_output(const __fp16 *out, const uint16_t *expected_bits,
                         const unsigned int n, const unsigned int hd) {
  const uint16_t *ow = (const uint16_t *)out;
  unsigned int fails = 0;
  for (unsigned int i = 0; i < n; ++i) {
    if (ow[i] != expected_bits[i]) {
      if (fails < 8)
        printf("FAILED q=%u d=%u got=0x%04x exp=0x%04x\n", i / hd, i % hd,
               (unsigned)ow[i], (unsigned)expected_bits[i]);
      fails++;
    }
  }
  if (fails)
    printf("FAILED total=%u/%u\n", fails, n);
  return fails > 255 ? 255 : (int)fails;
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
    // Start timer
    timer = benchmark_get_cycle();

    attn_vlxblk(attn_out, (const __fp16 *)attn_pool_bits, attn_idx_rows,
                (const __fp16 *)attn_p_bits, attn_l.NQ, attn_l.TOPK, attn_l.HD);
    asm volatile("fence" ::: "memory");

    // End timer
    timer = benchmark_get_cycle() - timer;

    error = verify_output(attn_out, attn_expected_bits, attn_l.NQ * attn_l.HD,
                          attn_l.HD);

#ifdef PRINT_RESULT
    // MACs = NQ * TOPK * HD (fp16)
    printf("attn-vagg vlxblk nq=%u topk=%u hd=%u ntok=%u cache=%d: took %u "
           "cycles %s (macs=%u)\n",
           attn_l.NQ, attn_l.TOPK, attn_l.HD, attn_l.NTOK, USE_CACHE, timer,
           error ? "CHECK-FAILED" : "CHECK-OK",
           attn_l.NQ * attn_l.TOPK * attn_l.HD);
#endif
  }

  // Wait for all cores to finish
  snrt_cluster_hw_barrier();
  set_eoc();

  return error;
}
