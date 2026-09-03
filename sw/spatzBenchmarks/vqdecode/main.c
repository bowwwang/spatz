// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// vqdecode: vector-quantized codebook decode on the cache-based cluster.
// A stream of u16 codes selects D-element blocks from a codebook of
// TBL_KB KiB; blocks are gathered and written out packed. Sweeping TBL_KB
// across the L1 cache capacity is the paper's large-target-set experiment.
//
// Arms (VARIANT): 1 = VLXBLK (flattened 2-deep pipelined block gather)
//                 0 = direct RVV baseline (piecewise rule, d < one register:
//                     vluxei16 + widening-doubling offset expansion;
//                     16-bit byte offsets => TBL_KB <= 64 only for now)
// Memory modes:   USE_CACHE=1: data in L2 behind the 112 KiB L1 cache
//                 USE_CACHE=0: SPM mode - table DMA'd into L1 SPM when it
//                     fits (<= SPM_TBL_MAX KiB), else gathered from L2
//                     directly (the conventional no-cache alternative).

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

// Raw encodings: the prebuilt LLVM 14 has no VLXBLK mnemonics.
#define VLXBLK_WORD(f7, f3, vd, rs1n, vs2) \
  ".word ((" #f7 ")<<25)|((" #vs2 ")<<20)|((" #rs1n ")<<15)|((" #f3 ")<<12)|((" #vd ")<<7)|0x2B\n"
#define VLXBLKEI16_V(vd, rs1n, vs2) VLXBLK_WORD(0x0C, 0x5, vd, rs1n, vs2)
#define VSETBLKLEN(rs1n)            VLXBLK_WORD(0x0F, 0x0, 0, rs1n, 0)

#ifndef DICT_D
#define DICT_D 8
#endif
#ifndef TBL_KB
#define TBL_KB 64
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#if DICT_D == 4
#define D_LOG2 2
#define BLK_SHIFT "4" // log2(D*4): 16-B blocks (dictionary-decode class)
#elif DICT_D == 8
#define D_LOG2 3
#define BLK_SHIFT "5" // log2(D*4): 32-B blocks (VQ-LLM class)
#else
#error "unsupported DICT_D"
#endif
#define TBL_WORDS (TBL_KB * 256)
#define NBLK (TBL_WORDS / DICT_D)
#define N_CODES 16384
#define N_PAD 64 // pipeline prologue overread guard

static uint32_t vq_tbl[TBL_WORDS] __attribute__((section(".data"), aligned(128)));
static uint16_t vq_codes[N_CODES + N_PAD] __attribute__((section(".data"), aligned(64)));
static uint32_t vq_out[N_CODES * DICT_D] __attribute__((section(".data"), aligned(128)));

// ---------------- VLXBLK arm: flattened 2-deep pipeline -------------------
static void vqdecode_vlxblk(uint32_t *o, const uint32_t *t,
                            const uint16_t *c, unsigned int n_codes) {
  register uint32_t d_reg asm("t0") = DICT_D;     // x5
  register const uint32_t *t_reg asm("t1") = t;   // x6
  size_t chunk;
  asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
               : "=r"(chunk) : "r"(n_codes << D_LOG2));
  const size_t cpg = chunk >> D_LOG2; // codes per gather
  asm volatile(VSETBLKLEN(5) :: "r"(d_reg), "r"(t_reg));
  asm volatile("vsetvli zero, %[cp], e16, m1, ta, ma\n"
               "vle16.v v2, (%[i0])\n"
               "vle16.v v3, (%[i1])\n"
               :
               : [cp] "r"(cpg), [i0] "r"(c), [i1] "r"(c + cpg)
               : "v2", "v3", "memory");
  for (unsigned int done = 0; done < n_codes; done += 2 * cpg) {
    asm volatile(
        "vsetvli zero, %[ec], e32, m8, ta, ma\n"
        VLXBLKEI16_V(8, 6, 2)
        "vsetvli zero, %[cp], e16, m1, ta, ma\n"
        "vle16.v v2, (%[n0])\n"
        "vsetvli zero, %[ec], e32, m8, ta, ma\n"
        "vse32.v v8, (%[o0])\n"
        VLXBLKEI16_V(16, 6, 3)
        "vsetvli zero, %[cp], e16, m1, ta, ma\n"
        "vle16.v v3, (%[n1])\n"
        "vsetvli zero, %[ec], e32, m8, ta, ma\n"
        "vse32.v v16, (%[o1])\n"
        :
        : [ec] "r"(chunk), [cp] "r"(cpg),
          [n0] "r"(c + done + 2 * cpg), [n1] "r"(c + done + 3 * cpg),
          [o0] "r"(o + (done << D_LOG2)), [o1] "r"(o + (done << D_LOG2) + chunk),
          [dict] "r"(t_reg)
        : "v2", "v3", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "memory");
  }
}

// ------------- RVV baseline arm (vq-date verified pattern) ----------------
#if VARIANT == 0
#if TBL_KB > 64
#error "rvv baseline currently supports TBL_KB <= 64 (e16 byte offsets); e32 variant is staged work"
#endif
// u16 codes -> per-element e16 byte offsets via widening-doubling expansion.
// Ported verbatim from vq-date sp-dictdecode (verified 84/84 there); result
// in v4..v7 (D_LOG2 even) or v24..v27 (odd) - selected by parity below.
#if (D_LOG2 & 1)
#define EXPANDED_IDX "v24"
#else
#define EXPANDED_IDX "v4"
#endif
static inline void expand_offsets_e16(const uint16_t *codes, size_t n_idx) {
  asm volatile("vsetvli zero, %[n], e16, m2, ta, ma\n"
               "vle16.v v4, (%[codes])\n"
               "vsll.vi v4, v4, " BLK_SHIFT "\n"
               :
               : [n] "r"(n_idx), [codes] "r"(codes)
               : "v4", "v5", "memory");
  size_t len = n_idx;
  unsigned long delta = (DICT_D * 4) >> 1;
  for (unsigned int t = 0; t < D_LOG2; ++t) {
    const unsigned long dhi = delta << 16;
    if (!(t & 1))
      asm volatile("vsetvli zero, %[l], e16, m2, ta, ma\n"
                   "vwaddu.vx v24, v4, zero\n"
                   "vsetvli zero, %[l], e32, m4, ta, ma\n"
                   "vsll.vi v28, v24, 16\n"
                   "vadd.vv v24, v24, v28\n"
                   "vadd.vx v24, v24, %[dh]\n"
                   :
                   : [l] "r"(len), [dh] "r"(dhi)
                   : "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");
    else
      asm volatile("vsetvli zero, %[l], e16, m2, ta, ma\n"
                   "vwaddu.vx v4, v24, zero\n"
                   "vsetvli zero, %[l], e32, m4, ta, ma\n"
                   "vsll.vi v28, v4, 16\n"
                   "vadd.vv v4, v4, v28\n"
                   "vadd.vx v4, v4, %[dh]\n"
                   :
                   : [l] "r"(len), [dh] "r"(dhi)
                   : "v4", "v5", "v6", "v7", "v28", "v29", "v30", "v31");
    len <<= 1;
    delta >>= 1;
  }
}

static void vqdecode_rvv(uint32_t *o, const uint32_t *t, const uint16_t *c,
                         unsigned int n_codes) {
  unsigned int rem = n_codes << D_LOG2;
  while (rem > 0) {
    size_t gvl;
    asm volatile("vsetvli %[g], %[r], e32, m8, ta, ma"
                 : [g] "=r"(gvl) : [r] "r"(rem));
    const size_t n_idx = gvl >> D_LOG2;
    expand_offsets_e16(c, n_idx);
    asm volatile("vsetvli zero, %[g], e32, m8, ta, ma\n"
                 "vluxei16.v v8, (%[dict]), " EXPANDED_IDX "\n"
                 "vse32.v v8, (%[o0])\n"
                 :
                 : [g] "r"(gvl), [dict] "r"(t), [o0] "r"(o)
                 : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                   "memory");
    c += n_idx;
    o += gvl;
    rem -= gvl;
  }
}
#endif

int main(void) {
  const unsigned int cid = snrt_cluster_core_idx();

#if USE_CACHE == 1
  uint32_t spm_size = 16;
#else
  uint32_t spm_size = 120;
#endif
  if (cid == 0)
    l1d_init(spm_size);
  snrt_cluster_hw_barrier();

  int fails = 0;
  if (cid == 0) {
    // Data init on the CPU
    for (unsigned int i = 0; i < TBL_WORDS; ++i)
      vq_tbl[i] = 0xA0000000u + i;
    for (unsigned int k = 0; k < N_CODES + N_PAD; ++k)
      vq_codes[k] = (uint16_t)((k * 2654435761u) % NBLK);
    memset(vq_out, 0, sizeof(vq_out));

    const uint32_t *tbl = vq_tbl;
#if USE_CACHE == 0
    // SPM mode: DMA the table into L1 SPM when it fits, else gather from L2.
#if TBL_KB <= 96
    uint32_t *tbl_spm = (uint32_t *)snrt_l1alloc(sizeof(vq_tbl));
    snrt_dma_start_1d(tbl_spm, vq_tbl, sizeof(vq_tbl));
    snrt_dma_wait_all();
    tbl = tbl_spm;
#endif
#else
    // Cache mode: flush so the timed region starts cold for both arms.
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    vqdecode_vlxblk(vq_out, tbl, vq_codes, N_CODES);
#else
    vqdecode_rvv(vq_out, tbl, vq_codes, N_CODES);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // Golden check (CPU): every element.
    for (unsigned int k = 0; k < N_CODES && fails == 0; ++k)
      for (unsigned int d = 0; d < DICT_D; ++d)
        if (vq_out[k * DICT_D + d] != vq_tbl[(uint32_t)vq_codes[k] * DICT_D + d]) {
          printf("FAILED at code %d elem %d: got %u expected %u\n", k, d,
                 vq_out[k * DICT_D + d],
                 vq_tbl[(uint32_t)vq_codes[k] * DICT_D + d]);
          fails = 1;
        }

    printf("vqdecode variant=%d tbl=%dKiB cache=%d: took %u cycles %s\n",
           VARIANT, TBL_KB, USE_CACHE, cycles, fails ? "CHECK-FAILED" : "CHECK-OK");
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
