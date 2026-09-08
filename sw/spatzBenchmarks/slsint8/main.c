// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// slsint8 (paper: sls, int8 deployed config): SparseLengthsSum /
// embedding-bag over a row-wise-quantized int8 table (Meta/FBGEMM
// EmbeddingSpMDM8Bit class). Deployed rows are [32 x u8 | fp16 scale |
// fp16 bias] = 36 B (non-pow2); we use the SPLIT LAYOUT ruled in
// BENCHMARK_PLAN (a load-time transformation with identical bytes,
// index stream, and arithmetic): a 2 MiB data table of 32-B rows plus a
// parallel (scale, bias) array fetched with the same index.
// Per bag b: out_b[d] = sum_l ( s_l * row_l[d] ) + sum_l b_l
// (row-wise bias is element-uniform, so it folds into one scalar).
//
// Arms (VARIANT): 1 = VLXBLK: per lookup one vlxblkei16 gathers the
//                     32-B u8 row; widen u8->u16->u32 (vwmulu x2),
//                     vfcvt to f32, one vfmacc.vf with the row scale.
//                     Scale/bias pairs are scalar-loaded via the same
//                     index (flh) — the dequant work is identical in
//                     both arms.
//                 0 = vle baseline: same pipeline, row loaded with a
//                     scalar-addressed vle8.
// Check: SAMPLED (every 64th bag + the last; all 32 elements), fp32
// accumulate mirrored; tolerance covers FMA-vs-mul/add rounding.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

#include "bench_fill.h"

#define VLXBLKEI16_V(vd, rs1n, vs2)  "vlxblkei16.v v" #vd ", (x" #rs1n "), v" #vs2 "\n"
#define VSETBLKLEN(rs1n)             "vsetblklen x" #rs1n "\n"

#ifndef NROWS
#define NROWS 65536 // table rows (u16 id ceiling) -> 2 MiB data table
#endif
#ifndef VARIANT
#define VARIANT 1
#endif

#define ROW_D 32 // u8 elements per row -> 32-B blocks
#define NB 2048  // bags (T in the paper table)
#define LP 40    // lookups per bag (P; HPCA'20 "~tens")

#if (NROWS & (NROWS - 1)) != 0
#error "NROWS must be a power of two (masked id generation)"
#endif

typedef __fp16 f16;

static uint8_t sl_tbl[NROWS * ROW_D] __attribute__((section(".data"), aligned(128)));
static f16 sl_sb[NROWS * 2] __attribute__((section(".data"), aligned(64)));
static uint16_t sl_idx[NB * LP + 128] __attribute__((section(".data"), aligned(64)));
static float sl_out[NB * ROW_D] __attribute__((section(".data"), aligned(128)));
static uint16_t sl_seed16[128] __attribute__((section(".data"), aligned(64)));

// Safe __fp16 -> float (software conversion is broken on this
// toolchain; see vqgemv).
static inline float f16f(const __fp16 *p) {
  float x;
  asm("flh ft3, 0(%1)\n\t"
      "fcvt.s.h %0, ft3"
      : "=f"(x)
      : "r"(p)
      : "ft3");
  return x;
}

// Dequant-accumulate one gathered/loaded u8 row (in v4, vl=32 e8):
// widen u8->u16 (v8), u16->u32 (v12), convert to f32, vfmacc with the
// row scale into the f32 accumulator v24 (m4 shapes throughout).
#define DEQUANT_MAC                                                          \
  "vsetvli zero, %[d], e8, m1, ta, ma\n"                                     \
  "vwmulu.vx v8, v4, %[one]\n"                                               \
  "vsetvli zero, %[d], e16, m2, ta, ma\n"                                    \
  "vwmulu.vx v12, v8, %[one]\n"                                              \
  "vsetvli zero, %[d], e32, m4, ta, ma\n"                                    \
  "vfcvt.f.xu.v v12, v12\n"                                                  \
  "vfmacc.vf v24, %[fs], v12\n"

#if VARIANT == 1
static void sls_vlxblk(float *out, const uint8_t *t, const f16 *sb,
                       const uint16_t *idx, unsigned int nb) {
  // Gather at e16 (blk_len = ROW_D/2 = 16 x e16 = the same 32 bytes,
  // same entry numbers): the e8-data miss path hangs on the first
  // missing gather (wave-4 bisection; probe P7 passes L1-resident).
  // The register bytes are identical; the widen chain reinterprets
  // them at e8.
  register uint32_t bl asm("t0") = ROW_D / 2;  // x5
  register const uint8_t *tp asm("t1") = t;    // x6
  asm volatile(VSETBLKLEN(5) :: "r"(bl), "r"(tp));
  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bi = idx + b * LP;
    float bias = 0.0f;
    asm volatile("vsetvli zero, %[d], e32, m4, ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [d] "r"((uint32_t)ROW_D)
                 : "v24", "v25", "v26", "v27");
    for (unsigned int l = 0; l < LP; ++l) {
      const uint32_t id = bi[l];
      const float fs = f16f(&sb[2u * id]);
      bias += f16f(&sb[2u * id + 1u]);
      asm volatile("vsetvli zero, %[one_e], e16, m1, ta, ma\n"
                   "vmv.s.x v2, %[id]\n"
                   "vsetvli zero, %[dh], e16, m1, ta, ma\n"
                   VLXBLKEI16_V(4, 6, 2)
                   DEQUANT_MAC
                   :
                   : [one_e] "r"(1u), [id] "r"(id), [d] "r"((uint32_t)ROW_D),
                     [dh] "r"((uint32_t)(ROW_D / 2)),
                     [one] "r"(1u), [fs] "f"(fs), [dict] "r"(tp)
                   : "v2", "v4", "v8", "v9", "v12", "v13", "v14", "v15",
                     "v24", "v25", "v26", "v27", "memory");
    }
    asm volatile("vsetvli zero, %[d], e32, m4, ta, ma\n"
                 "vfadd.vf v24, v24, %[fb]\n"
                 "vse32.v v24, (%[o])\n"
                 :: [d] "r"((uint32_t)ROW_D), [fb] "f"(bias),
                    [o] "r"(out + b * ROW_D) : "memory");
  }
}
#else
static void sls_vle(float *out, const uint8_t *t, const f16 *sb,
                    const uint16_t *idx, unsigned int nb) {
  for (unsigned int b = 0; b < nb; ++b) {
    const uint16_t *bi = idx + b * LP;
    float bias = 0.0f;
    asm volatile("vsetvli zero, %[d], e32, m4, ta, ma\n"
                 "vmv.v.i v24, 0\n" :: [d] "r"((uint32_t)ROW_D)
                 : "v24", "v25", "v26", "v27");
    for (unsigned int l = 0; l < LP; ++l) {
      const uint32_t id = bi[l];
      const float fs = f16f(&sb[2u * id]);
      bias += f16f(&sb[2u * id + 1u]);
      const uint8_t *row = t + id * ROW_D;
      // Load as 16 x e16 (same 32 bytes) and reinterpret at e8: the
      // e8-data miss path hangs (see the vlxblk arm note).
      asm volatile("vsetvli zero, %[dh], e16, m1, ta, ma\n"
                   "vle16.v v4, (%[r])\n"
                   DEQUANT_MAC
                   :
                   : [r] "r"(row), [d] "r"((uint32_t)ROW_D),
                     [dh] "r"((uint32_t)(ROW_D / 2)), [one] "r"(1u),
                     [fs] "f"(fs)
                   : "v4", "v8", "v9", "v12", "v13", "v14", "v15",
                     "v24", "v25", "v26", "v27", "memory");
    }
    asm volatile("vsetvli zero, %[d], e32, m4, ta, ma\n"
                 "vfadd.vf v24, v24, %[fb]\n"
                 "vse32.v v24, (%[o])\n"
                 :: [d] "r"((uint32_t)ROW_D), [fb] "f"(bias),
                    [o] "r"(out + b * ROW_D) : "memory");
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
    // Data table: prime-period byte pattern head + vector replication;
    // (scale, bias) array likewise; ids vector-generated (bench_fill.h).
    {
      const unsigned int n = NROWS * ROW_D;
      for (unsigned int i = 0; i < 3904u; ++i)
        sl_tbl[i] = (uint8_t)((i * 61u) & 255u);
      bench_fill_rep(sl_tbl, n, 3904u); // u8: BF_HEAD_BYTES
    }
    {
      const unsigned int n = NROWS * 2u;
      for (unsigned int i = 0; i < 1952u; ++i)
        sl_sb[i] = (f16)(0.05f + 0.001f * (float)((i * 37u) & 511u));
      bench_fill_rep(sl_sb, n * 2u, 1952u * 2u); // f16: BF_HEAD_BYTES/2
    }
    for (unsigned int i = 0; i < 128; ++i)
      sl_seed16[i] = (uint16_t)i;
    for (unsigned int c = 0; c < NB * LP + 16u; c += 128) {
      asm volatile("vsetvli zero, %[n], e16, m4, ta, ma\n"
                   "vle16.v v8, (%[seed])\n"
                   "vadd.vx v8, v8, %[c]\n"
                   "vmul.vx v8, v8, %[mult]\n"
                   "vand.vx v8, v8, %[mask]\n"
                   "vse16.v v8, (%[dst])\n"
                   :
                   : [n] "r"(128u), [seed] "r"(sl_seed16), [c] "r"(c),
                     [mult] "r"(2654435761u), [mask] "r"(NROWS - 1u),
                     [dst] "r"(sl_idx + c)
                   : "v8", "v9", "v10", "v11", "memory");
    }
    bench_fill_zero(sl_out, sizeof(sl_out));
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif

    uint32_t t0 = benchmark_get_cycle();
#if VARIANT == 1
    sls_vlxblk(sl_out, sl_tbl, sl_sb, sl_idx, NB);
#else
    sls_vle(sl_out, sl_tbl, sl_sb, sl_idx, NB);
#endif
    asm volatile("fence" ::: "memory");
    uint32_t cycles = benchmark_get_cycle() - t0;

    // SAMPLED check (every 64th bag + the last; all elements): mirror
    // the kernel's association (fp32 scale-MACs in ascending l, bias
    // sum folded at the end). Tolerance covers vfmacc's fused rounding
    // vs the reference's separate mul+add.
    for (unsigned int s64 = 0; s64 <= NB / 64 && fails == 0; ++s64) {
      const unsigned int b = (s64 == NB / 64) ? (NB - 1) : (s64 * 64);
      float rs[LP], rb = 0.0f;
      for (unsigned int l = 0; l < LP; ++l) {
        const uint32_t id = sl_idx[b * LP + l];
        rs[l] = f16f(&sl_sb[2u * id]);
        rb += f16f(&sl_sb[2u * id + 1u]);
      }
      for (unsigned int d = 0; d < ROW_D && fails == 0; ++d) {
        float acc = 0.0f;
        for (unsigned int l = 0; l < LP; ++l)
          acc += rs[l] *
                 (float)sl_tbl[(uint32_t)sl_idx[b * LP + l] * ROW_D + d];
        acc += rb;
        const float got = sl_out[b * ROW_D + d];
        float err = got > acc ? got - acc : acc - got;
        float mag = acc < 0 ? -acc : acc;
        if (err > 0.02f + 0.0001f * mag) {
          printf("FAILED b=%d d=%d got=%f exp=%f\n", b, d, got, acc);
          fails = 1;
        }
      }
    }

    printf("slsint8 variant=%d nrows=%d bags=%d cache=%d: took %u cycles %s "
           "(rowbytes=%u)\n", VARIANT, NROWS, NB, USE_CACHE, cycles,
           fails ? "CHECK-FAILED" : "CHECK-OK",
           (unsigned)(NB * LP * ROW_D));
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return fails;
}
