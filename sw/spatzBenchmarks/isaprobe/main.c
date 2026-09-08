// Copyright 2026 ETH Zurich and University of Bologna.
// SPDX-License-Identifier: Apache-2.0
//
// isaprobe: minimal instruction-level hang probes for the flamingo
// cache port (VLEN=1024), motivated by the 2026-09-07 benchmark hangs.
// Each phase runs 512 iterations of one suspect pattern and prints its
// marker AFTER completing — where the UART stream stops names the
// trigger. Evidence so far: vle16 vl=8 loads PASS inside vqgemm; the
// hanging kernels differ in (P2) tiny-vl e16 arithmetic, (P4/P5)
// vluxei32, (P6) strided loads.

#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>
#include <string.h>

static __fp16 pb_a[64] __attribute__((section(".data"), aligned(128)));
static __fp16 pb_b[64] __attribute__((section(".data"), aligned(128)));
static __fp16 pb_c[64] __attribute__((section(".data"), aligned(128)));
static double pb_d[64] __attribute__((section(".data"), aligned(128)));
static uint32_t pb_off[64] __attribute__((section(".data"), aligned(128)));
static uint16_t pb_i16[512] __attribute__((section(".data"), aligned(128)));

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

  if (cid == 0) {
    for (int i = 0; i < 64; ++i) {
      pb_a[i] = (__fp16)(0.5f + 0.01f * i);
      pb_b[i] = (__fp16)(0.25f + 0.005f * i);
      pb_c[i] = 0;
      pb_d[i] = 1.0 + 0.1 * i;
      pb_off[i] = (uint32_t)((i * 7) % 32) * 8u; // e64 byte offsets, 8B-aligned
    }
    for (int i = 0; i < 512; ++i)
      pb_i16[i] = (uint16_t)i;
#if USE_CACHE == 1
    l1d_flush();
    l1d_wait();
#endif
    printf("P0 setup ok\n");

    // P1: vle16 vl=8 unit loads (16-B), aligned.
    for (int i = 0; i < 512; ++i)
      asm volatile("vsetvli zero, %[n], e16, m1, ta, ma\n"
                   "vle16.v v8, (%[p])\n" ::[n] "r"(8u),
                   [p] "r"(pb_a + ((i & 3) * 8))
                   : "v8", "memory");
    printf("P1 vle16-vl8 ok\n");

    // P2: the vqgemv-rvv inner shape — two back-to-back vle16 vl=8 +
    // vfadd.vv / vfmul.vf / vfmacc.vf at e16 m1 vl=8, flh scalars.
    asm volatile("vsetvli zero, %[n], e16, m1, ta, ma\n"
                 "vmv.v.x v0, zero\n" ::[n] "r"(8u) : "v0");
    for (int i = 0; i < 512; ++i) {
      float av, sc;
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(av) : [p] "r"(pb_a + (i & 31)));
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(sc) : [p] "r"(pb_b + (i & 31)));
      asm volatile("vle16.v v16, (%[p0])\n"
                   "vle16.v v20, (%[p1])\n"
                   "vfadd.vv v16, v16, v20\n"
                   "vfmul.vf v16, v16, %[sc]\n"
                   "vfmacc.vf v0, %[av], v16\n"
                   :
                   : [p0] "r"(pb_a + ((i & 3) * 8)), [p1] "r"(pb_b + ((i & 3) * 8)),
                     [sc] "f"(sc), [av] "f"(av)
                   : "v16", "v20", "memory");
    }
    printf("P2 vl8-arith ok\n");

    // P4: vluxei32 with e64 data, vl=16 (the pr-gather baseline shape).
    for (int i = 0; i < 512; ++i)
      asm volatile("vsetvli zero, %[n], e32, m1, ta, ma\n"
                   "vle32.v v4, (%[o])\n"
                   "vsetvli zero, %[n], e64, m4, ta, ma\n"
                   "vluxei32.v v8, (%[b]), v4\n"
                   :
                   : [n] "r"(16u), [o] "r"(pb_off + ((i & 3) * 16)),
                     [b] "r"(pb_d)
                   : "v4", "v8", "v9", "v10", "v11", "memory");
    printf("P4 vluxei32-e64 ok\n");

    // P5: vluxei32 with e32 data, vl=16 (the nbforce baseline shape).
    for (int i = 0; i < 512; ++i)
      asm volatile("vsetvli zero, %[n], e32, m1, ta, ma\n"
                   "vle32.v v4, (%[o])\n"
                   "vluxei32.v v8, (%[b]), v4\n"
                   :
                   : [n] "r"(16u), [o] "r"(pb_off + ((i & 3) * 16)),
                     [b] "r"(pb_d)
                   : "v4", "v8", "memory");
    printf("P5 vluxei32-e32 ok\n");

    // P6: vlse16 strided load, vl=8, stride 80 B (the gatheragg shape).
    for (int i = 0; i < 512; ++i)
      asm volatile("vsetvli zero, %[n], e16, m1, ta, ma\n"
                   "vlse16.v v8, (%[p]), %[st]\n"
                   :
                   : [n] "r"(8u), [p] "r"(pb_i16 + (i & 63)), [st] "r"(80u)
                   : "v8", "memory");
    printf("P6 vlse16 ok\n");

    // P7: vmv.s.x + vlxblkei16 with e8 DATA (32-B byte blocks) — the
    // slsint8 gather shape.
    {
      register uint32_t bl asm("t0") = 32;
      register const uint8_t *tp asm("t1") = (const uint8_t *)pb_i16;
      asm volatile("vsetblklen x5" :: "r"(bl), "r"(tp));
      for (int i = 0; i < 512; ++i)
        asm volatile("vsetvli zero, %[one], e16, m1, ta, ma\n"
                     "vmv.s.x v2, %[id]\n"
                     "vsetvli zero, %[d], e8, m1, ta, ma\n"
                     "vlxblkei16.v v4, (x6), v2\n"
                     :
                     : [one] "r"(1u), [id] "r"((uint32_t)(i & 15)),
                       [d] "r"(32u), "r"(tp)
                     : "v2", "v4", "memory");
    }
    printf("P7 vlxblk-e8 ok\n");

    // P8: the slsint8 dequant pipeline — vwmulu e8->e16, e16->e32,
    // vfcvt.f.xu, vfmacc.vf at vl=32.
    asm volatile("vsetvli zero, %[n], e32, m4, ta, ma\n"
                 "vmv.v.i v24, 0\n" ::[n] "r"(32u)
                 : "v24", "v25", "v26", "v27");
    for (int i = 0; i < 512; ++i) {
      float fs;
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(fs) : [p] "r"(pb_a + (i & 31)));
      asm volatile("vsetvli zero, %[d], e8, m1, ta, ma\n"
                   "vle8.v v4, (%[r])\n"
                   "vwmulu.vx v8, v4, %[one]\n"
                   "vsetvli zero, %[d], e16, m2, ta, ma\n"
                   "vwmulu.vx v12, v8, %[one]\n"
                   "vsetvli zero, %[d], e32, m4, ta, ma\n"
                   "vfcvt.f.xu.v v12, v12\n"
                   "vfmacc.vf v24, %[fs], v12\n"
                   :
                   : [r] "r"((const uint8_t *)pb_i16 + ((i & 3) * 32)),
                     [d] "r"(32u), [one] "r"(1u), [fs] "f"(fs)
                   : "v4", "v8", "v9", "v12", "v13", "v14", "v15", "v24",
                     "v25", "v26", "v27", "memory");
    }
    printf("P8 dequant-pipe ok\n");

    // P9: vfadd.vf broadcast-add + vse32 (the slsint8 epilogue).
    for (int i = 0; i < 128; ++i) {
      float fb;
      asm volatile("flh %[v], 0(%[p])" : [v] "=f"(fb) : [p] "r"(pb_b + (i & 31)));
      asm volatile("vsetvli zero, %[d], e32, m4, ta, ma\n"
                   "vfadd.vf v24, v24, %[fb]\n"
                   "vse32.v v24, (%[o])\n"
                   :: [d] "r"(32u), [fb] "f"(fb), [o] "r"(pb_d)
                   : "v24", "v25", "v26", "v27", "memory");
    }
    printf("P9 vfadd.vf ok\n");

    // P10: seeded vfredusum + immediate scalar read-back at e32 m4
    // vl=64 (the nbforce reduction shape), serialized.
    {
      float acc = 0.0f;
      asm volatile("vsetvli zero, %[n], e32, m4, ta, ma\n"
                   "vmv.v.i v24, 0\n" ::[n] "r"(64u)
                   : "v24", "v25", "v26", "v27");
      for (int i = 0; i < 512; ++i) {
        asm volatile("vfmv.s.f v4, %[a]\n"
                     "vfredusum.vs v4, v24, v4\n" ::[a] "f"(acc) : "v4");
        asm volatile("vfmv.f.s %0, v4" : "=f"(acc));
      }
      pb_d[0] = (double)acc;
    }
    printf("P10 vfredusum ok\n");

    // P11: FOUR back-to-back vlxblkei16 gathers at e32 m4 vl=64
    // (the nbforce field-gather shape; two are proven, four are not).
    {
      register uint32_t bl asm("t0") = 4;
      register const float *b0 asm("t1") = (const float *)pb_d;
      asm volatile("vsetblklen x5" :: "r"(bl), "r"(b0));
      for (int i = 0; i < 512; ++i)
        asm volatile("vsetvli zero, %[g], e16, m1, ta, ma\n"
                     "vle16.v v4, (%[ip])\n"
                     "vsetvli zero, %[n], e32, m4, ta, ma\n"
                     "vlxblkei16.v v8, (x6), v4\n"
                     "vlxblkei16.v v12, (x6), v4\n"
                     "vlxblkei16.v v16, (x6), v4\n"
                     "vlxblkei16.v v20, (x6), v4\n"
                     :
                     : [g] "r"(16u), [n] "r"(64u), [ip] "r"(pb_i16 + (i & 15)),
                       "r"(b0)
                     : "v4", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                       "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                       "v23", "memory");
    }
    printf("P11 4x-vlxblk ok\n");

    // P13: vfadd.vv at e32 m8 vl=256 (m8 ARITHMETIC — the remaining
    // gatheragg suspect; vqdecode proves m8 gathers/stores are fine).
    asm volatile("vsetvli zero, %[n], e32, m8, ta, ma\n"
                 "vmv.v.i v8, 0\n"
                 "vmv.v.i v16, 0\n"
                 :: [n] "r"(256u)
                 : "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
                   "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    for (int i = 0; i < 512; ++i)
      asm volatile("vfadd.vv v16, v16, v8\n" :::
                   "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23");
    printf("P13 m8-vfadd ok\n");

    // P14: m8 vlxblk gather + DEPENDENT m8 vfadd (the exact gatheragg
    // inner shape).
    {
      register uint32_t bl asm("t0") = 32;
      register const uint8_t *tp asm("t1") = (const uint8_t *)pb_i16;
      asm volatile("vsetblklen x5" :: "r"(bl), "r"(tp));
      for (int i = 0; i < 512; ++i)
        asm volatile("vsetvli zero, %[g], e16, m1, ta, ma\n"
                     "vle16.v v2, (%[ip])\n"
                     "vsetvli zero, %[n], e32, m8, ta, ma\n"
                     "vlxblkei16.v v8, (x6), v2\n"
                     "vfadd.vv v16, v16, v8\n"
                     :
                     : [g] "r"(8u), [n] "r"(256u), [ip] "r"(pb_i16),
                       "r"(tp)
                     : "v2", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
                       "v15", "v16", "v17", "v18", "v19", "v20", "v21",
                       "v22", "v23", "memory");
    }
    printf("P14 m8-gather-vfadd ok\n");

    // P12 (LAST — known to hang, 2026-09-08 probe run): vse16 vl=8 =
    // 16-B vector stores. The cache port's store path hangs on vector
    // stores narrower than 32 B; loads (P1) and vl=8 arithmetic (P2)
    // are fine.
    for (int i = 0; i < 512; ++i)
      asm volatile("vsetvli zero, %[n], e16, m1, ta, ma\n"
                   "vse16.v v0, (%[p])\n" ::[n] "r"(8u),
                   [p] "r"(pb_c + ((i & 3) * 8))
                   : "memory");
    printf("P12 vse16-vl8 ok\n");

    printf("isaprobe: took 0 cycles CHECK-OK (all probes passed)\n");
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return 0;
}
