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
//
// Style: exactly one instruction per asm statement; base addresses are
// passed as "r" operands (no register pinning, no mnemonic macros).

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
    for (int i = 0; i < 512; ++i) {
      const __fp16 *p = pb_a + ((i & 3) * 8);
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(8u));
      asm volatile("vle16.v v8, (%0)" ::"r"(p) : "memory");
    }
    printf("P1 vle16-vl8 ok\n");

    // P2: the vqgemv-rvv inner shape — two back-to-back vle16 vl=8 +
    // vfadd.vv / vfmul.vf / vfmacc.vf at e16 m1 vl=8, flh scalars.
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(8u));
    asm volatile("vmv.v.x v0, zero");
    for (int i = 0; i < 512; ++i) {
      float av, sc;
      asm volatile("flh %0, 0(%1)" : "=f"(av) : "r"(pb_a + (i & 31)));
      asm volatile("flh %0, 0(%1)" : "=f"(sc) : "r"(pb_b + (i & 31)));
      const __fp16 *p0 = pb_a + ((i & 3) * 8);
      const __fp16 *p1 = pb_b + ((i & 3) * 8);
      asm volatile("vle16.v v16, (%0)" ::"r"(p0) : "memory");
      asm volatile("vle16.v v20, (%0)" ::"r"(p1) : "memory");
      asm volatile("vfadd.vv v16, v16, v20");
      asm volatile("vfmul.vf v16, v16, %0" ::"f"(sc));
      asm volatile("vfmacc.vf v0, %0, v16" ::"f"(av));
    }
    printf("P2 vl8-arith ok\n");

    // P4: vluxei32 with e64 data, vl=16 (the pr-gather baseline shape).
    for (int i = 0; i < 512; ++i) {
      const uint32_t *o = pb_off + ((i & 3) * 16);
      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(16u));
      asm volatile("vle32.v v4, (%0)" ::"r"(o) : "memory");
      asm volatile("vsetvli zero, %0, e64, m4, ta, ma" ::"r"(16u));
      asm volatile("vluxei32.v v8, (%0), v4" ::"r"(pb_d) : "memory");
    }
    printf("P4 vluxei32-e64 ok\n");

    // P5: vluxei32 with e32 data, vl=16 (the nbforce baseline shape).
    for (int i = 0; i < 512; ++i) {
      const uint32_t *o = pb_off + ((i & 3) * 16);
      asm volatile("vsetvli zero, %0, e32, m1, ta, ma" ::"r"(16u));
      asm volatile("vle32.v v4, (%0)" ::"r"(o) : "memory");
      asm volatile("vluxei32.v v8, (%0), v4" ::"r"(pb_d) : "memory");
    }
    printf("P5 vluxei32-e32 ok\n");

    // P6: vlse16 strided load, vl=8, stride 80 B (the gatheragg shape).
    for (int i = 0; i < 512; ++i) {
      const uint16_t *p = pb_i16 + (i & 63);
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(8u));
      asm volatile("vlse16.v v8, (%0), %1" ::"r"(p), "r"(80u) : "memory");
    }
    printf("P6 vlse16 ok\n");

    // P7: vmv.s.x + vlxblkei16 with e8 DATA (32-B byte blocks) — the
    // slsint8 gather shape.
    {
      const uint32_t bl = 32;
      const uint8_t *tp = (const uint8_t *)pb_i16;
      asm volatile("vsetblklen %0" ::"r"(bl));
      for (int i = 0; i < 512; ++i) {
        const uint32_t id = (uint32_t)(i & 15);
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(1u));
        asm volatile("vmv.s.x v2, %0" ::"r"(id));
        asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(32u));
        asm volatile("vlxblkei16.v v4, (%0), v2" ::"r"(tp) : "memory");
      }
    }
    printf("P7 vlxblk-e8 ok\n");

    // P8: the slsint8 dequant pipeline — vwmulu e8->e16, e16->e32,
    // vfcvt.f.xu, vfmacc.vf at vl=32.
    asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(32u));
    asm volatile("vmv.v.i v24, 0");
    for (int i = 0; i < 512; ++i) {
      float fs;
      asm volatile("flh %0, 0(%1)" : "=f"(fs) : "r"(pb_a + (i & 31)));
      const uint8_t *r = (const uint8_t *)pb_i16 + ((i & 3) * 32);
      asm volatile("vsetvli zero, %0, e8, m1, ta, ma" ::"r"(32u));
      asm volatile("vle8.v v4, (%0)" ::"r"(r) : "memory");
      asm volatile("vwmulu.vx v8, v4, %0" ::"r"(1u));
      asm volatile("vsetvli zero, %0, e16, m2, ta, ma" ::"r"(32u));
      asm volatile("vwmulu.vx v12, v8, %0" ::"r"(1u));
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(32u));
      asm volatile("vfcvt.f.xu.v v12, v12");
      asm volatile("vfmacc.vf v24, %0, v12" ::"f"(fs));
    }
    printf("P8 dequant-pipe ok\n");

    // P9: vfadd.vf broadcast-add + vse32 (the slsint8 epilogue).
    for (int i = 0; i < 128; ++i) {
      float fb;
      asm volatile("flh %0, 0(%1)" : "=f"(fb) : "r"(pb_b + (i & 31)));
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(32u));
      asm volatile("vfadd.vf v24, v24, %0" ::"f"(fb));
      asm volatile("vse32.v v24, (%0)" ::"r"(pb_d) : "memory");
    }
    printf("P9 vfadd.vf ok\n");

    // P10: seeded vfredusum + immediate scalar read-back at e32 m4
    // vl=64 (the nbforce reduction shape), serialized.
    {
      float acc = 0.0f;
      asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(64u));
      asm volatile("vmv.v.i v24, 0");
      for (int i = 0; i < 512; ++i) {
        asm volatile("vfmv.s.f v4, %0" ::"f"(acc));
        asm volatile("vfredusum.vs v4, v24, v4");
        asm volatile("vfmv.f.s %0, v4" : "=f"(acc));
      }
      pb_d[0] = (double)acc;
    }
    printf("P10 vfredusum ok\n");

    // P11: FOUR back-to-back vlxblkei16 gathers at e32 m4 vl=64
    // (the nbforce field-gather shape; two are proven, four are not).
    {
      const uint32_t bl = 4;
      const float *b0 = (const float *)pb_d;
      asm volatile("vsetblklen %0" ::"r"(bl));
      for (int i = 0; i < 512; ++i) {
        const uint16_t *ip = pb_i16 + (i & 15);
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(16u));
        asm volatile("vle16.v v4, (%0)" ::"r"(ip) : "memory");
        asm volatile("vsetvli zero, %0, e32, m4, ta, ma" ::"r"(64u));
        asm volatile("vlxblkei16.v v8, (%0), v4" ::"r"(b0) : "memory");
        asm volatile("vlxblkei16.v v12, (%0), v4" ::"r"(b0) : "memory");
        asm volatile("vlxblkei16.v v16, (%0), v4" ::"r"(b0) : "memory");
        asm volatile("vlxblkei16.v v20, (%0), v4" ::"r"(b0) : "memory");
      }
    }
    printf("P11 4x-vlxblk ok\n");

    // P13: vfadd.vv at e32 m8 vl=256 (m8 ARITHMETIC — the remaining
    // gatheragg suspect; vqdecode proves m8 gathers/stores are fine).
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(256u));
    asm volatile("vmv.v.i v8, 0");
    asm volatile("vmv.v.i v16, 0");
    for (int i = 0; i < 512; ++i)
      asm volatile("vfadd.vv v16, v16, v8");
    printf("P13 m8-vfadd ok\n");

    // P14: m8 vlxblk gather + DEPENDENT m8 vfadd (the exact gatheragg
    // inner shape).
    {
      const uint32_t bl = 32;
      const uint8_t *tp = (const uint8_t *)pb_i16;
      asm volatile("vsetblklen %0" ::"r"(bl));
      for (int i = 0; i < 512; ++i) {
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(8u));
        asm volatile("vle16.v v2, (%0)" ::"r"(pb_i16) : "memory");
        asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(256u));
        asm volatile("vlxblkei16.v v8, (%0), v2" ::"r"(tp) : "memory");
        asm volatile("vfadd.vv v16, v16, v8");
      }
    }
    printf("P14 m8-gather-vfadd ok\n");

    // P12 (LAST — known to hang, 2026-09-08 probe run): vse16 vl=8 =
    // 16-B vector stores. The cache port's store path hangs on vector
    // stores narrower than 32 B; loads (P1) and vl=8 arithmetic (P2)
    // are fine.
    for (int i = 0; i < 512; ++i) {
      __fp16 *p = pb_c + ((i & 3) * 8);
      asm volatile("vsetvli zero, %0, e16, m1, ta, ma" ::"r"(8u));
      asm volatile("vse16.v v0, (%0)" ::"r"(p) : "memory");
    }
    printf("P12 vse16-vl8 ok\n");

    printf("isaprobe: took 0 cycles CHECK-OK (all probes passed)\n");
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return 0;
}
