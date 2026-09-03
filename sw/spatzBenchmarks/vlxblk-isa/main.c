#include <benchmark.h>
#include <snrt.h>
#include <stdio.h>

// Gate 1 (USE_CACHE=1): place all test buffers behind the L1 cache in
// DRAM/L2 so every gather exercises the cache hierarchy; Gate 0
// (USE_CACHE=0) keeps them in L1 SPM, matching the vq-date environment.
#if USE_CACHE == 1
#define TEST_ALLOC snrt_l3alloc
#else
#define TEST_ALLOC snrt_l1alloc
#endif

// Copyright 2026 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Fatih Özdemir, ETH Zurich
#include "vector_macros.h" // via target_include_directories -> riscvTests macros

// The prebuilt LLVM 14 toolchain has no VLXBLK mnemonics: emit the raw
// encodings (custom-1 = 0x2B; see sw/toolchain/riscv-opcodes/
// opcodes-vlxblk_CUSTOM). Scalar operands are pinned to fixed registers
// via register-asm variables; vector register numbers are literal.
#define VLXBLK_WORD(f7, f3, vd, rs1n, vs2) \
  ".word ((" #f7 ")<<25)|((" #vs2 ")<<20)|((" #rs1n ")<<15)|((" #f3 ")<<12)|((" #vd ")<<7)|0x2B\n"
#define VLXBLKEI8_V(vd, rs1n, vs2)  VLXBLK_WORD(0x0C, 0x0, vd, rs1n, vs2)
#define VLXBLKEI16_V(vd, rs1n, vs2) VLXBLK_WORD(0x0C, 0x5, vd, rs1n, vs2)
#define VSETBLKLEN(rs1n)            VLXBLK_WORD(0x0F, 0x0, 0, rs1n, 0)

#include <stdint.h>
#include <string.h>

#define MAX_BLOCK_LEN 16
#define MAX_GROUPS 4
#define MAX_ELEMS (MAX_BLOCK_LEN * MAX_GROUPS)
#define NUM_CODEBOOK_BLOCKS 4

static const uint16_t codebook_init[NUM_CODEBOOK_BLOCKS * MAX_BLOCK_LEN] = {
    0x0100, 0x0101, 0x0102, 0x0103, 0x0104, 0x0105, 0x0106, 0x0107,
    0x0108, 0x0109, 0x010a, 0x010b, 0x010c, 0x010d, 0x010e, 0x010f,
    0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207,
    0x0208, 0x0209, 0x020a, 0x020b, 0x020c, 0x020d, 0x020e, 0x020f,
    0x0300, 0x0301, 0x0302, 0x0303, 0x0304, 0x0305, 0x0306, 0x0307,
    0x0308, 0x0309, 0x030a, 0x030b, 0x030c, 0x030d, 0x030e, 0x030f,
    0x0400, 0x0401, 0x0402, 0x0403, 0x0404, 0x0405, 0x0406, 0x0407,
    0x0408, 0x0409, 0x040a, 0x040b, 0x040c, 0x040d, 0x040e, 0x040f,
};

static const uint8_t indices_init[MAX_GROUPS] = {2, 0, 3, 1};

static uint16_t actual_init[MAX_ELEMS];
static uint16_t golden_init[MAX_ELEMS];

static uint16_t *codebook;
static uint8_t *indices;
static uint16_t *actual;
static uint16_t *golden;

static void init_buffers(void) {
#ifdef __SPIKE__
  codebook = (uint16_t *)codebook_init;
  indices = (uint8_t *)indices_init;
  actual = actual_init;
  golden = golden_init;
#else
  codebook = (uint16_t *)TEST_ALLOC(sizeof(codebook_init));
  indices = (uint8_t *)TEST_ALLOC(sizeof(indices_init));
  actual = (uint16_t *)TEST_ALLOC(sizeof(actual_init));
  golden = (uint16_t *)TEST_ALLOC(sizeof(golden_init));
#endif

  memcpy(codebook, codebook_init, sizeof(codebook_init));
  memcpy(indices, indices_init, sizeof(indices_init));
}

static void fill_golden(unsigned int block_len, unsigned int groups) {
  for (unsigned int g = 0; g < groups; ++g) {
    const uint16_t *block = codebook + ((unsigned int)indices[g] * block_len);
    for (unsigned int i = 0; i < block_len; ++i) {
      golden[g * block_len + i] = block[i];
    }
  }
}

static void run_vlxblkei8_case(unsigned int case_id, unsigned int block_len,
                               unsigned int groups) {
  const unsigned int elem_count = block_len * groups;

  memset(actual, 0, sizeof(actual_init));
  memset(golden, 0, sizeof(golden_init));
  fill_golden(block_len, groups);

  register const uint16_t *codebook_reg asm("t2") = codebook;   // x7
  register unsigned int block_len_reg asm("t3") = block_len;    // x28
  asm volatile("" ::"r"(codebook_reg), "r"(block_len_reg));

  asm volatile(VSETBLKLEN(28)
               "vsetvli zero, %[groups], e8, m2, ta, ma\n"
               "vle8.v v28, (%[indices])\n"
               "vsetvli zero, %[elem_count], e16, m4, ta, ma\n"
               VLXBLKEI8_V(16, 7, 28)
               "vse16.v v16, (%[actual])\n"
               :
               : [block_len] "r"(block_len_reg), [groups] "r"(groups),
                 [elem_count] "r"(elem_count), [indices] "r"(indices),
                 [codebook] "r"(codebook_reg), [actual] "r"(actual)
               : "v16", "v28", "memory");

  VMCMP(uint16_t, %hu, case_id, actual, golden, elem_count);
}

void TEST_CASE1(void) { run_vlxblkei8_case(1, 4, 4); }

void TEST_CASE2(void) { run_vlxblkei8_case(2, 8, 4); }

void TEST_CASE3(void) { run_vlxblkei8_case(3, 16, 2); }

// Benchmark-geometry cases: e32 data at m8, u8 indices, multiple vlxblk
// instructions interleaved with unit-stride stores - the sp-dictdecode
// check-kernel pattern that exposed a hang on the doublebw VLSU.
#define BG_ELEMS 128 // one m8 group at VLEN=512 (e32)
static uint32_t bg_dict_init[64];   // 32 blocks of 2 / 8 blocks of 16
static uint8_t bg_idx_init[2][128]; // 128 idx/chunk: sized for blk_len=1 (TC10)
static uint32_t bg_out_init[2 * BG_ELEMS];
static uint32_t bg_gold_init[2 * BG_ELEMS];

static void run_bg_case(unsigned int case_id, unsigned int block_len) {
  const unsigned int groups = BG_ELEMS / block_len;
  uint32_t *bg_dict;
  uint8_t (*bg_idx)[128];
  uint32_t *bg_out;
  uint32_t *bg_gold;
#ifdef __SPIKE__
  bg_dict = bg_dict_init; bg_idx = bg_idx_init;
  bg_out = bg_out_init; bg_gold = bg_gold_init;
#else
  bg_dict = (uint32_t *)TEST_ALLOC(sizeof(bg_dict_init) + 128);
  bg_dict = (uint32_t *)((((uintptr_t)bg_dict) + 127) & ~(uintptr_t)127);
  bg_idx = (uint8_t(*)[128])TEST_ALLOC(sizeof(bg_idx_init));
  bg_out = (uint32_t *)TEST_ALLOC(sizeof(bg_out_init));
  bg_gold = (uint32_t *)TEST_ALLOC(sizeof(bg_gold_init));
#endif
  for (unsigned int i = 0; i < 64; ++i)
    bg_dict[i] = 0xA0000000u + i;
  const unsigned int n_blocks = 64 / block_len;
  for (unsigned int c = 0; c < 2; ++c)
    for (unsigned int g = 0; g < 128; ++g) // fill full rows: blk_len=1 consumes 128 idx/chunk
      bg_idx[c][g] = (uint8_t)((g * 7 + c * 3) % n_blocks);
  for (unsigned int c = 0; c < 2; ++c)
    for (unsigned int g = 0; g < groups; ++g)
      for (unsigned int i = 0; i < block_len; ++i)
        bg_gold[c * BG_ELEMS + g * block_len + i] =
            bg_dict[(unsigned int)bg_idx[c][g] * block_len + i];
  memset(bg_out, 0, sizeof(bg_out_init));

  // Two chunks: loads first, then stores - the scheduled-kernel pattern.
  {
  register unsigned int bl_reg asm("t0") = block_len;   // x5
  register uint32_t *dict_reg asm("t1") = bg_dict;      // x6
  asm volatile(VSETBLKLEN(5)
               "vsetvli zero, %[gr], e8, m1, ta, ma\n"
               "vle8.v v4, (%[i0])\n"
               "vle8.v v5, (%[i1])\n"
               "vsetvli zero, %[ec], e32, m8, ta, ma\n"
               VLXBLKEI8_V(8, 6, 4)
               VLXBLKEI8_V(16, 6, 5)
               "vse32.v v8, (%[o0])\n"
               "vse32.v v16, (%[o1])\n"
               :
               : [bl] "r"(bl_reg), [gr] "r"(groups), [ec] "r"(BG_ELEMS),
                 [i0] "r"(&bg_idx[0][0]), [i1] "r"(&bg_idx[1][0]),
                 [dict] "r"(dict_reg), [o0] "r"(bg_out),
                 [o1] "r"(bg_out + BG_ELEMS)
               : "v4", "v5", "v8", "v16", "memory");
  }

  VMCMP(uint32_t, %u, case_id, bg_out, bg_gold, 2 * BG_ELEMS);
}

void TEST_CASE4(void) { run_bg_case(4, 2); }  // 8-B blocks (d2 geometry)
void TEST_CASE5(void) { run_bg_case(5, 16); } // 64-B blocks

// TC6: the sp-dictdecode check-kernel pattern at scale - the 4-chunk
// loads-first group (3 gathers, 3 stores, 1 gather, 1 store) repeated many
// times. TC4/TC5 (one 2-chunk group) pass on the doublebw VLSU while the
// benchmark check kernel deadlocks, so the hang needs accumulated
// instruction count / state - this reproduces it at ISA-test runtime.
#define TC6_ITERS 8
static uint32_t tc6_out_init[4 * BG_ELEMS];

static void run_tc6(unsigned int case_id, unsigned int block_len) {
  const unsigned int groups = BG_ELEMS / block_len;
  uint32_t *bg_dict;
  uint8_t (*bg_idx)[128];
  uint32_t *bg_out;
  uint32_t *bg_gold;
#ifdef __SPIKE__
  bg_dict = bg_dict_init; bg_idx = bg_idx_init;
  bg_out = tc6_out_init; bg_gold = bg_gold_init;
#else
  bg_dict = (uint32_t *)TEST_ALLOC(sizeof(bg_dict_init) + 128);
  bg_dict = (uint32_t *)((((uintptr_t)bg_dict) + 127) & ~(uintptr_t)127);
  bg_idx = (uint8_t(*)[128])TEST_ALLOC(sizeof(bg_idx_init));
  bg_out = (uint32_t *)TEST_ALLOC(sizeof(tc6_out_init));
  bg_gold = (uint32_t *)TEST_ALLOC(sizeof(bg_gold_init));
#endif
  for (unsigned int i = 0; i < 64; ++i)
    bg_dict[i] = 0xB0000000u + i;
  const unsigned int n_blocks = 64 / block_len;
  for (unsigned int c = 0; c < 2; ++c)
    for (unsigned int g = 0; g < 64; ++g)
      bg_idx[c][g] = (uint8_t)((g * 5 + c) % n_blocks);
  for (unsigned int c = 0; c < 2; ++c)
    for (unsigned int g = 0; g < groups; ++g)
      for (unsigned int i = 0; i < block_len; ++i)
        bg_gold[c * BG_ELEMS + g * block_len + i] =
            bg_dict[(unsigned int)bg_idx[c][g] * block_len + i];

  for (unsigned int it = 0; it < TC6_ITERS; ++it) {
    // Rotate output base so stores hit fresh addresses like the benchmark.
    uint32_t *out = bg_out + (it % 2) * (2 * BG_ELEMS);
    memset(out, 0, 2 * BG_ELEMS * sizeof(uint32_t));
    register unsigned int bl_reg asm("t0") = block_len;   // x5
    register uint32_t *dict_reg asm("t1") = bg_dict;      // x6
    asm volatile(VSETBLKLEN(5)
                 "vsetvli zero, %[gr], e8, m1, ta, ma\n"
                 "vle8.v v4, (%[i0])\n"
                 "vle8.v v5, (%[i1])\n"
                 "vle8.v v6, (%[i0])\n"
                 "vle8.v v7, (%[i1])\n"
                 "vsetvli zero, %[ec], e32, m8, ta, ma\n"
                 VLXBLKEI8_V(8, 6, 4)
                 VLXBLKEI8_V(16, 6, 5)
                 VLXBLKEI8_V(24, 6, 6)
                 "vse32.v v8, (%[o0])\n"
                 "vse32.v v16, (%[o1])\n"
                 "vse32.v v24, (%[o0])\n"
                 VLXBLKEI8_V(8, 6, 7)
                 "vse32.v v8, (%[o1])\n"
                 :
                 : [bl] "r"(bl_reg), [gr] "r"(groups), [ec] "r"(BG_ELEMS),
                   [i0] "r"(&bg_idx[0][0]), [i1] "r"(&bg_idx[1][0]),
                   [dict] "r"(dict_reg), [o0] "r"(out), [o1] "r"(out + BG_ELEMS)
                 : "v4", "v5", "v6", "v7", "v8", "v16", "v24", "memory");
  }
  // Final iteration wrote (idx0->o0 overwritten by idx0 chunk3, idx1->o1):
  // out0 = gather(idx[0]) [third store overwrote first], out1 = gather(idx[1]).
  uint32_t *out = bg_out + ((TC6_ITERS - 1) % 2) * (2 * BG_ELEMS);
  VMCMP(uint32_t, %u, case_id, out, bg_gold, 2 * BG_ELEMS);
}

void TEST_CASE6(void) { run_tc6(6, 2); }
void TEST_CASE7(void) { run_tc6(7, 16); }

// TC8/TC9: TC6's at-scale pattern with PURE back-to-back gathers -- no
// interleaved stores. Isolates the latency-sensitive corruption class:
// if these pass where TC6/TC7 fail (tcdm_latency >= 16), the failure
// needs load/store interleave, and load-only benchmark kernels are
// unaffected. One store AFTER the gather burst moves data out for the
// check (never concurrent with a gather).
static void run_tc8(unsigned int case_id, unsigned int block_len) {
  const unsigned int groups = BG_ELEMS / block_len;
  uint32_t *bg_dict;
  uint8_t (*bg_idx)[128];
  uint32_t *bg_out;
  uint32_t *bg_gold;
#ifdef __SPIKE__
  bg_dict = bg_dict_init; bg_idx = bg_idx_init;
  bg_out = tc6_out_init; bg_gold = bg_gold_init;
#else
  bg_dict = (uint32_t *)TEST_ALLOC(sizeof(bg_dict_init) + 128);
  bg_dict = (uint32_t *)((((uintptr_t)bg_dict) + 127) & ~(uintptr_t)127);
  bg_idx = (uint8_t(*)[128])TEST_ALLOC(sizeof(bg_idx_init));
  bg_out = (uint32_t *)TEST_ALLOC(sizeof(tc6_out_init));
  bg_gold = (uint32_t *)TEST_ALLOC(sizeof(bg_gold_init));
#endif
  for (unsigned int i = 0; i < 64; ++i)
    bg_dict[i] = 0xB0000000u + i;
  const unsigned int n_blocks = 64 / block_len;
  for (unsigned int c = 0; c < 2; ++c)
    for (unsigned int g = 0; g < 64; ++g)
      bg_idx[c][g] = (uint8_t)((g * 5 + c) % n_blocks);
  for (unsigned int c = 0; c < 2; ++c)
    for (unsigned int g = 0; g < groups; ++g)
      for (unsigned int i = 0; i < block_len; ++i)
        bg_gold[c * BG_ELEMS + g * block_len + i] =
            bg_dict[(unsigned int)bg_idx[c][g] * block_len + i];

  memset(bg_out, 0, 2 * BG_ELEMS * sizeof(uint32_t));
  for (unsigned int it = 0; it < TC6_ITERS; ++it) {
    register unsigned int bl_reg asm("t0") = block_len;   // x5
    register uint32_t *dict_reg asm("t1") = bg_dict;      // x6
    asm volatile(VSETBLKLEN(5)
                 "vsetvli zero, %[gr], e8, m1, ta, ma\n"
                 "vle8.v v4, (%[i0])\n"
                 "vle8.v v5, (%[i1])\n"
                 "vle8.v v6, (%[i0])\n"
                 "vle8.v v7, (%[i1])\n"
                 "vsetvli zero, %[ec], e32, m8, ta, ma\n"
                 VLXBLKEI8_V(8, 6, 4)
                 VLXBLKEI8_V(16, 6, 5)
                 VLXBLKEI8_V(24, 6, 6)
                 VLXBLKEI8_V(8, 6, 4)
                 VLXBLKEI8_V(16, 6, 7)
                 :
                 : [bl] "r"(bl_reg), [gr] "r"(groups), [ec] "r"(BG_ELEMS),
                   [i0] "r"(&bg_idx[0][0]), [i1] "r"(&bg_idx[1][0]),
                   [dict] "r"(dict_reg)
                 : "v4", "v5", "v6", "v7", "v8", "v16", "v24", "memory");
  }
  // Post-burst: single non-interleaved stores for the check.
  asm volatile("vsetvli zero, %[ec], e32, m8, ta, ma\n"
               "vse32.v v8, (%[o0])\n"
               "vse32.v v16, (%[o1])\n"
               :
               : [ec] "r"(BG_ELEMS), [o0] "r"(bg_out), [o1] "r"(bg_out + BG_ELEMS)
               : "memory");
  VMCMP(uint32_t, %u, case_id, bg_out, bg_gold, 2 * BG_ELEMS);
}

void TEST_CASE8(void) { run_tc8(8, 2); }
void TEST_CASE9(void) { run_tc8(9, 16); }

// TC10: blk_len=1 - one e32 element per block, i.e. a 4-B block. In scope
// when the block covers at least one memory port word: on ELEN=32
// datapaths 4 B == one port word; on ELEN=64 it is sub-word and excluded.
// History (issue 7, RESOLVED 2026-08-28 as a TEST bug): the original
// harness sized bg_idx rows at 64 B, but blk_len=1 consumes 128 indices
// per chunk, so the second gather's vle8 read 64 B past the allocation -
// X indices -> X addresses -> xbar/Snitch X asserts + hang. The RTL was
// exonerated by trace: index-word sync correct in every cycle. Fixed by
// widening bg_idx rows to 128 B; TC10 passes (gate = 11 PASSED on w32).
#if ELEN == 32
void TEST_CASE10(void) { run_bg_case(10, 1); }
#endif

// Benchmark-harness wrapper: ennest's riscvTests runtime glue never calls
// set_eoc(), so the whole rv64uv suite hangs the testbench on this branch
// (their own vadd does too). This target runs the identical VLXBLK test
// cases through the WORKING spatzBenchmarks choreography instead.
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

  int failed = 0;
  if (cid == 0) {
    num_failed = 0;
    test_case  = 0;
    enable_vec();
    init_buffers();

    TEST_CASE1();
    TEST_CASE2();
    TEST_CASE3();
    TEST_CASE4();
    TEST_CASE5();
    TEST_CASE6();
    TEST_CASE7();
    TEST_CASE8();
    TEST_CASE9();
#if ELEN == 32
    TEST_CASE10(); // blk_len=1 (one port word on 32-bit datapaths)
#endif

    if (num_failed > 0)
      printf("ERROR: vlxblk-isa failed %d tests!\n", num_failed);
    else
      printf("PASSED: vlxblk-isa, all test cases!\n");
    failed = num_failed;
  }

  snrt_cluster_hw_barrier();
  set_eoc();
  return failed;
}
