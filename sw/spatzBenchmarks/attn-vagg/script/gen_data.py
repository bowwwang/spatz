#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# attn-vagg data generator: emits data/data_n<ntok>.h for the paper config
#   NQ=16 decode queries, TOPK=1024 selected tokens per query, HD=128
#   (256-B fp16 V rows = exactly m4 @ e16), NTOK=16384-token V pool (4 MiB).
# out[q, d] = sum_k p[q, k] * pool[idx[q, k], d]
#
# All values are exact dyadic rationals (multiples of 2^-10 / 2^-11), so the
# on-core fp16 arrays and this script agree bit-for-bit. The V pool is NOT
# emitted literally (4 MiB): the core fills it from the same closed-form
# pattern (head of HEAD elements, tiled) with the vector helper in
# include/bench_fill.h; this script reproduces that fill exactly to compute
# the expected output. Indices (32 KiB) and scores (32 KiB) are literals.

import argparse
import pathlib
import numpy as np

HEAD = 1952  # fp16 elements in the fill head = BF_HEAD_BYTES / 2


def pool_fill(n):
    # Mirror of fill_pool() in main-*.c: head[i] = (mix(i) >> 22 & 1023 - 512)
    # / 1024 (exact multiples of 2^-10 in [-0.5, 0.5)), tiled with period HEAD
    # by bench_fill_rep (doubling copies of the head keep the period).
    # mix() is an fmix32-style u32 integer mixer. It must be NONLINEAR in i:
    # rows start at offsets r*128 mod HEAD, so with a linear head (i*mult &
    # 1023, or the top bits of a multiplicative hash) two row patterns differ
    # by a near-CONSTANT per lane (as small as 32/1024) and a wrong gathered
    # row hides under the verifier's slack; with the mixer, two rows differ
    # by ~triangular(-1, 1) per lane and a wrong row is flagged on >= 64 of
    # the 128 lanes for every pattern pair.
    i = np.arange(HEAD, dtype=np.uint64)
    x = (i * 2654435761) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 13
    head = ((((x >> 22) & 1023).astype(np.int64) - 512) / 1024.0).astype(np.float16)
    if n <= HEAD:
        return head[:n]
    reps = (n + HEAD - 1) // HEAD
    return np.tile(head, reps)[:n]


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays (expected output) must not carry the .data section
    # attribute: a const object in .data is a section type conflict.
    attr = ("__attribute__((section(\".data\"), aligned({})))".format(align)
            if data_section else "__attribute__((aligned({})))".format(align))
    out = "static {} {}[{}] {} = {{\n".format(ctype, name, len(vals), attr)
    line = "   "
    for v in vals:
        s = " " + fmt.format(v) + ","
        if len(line) + len(s) > 78:
            out += line + "\n"
            line = "   "
        line += s
    out += line.rstrip(",") + "};\n\n"
    return out


def emit(NQ, TOPK, HD, NTOK, out_dir):
    assert NTOK & (NTOK - 1) == 0, "NTOK must be a power of two (masked ids)"
    ii = np.arange(NQ * TOPK, dtype=np.int64)
    # Top-K token ids: Knuth multiplicative hash scatter over the pool (the
    # access pattern of the measured kernel); low bits of the product are
    # wrap-independent, so int64 here == the u32 arithmetic on the core.
    idx = (ii * 2654435761) & (NTOK - 1)
    # Scores: exact multiples of 2^-11 in [32/2048, 62/2048] = [0.0156, 0.0303]
    # (the measured kernel used 0.01..0.023). The lower bound keeps ONE wrong
    # gathered row detectable: it perturbs lane d by p * |dx|, |dx| a multiple
    # of 2^-10 up to ~1, above the verifier's 0.002 + 0.002*|exp| slack on
    # >= 64 of the 128 lanes (|exp| <= ~2.05, so the slack is ~3 fp16 ulp).
    p = ((32 + ii % 31) / 2048.0).astype(np.float16)
    V = pool_fill(NTOK * HD).astype(np.float64).reshape(NTOK, HD)

    # Bit-exact emulation of the kernel's fp16 arithmetic. Each output lane
    # (q, d) accumulates independently in ascending k, and both arms execute
    # the same op per k: acc = f16(p * x + acc) (vfmacc.vf = single-rounding
    # FMA). p has <= 6 significant bits at a 2^-11 granule and x <= 10 at
    # 2^-10, so p * x is a multiple of 2^-21, acc (fp16, |acc| < 4) one of
    # 2^-24: the sum spans <= 26 bits and is exact in float64, and the cast
    # to float16 is exactly the RNE-rounded fp16 result. The expected values
    # match the hardware to the bit and the verifier can be tight (few ulp).
    idx_qk = idx.reshape(NQ, TOPK)
    p64 = p.astype(np.float64).reshape(NQ, TOPK)
    acc = np.zeros((NQ, HD), dtype=np.float16)
    for k in range(TOPK):
        x = V[idx_qk[:, k]]                                                   # gathered rows (NQ, HD)
        acc = (p64[:, k:k + 1] * x + acc.astype(np.float64)).astype(np.float16)  # vfmacc.vf
    expected = acc.reshape(NQ * HD).astype(np.float64)

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by attn-vagg/script/gen_data.py\n"
         "// config: n{}\n\n".format(NTOK))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NQ;   // decode queries\n"
          "  unsigned int TOPK; // selected tokens per query\n"
          "  unsigned int HD;   // head dim = fp16 elements per V row (128 = m4 @ e16)\n"
          "  unsigned int NTOK; // tokens in the V pool (NTOK * HD * 2 B footprint)\n"
          "  unsigned int HEAD; // pool fill head length (elements)\n"
          "} attn_layer;\n\n")
    s += ("const attn_layer attn_l = {{.NQ = {}, .TOPK = {}, .HD = {}, .NTOK = {}, "
          ".HEAD = {}}};\n\n").format(NQ, TOPK, HD, NTOK, HEAD)
    s += c_array("attn_idx", "uint16_t", idx, "{}")
    s += c_array("attn_p", "__fp16", p.astype(np.float64), "(__fp16){:.10g}")
    s += c_array("attn_expected", "const float", expected, "{:.9g}f", data_section=False)
    # V pool + output buffers: sized here, filled on-core
    s += ("static __fp16 attn_pool[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static __fp16 attn_out[{1}] __attribute__((section(\".data\"), aligned(128)));\n"
          ).format(NTOK * HD, NQ * HD)
    (out_dir / "data_n{}.h".format(NTOK)).write_text(s)
    print("wrote data_n{}.h  (NQ={} TOPK={} HD={} NTOK={}, |expected|max={:.4f})".format(
        NTOK, NQ, TOPK, HD, NTOK, np.abs(expected).max()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, default=16)
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--hd", type=int, default=128)
    ap.add_argument("--ntok", type=int, default=16384)
    args = ap.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    emit(args.nq, args.topk, args.hd, args.ntok, out_dir)
