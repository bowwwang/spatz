#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# attn-vagg (paper row "spattn", sa-gemv) data generator: emits
# data/data_nq<NQ>.h
#   nq16 : paper config - NQ = 16 decode steps, TOPK = 1024 selected tokens
#          per step (Quest near-lossless budget), HD = 128 fp16 (256-B V
#          rows, FlashInfer page_size = 1), NTOK = 16384-token V pool (4 MiB)
#   nq2  : NQ = 2 (quick development config), same pool
# out[q, :] = sum_k p[q, k] * V[idx[q, k], :]      (fp16)
#
# Every array the kernels read is emitted literally (no on-core data
# generation): the V pool as fp16 bit patterns (uniform in [-0.5, 0.5)),
# the top-K token ids in the vlxblk arm's TRANSPOSED pair order (queries
# q, q+1 form a pair; position (q // 2) * 2 * TOPK + 2 * k + (q % 2) holds
# idx[q, k], so the two ids of step k are contiguous; +16 zero padding for
# the 32-B index loads that use 2), the same ids row-major for the vle
# baseline, the scores as fp16 bit patterns (uniform in [1/64, 1/32]), and
# the EXPECTED output as fp16 bit patterns: both arms accumulate per lane
# acc = f16(p * x + acc) in ascending k (vfmacc, single rounding; step 0 is
# the plain product), which is exact in float64 and RNE-cast to float16.

import argparse
import pathlib
import numpy as np


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays must not carry the .data section attribute: a const
    # object in .data is a section type conflict.
    attr = ("__attribute__((section(\".data\"), aligned({})))".format(align)
            if data_section else "__attribute__((aligned({})))".format(align))
    out = "static {} {}[{}] {} = {{\n".format(ctype, name, len(vals), attr)
    line = "   "
    for v in vals:
        s = " " + fmt(v) + ","
        if len(line) + len(s) > 78:
            out += line + "\n"
            line = "   "
        line += s
    out += line.rstrip(",") + "};\n\n"
    return out


def hex16(x):
    return "0x{:04x}".format(int(x))


def f16_bits(a):
    return np.asarray(a, dtype=np.float16).view(np.uint16)


def f16(a):
    return np.asarray(a, dtype=np.float64).astype(np.float16)


def emit(cfg, NQ, TOPK, HD, NTOK, out_dir, seed=42):
    assert NTOK <= 65536, "u16 token ids"
    assert TOPK % 2 == 0, "two tokens per gather step"
    assert HD == 128, "one 256-B fp16 row = e16 m4; two rows = m8"
    rng = np.random.default_rng(seed)

    V = rng.uniform(-0.5, 0.5, size=(NTOK, HD)).astype(np.float16)
    idx = rng.integers(0, NTOK, size=(NQ, TOPK), dtype=np.int64)
    p = rng.uniform(1.0 / 64, 1.0 / 32, size=(NQ, TOPK)).astype(np.float16)

    V64 = V.astype(np.float64)
    p64 = p.astype(np.float64)
    idx_pairs = idx.reshape(NQ // 2, 2, TOPK).transpose(0, 2, 1).reshape(-1)
    qq = min(3, NQ - 1)
    assert idx_pairs[(qq // 2) * 2 * TOPK + 2 * 5 + (qq % 2)] == idx[qq, 5]
    # vlxblk arm (v1): product rounded to fp16, then accumulated pairwise
    # (step s = tokens 2s, 2s+1): step 0 acc = f16(f16(p0 x0) + f16(p1 x1)),
    # then acc = f16(acc + f16(p_2s x)); acc = f16(acc + f16(p_2s+1 x))
    prod = f16(p64[:, :, None] * V64[idx])  # (NQ, TOPK, HD)
    prod64 = prod.astype(np.float64)
    acc = f16(prod64[:, 0] + prod64[:, 1])
    for k in range(2, TOPK):
        acc = f16(acc.astype(np.float64) + prod64[:, k])
    expected = acc
    # vle baseline: fused multiply-add in ascending k
    accf = f16(p64[:, 0:1] * V64[idx[:, 0]])
    for k in range(1, TOPK):
        accf = f16(p64[:, k:k + 1] * V64[idx[:, k]] + accf.astype(np.float64))
    expected_fused = accf

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by attn-vagg/script/gen_data.py\n"
         "// config: {} (NQ={} TOPK={} HD={} NTOK={}; V pool {} MiB)\n"
         "// Check: exact fp16 bit compare against the per-arm emulation.\n\n"
         ).format(cfg, NQ, TOPK, HD, NTOK, NTOK * HD * 2 // 2**20)
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NQ;   // decode queries (steps)\n"
          "  unsigned int TOPK; // selected tokens per query\n"
          "  unsigned int HD;   // head dim = fp16 elements per V row (128 = 256 B)\n"
          "  unsigned int NTOK; // tokens in the V pool (NTOK * HD * 2 B footprint)\n"
          "} attn_layer;\n\n")
    s += ("const attn_layer attn_l = {{.NQ = {}, .TOPK = {}, .HD = {}, .NTOK = {}}};\n\n"
          ).format(NQ, TOPK, HD, NTOK)
    s += "// V pool, fp16 bit patterns (read as const __fp16 * by the kernels)\n"
    s += c_array("attn_pool_bits", "uint16_t", f16_bits(V.reshape(-1)), hex16, align=128)
    s += "// top-K token ids, vlxblk arm: TRANSPOSED pairs [pair][k][2], +16 zero padding\n"
    s += c_array("attn_idx", "uint16_t",
                 np.concatenate([idx_pairs, np.zeros(16, dtype=np.int64)]).astype(np.uint16),
                 "{}".format)
    s += "// top-K token ids, row-major idx[q * TOPK + k] (+16 zero padding): vlxblk v1 + vle\n"
    s += c_array("attn_idx_rows", "uint16_t",
                 np.concatenate([idx.reshape(-1), np.zeros(16, dtype=np.int64)]).astype(np.uint16),
                 "{}".format)
    s += "// lane selector for the two-query score vector: 0 for lanes 0..127, 1 for 128..255\n"
    s += c_array("attn_sel", "uint16_t", np.concatenate([np.zeros(HD, dtype=np.uint16), np.ones(HD, dtype=np.uint16)]), "{}".format, align=128)
    s += "// attention scores p[q * TOPK + k], fp16 bit patterns\n"
    s += c_array("attn_p_bits", "uint16_t", f16_bits(p.reshape(-1)), hex16)
    s += "// expected output, vlxblk arm v1 (fp16 products, pairwise adds), fp16 bits\n"
    s += c_array("attn_expected_bits", "const uint16_t", f16_bits(expected.reshape(-1)), hex16,
                 data_section=False)
    s += "// expected output, vle baseline (fused vfmacc.vf, ascending k), fp16 bits\n"
    s += c_array("attn_expected_fused_bits", "const uint16_t",
                 f16_bits(expected_fused.reshape(-1)), hex16, data_section=False)
    s += ("static __fp16 attn_out[{}] __attribute__((section(\".data\"), aligned(128)));\n"
          ).format(NQ * HD)
    path = out_dir / "data_{}.h".format(cfg)
    path.write_text(s)
    print("wrote data_{}.h  (NQ={} TOPK={} HD={} NTOK={}, |out|max={:.3f}, header {:.1f} MB)".format(
        cfg, NQ, TOPK, HD, NTOK, float(np.abs(expected.astype(np.float64)).max()),
        path.stat().st_size / 1e6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq", type=int, nargs="+", default=[2, 16])
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--hd", type=int, default=128)
    ap.add_argument("--ntok", type=int, default=16384)
    args = ap.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for nq in args.nq:
        emit("nq{}".format(nq), nq, args.topk, args.hd, args.ntok, out_dir)
