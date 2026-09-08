#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# pr-gather data generator: emits data/data_n<NV>.h for the paper config
#   n65536 : NV = 65,536 vertices (512 KiB fp64 contrib table > L1),
#            NACT = 4,096 destination vertices x DEG = 16 neighbors
# out[v] = base + damp * sum_{e < DEG} contrib[nbr[v*DEG + e]],
# base = 0.15 / NV, damp = 0.85 (fp64).
#
# Both big tables are filled on-core and reproduced here bit-exactly:
# - contrib (NV x f64): exact dyadic head of HEAD = BF_HEAD_BYTES/8 = 488
#   elements, contrib[i] = (((i*211) & 2047) + 1) / 2^16, tiled by the
#   vector helper in include/bench_fill.h (period HEAD).
# - nbr (NACT*DEG x u16): the vectorized u16 multiplicative hash
#   nbr[i] = ((i * 0x9E3779B1) mod 2^16) & (NV-1) - the core runs
#   vadd.vx/vmul.vx/vand.vx at e16, so only the low 16 bits of the product
#   survive; emulated here in uint64 then masked. (A literal 65,536-entry
#   array would be ~450 KB of header text.)
# The header carries the pr_layer parameter struct, base/damp, the 128-lane
# hash seed vector and the EXPECTED output (float64, sequential-order sum;
# with dyadic contributions the sum is exact in any association, the only
# rounding is in base + damp*s).

import argparse
import pathlib
import numpy as np

HEAD = 488               # f64 elements in the fill head = BF_HEAD_BYTES / 8
SEED_N = 128             # u16 lanes per hash step (e16 m4 on VLEN=512)
HASH_MULT = 2654435761   # 0x9E3779B1
DAMP = 0.85


def contrib_table(nv):
    i = np.arange(HEAD, dtype=np.uint64)
    head = (((i * 211) & 2047) + 1).astype(np.float64) / 65536.0
    if nv <= HEAD:
        return head[:nv]
    reps = (nv + HEAD - 1) // HEAD
    return np.tile(head, reps)[:nv]


def nbr_table(nv, n):
    # u16 lanes: (seed[j] + c) mod 2^16 = i, then the low 16 bits of i*mult
    i = np.arange(n, dtype=np.uint64) & 0xFFFF
    return (((i * HASH_MULT) & 0xFFFF) & (nv - 1)).astype(np.uint16)


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays (expected output) must not carry the .data section
    # attribute: a const object in .data is a section type conflict.
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


def emit(NV, NACT, DEG, out_dir):
    assert NV & (NV - 1) == 0, "NV must be a power of two (masked id generation)"
    assert DEG <= 32, "DEG > 32 exceeds one vsetvli group (e64 m4 / e16 m1); strip-mine"
    assert (NACT * DEG) % SEED_N == 0, "NACT*DEG must be a multiple of the 128-lane hash step"
    assert NACT * DEG <= 65536, "hash lane (seed + c) must not wrap past u16 for a bijective id stream"
    cfg = "n{}".format(NV)

    contrib = contrib_table(NV)
    nbr = nbr_table(NV, NACT * DEG).reshape(NACT, DEG)
    base = 0.15 / NV  # == the fp64 division the old kernel did on-core

    # Sequential-order float64 sum per vertex (the scalar arm's order and
    # vfredosum's ordered semantics). Every contrib is a multiple of 2^-16
    # and 16 of them sum below 1, so s is exact regardless of association;
    # base + damp*s is then two fp64 roundings (a contracted fmadd on the
    # core differs by <= 1 ulp, far inside the 1e-9 verifier tolerance).
    s = np.zeros(NACT, dtype=np.float64)
    for e in range(DEG):
        s = s + contrib[nbr[:, e]]
    expected = base + DAMP * s

    seed = np.arange(SEED_N, dtype=np.uint16)

    hdr = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
           "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
           "// SPDX-License-Identifier: Apache-2.0\n\n"
           "// This file was generated automatically by pr-gather/script/gen_data.py\n"
           "// config: {}\n\n".format(cfg))
    hdr += "#include <stdint.h>\n\n"
    hdr += ("typedef struct {\n"
            "  unsigned int NV;   // vertices (contrib table NV * 8 B)\n"
            "  unsigned int NACT; // destination vertices processed (timed tile)\n"
            "  unsigned int DEG;  // uniform in-degree (GAP/Graph500 edgefactor 16)\n"
            "  unsigned int HEAD; // contrib fill head length (f64 elements)\n"
            "} pr_layer;\n\n")
    hdr += ("const pr_layer pr_l = {{.NV = {}, .NACT = {}, .DEG = {}, .HEAD = {}}};\n\n"
            ).format(NV, NACT, DEG, HEAD)
    hdr += "const double pr_base = {!r}; // 0.15 / NV\n".format(base)
    hdr += "const double pr_damp = {!r};\n\n".format(DAMP)
    hdr += "// hash seed lanes 0..127 for the on-core neighbor-id generation\n"
    hdr += c_array("pr_seed16", "uint16_t", seed, "{}".format)
    hdr += c_array("pr_expected", "const double", expected, repr, data_section=False)
    # contrib / nbr / output buffers: sized here, filled on-core
    hdr += ("static double pr_contrib[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
            "static uint16_t pr_nbr[{1}] __attribute__((section(\".data\"), aligned(64)));\n"
            "static double pr_out[{2}] __attribute__((section(\".data\"), aligned(64)));\n"
            ).format(NV, NACT * DEG, NACT)
    (out_dir / "data_{}.h".format(cfg)).write_text(hdr)
    print("wrote data_{}.h  (NV={} NACT={} DEG={} HEAD={}, |expected| in [{:.6g}, {:.6g}])".format(
        cfg, NV, NACT, DEG, HEAD, expected.min(), expected.max()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--NV", type=int, default=65536)
    p.add_argument("--NACT", type=int, default=4096)
    p.add_argument("--DEG", type=int, default=16)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    emit(args.NV, args.NACT, args.DEG, out_dir)
