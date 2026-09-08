#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# gatheragg data generator: emits data/data_<cfg>.h for the paper configs
#   d32_n16384     : sls-fp32 control (DLRM SparseLengthsSum, Gupta et al.
#                    HPCA'20): NB=2048 bags x LP=40 rows of 32 f32, 16384-row
#                    table (2 MiB)
#   d64_n16384     : gnnagg (GraphSAGE fanout S1=25): NB=2048 nodes x LP=25
#                    neighbours of 64 f32, 16384-row table (4 MiB)
#   d32_n16384_dbg : d32_n16384 with DBG_EVERY=32 (bug-A hang locator, prints
#                    "DBGG <bag>" every 32 bags; never used for measurement)
# out[b, :] = sum_l T[idx[b, l], :]
#
# The table is NOT emitted literally (2 / 4 MiB): the core fills it from a
# closed-form exact-dyadic pattern (head of HEAD f32, tiled) with the vector
# helper in include/bench_fill.h; this script reproduces that fill exactly.
# The pattern is a centred RAMP, t[i] = (i - 488) / 1024 for i < HEAD: the
# per-bag checksum verifier needs the distinct table rows to have pairwise
# well-separated sums, and hash-like fills (e.g. (37*i & 1023) - 512) give
# rows with EXACTLY equal sums (gap 0). With the ramp every row is a
# 16-aligned window of the period, so row sums step by >= 0.25 (ratio to the
# max tolerance ~13x at d32, ~8x at d64; asserted below).
# The indices (NB x LP u16, seed 42) ARE emitted literally, in the exact
# TRANSPOSED memory order the kernels read: within each group of
# UPG = 128 / ROW_D destinations, round-major -> position
# (b // UPG) * LP * UPG + l * UPG + (b % UPG) holds idx[b, l].
# Expected output = per-bag checksums (float64 sum over the row of the
# bit-exact fp32 emulation of the kernel's ascending-l accumulation).

import argparse
import math
import pathlib
import numpy as np

HEAD = 976  # f32 elements in the fill head = BF_HEAD_BYTES / 4


def table(n):
    i = np.arange(HEAD, dtype=np.int64)
    head = ((i - 488) / 1024.0).astype(np.float32)  # exact: |N| < 2^24, /2^10
    if n <= HEAD:
        return head[:n]
    reps = (n + HEAD - 1) // HEAD
    return np.tile(head, reps)[:n]


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays (checksums) must not carry the .data section attribute:
    # a const object in .data is a section type conflict.
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


def emit(cfg, NB, ROW_D, LP, NROWS, dbg_every, out_dir):
    UPG = 128 // ROW_D
    assert 128 % ROW_D == 0, "ROW_D must divide the 128-e32 m4 group"
    assert NB % UPG == 0, "NB must be a multiple of the units-per-group width"
    assert NROWS <= 65536, "u16 row ids"

    rng = np.random.default_rng(42)
    idx = rng.integers(0, NROWS, size=(NB, LP), dtype=np.int64)  # idx[b, l]
    # transposed memory order: [group][round][unit]
    idx_mem = idx.reshape(NB // UPG, UPG, LP).transpose(0, 2, 1).reshape(-1)
    assert idx_mem[(3 // UPG) * LP * UPG + 5 * UPG + (3 % UPG)] == idx[3, 5]

    T = table(NROWS * ROW_D).reshape(NROWS, ROW_D)
    T64 = T.astype(np.float64)

    # Bit-exact emulation of the kernels' fp32 arithmetic. Each output lane
    # accumulates independently in ascending l, and both arms execute the
    # same op per l: acc = f32(acc + row) (vfadd.vv, RNE). A float64 add of
    # two fp32 values is exact, so the cast back to float32 is exactly the
    # hardware result.
    acc = np.zeros((NB, ROW_D), dtype=np.float32)
    for l in range(LP):
        acc = (acc.astype(np.float64) + T64[idx[:, l]]).astype(np.float32)
    chk = acc.astype(np.float64).sum(axis=1)  # per-bag checksum (float64)

    # Detection margin of the checksum verifier: the fill period is HEAD
    # elements, so the table has only `period` distinct rows (a wrong index
    # differing by a multiple of `period` is indistinguishable by ANY
    # verifier). Any other wrong row shifts the bag checksum by at least
    # min_gap (pairwise gap of the distinct row sums), which must clear the
    # on-core tolerance 1e-4 * |chk| + 1e-3.
    period = HEAD // math.gcd(HEAD, ROW_D)
    rowsum = T64[:period].sum(axis=1)
    gaps = np.abs(rowsum[:, None] - rowsum[None, :])[~np.eye(period, dtype=bool)]
    min_gap = gaps.min()
    tol_max = 1e-4 * np.abs(chk).max() + 1e-3
    assert min_gap > 4 * tol_max, "row sums too close for the checksum verifier"

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by gatheragg/script/gen_data.py\n"
         "// config: {}\n"
         "// checksum verifier margin: {} distinct table rows, min row-sum gap\n"
         "// {:.4f} vs max tolerance {:.4f}\n\n".format(cfg, period, min_gap, tol_max))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NB;        // destinations (bags / nodes)\n"
          "  unsigned int ROW_D;     // row length (f32): 32 = sls-fp32, 64 = gnnagg\n"
          "  unsigned int LP;        // rows pooled per destination\n"
          "  unsigned int NROWS;     // table rows (NROWS x ROW_D f32, filled on-core)\n"
          "  unsigned int HEAD;      // table fill head length (elements)\n"
          "  unsigned int DBG_EVERY; // vlxblk arm: print \"DBGG <b>\" every DBG_EVERY bags (0 = off)\n"
          "} gatheragg_layer;\n\n")
    s += ("const gatheragg_layer ga_l = {{.NB = {}, .ROW_D = {}, .LP = {}, .NROWS = {}, "
          ".HEAD = {}, .DBG_EVERY = {}}};\n\n").format(NB, ROW_D, LP, NROWS, HEAD, dbg_every)
    s += "// row ids, TRANSPOSED: [group][round][unit], group = NB / (128 / ROW_D)\n"
    s += c_array("ga_idx", "uint16_t", idx_mem, "{}")
    s += "// per-bag checksums: float64 sum over the row of the fp32 emulation\n"
    s += c_array("ga_chk", "const float", chk, "{:.9g}f", data_section=False)
    # table + output buffers: sized here, filled on-core
    s += ("static float ga_tbl[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static float ga_out[{1}] __attribute__((section(\".data\"), aligned(128)));\n"
          ).format(NROWS * ROW_D, NB * ROW_D)
    (out_dir / "data_{}.h".format(cfg)).write_text(s)
    print("wrote data_{}.h  (NB={} ROW_D={} LP={} NROWS={} dbg_every={}, "
          "|chk|max={:.3f}, {} distinct rows, min row-sum gap {:.4f})".format(
              cfg, NB, ROW_D, LP, NROWS, dbg_every, np.abs(chk).max(), period, min_gap))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--NB", type=int, default=2048)
    p.add_argument("--NROWS", type=int, default=16384)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    n = args.NROWS
    emit("d32_n{}".format(n), args.NB, ROW_D=32, LP=40, NROWS=n, dbg_every=0, out_dir=out_dir)
    emit("d64_n{}".format(n), args.NB, ROW_D=64, LP=25, NROWS=n, dbg_every=0, out_dir=out_dir)
    emit("d32_n{}_dbg".format(n), args.NB, ROW_D=32, LP=40, NROWS=n, dbg_every=32, out_dir=out_dir)
