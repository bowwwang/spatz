#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# gatheragg (paper row "gnnagg") data generator: emits data/data_<cfg>.h
#   d64_nb2048 : GNN pull-mode neighbour aggregation (GraphSAGE fanout
#                S1 = 25; HyGCN): NB = 2048 nodes x LP = 25 neighbours, rows
#                of 64 fp32 (256 B), 16,384-row table (4 MiB)
#   d64_nb64   : same table, NB = 64 (quick development config)
# out[b, :] = sum_l T[idx[b, l], :]   (fp32, ascending l)
#
# Every array the kernels read is emitted literally (no on-core data
# generation): the table as fp32 bit patterns (uniform in [-0.5, 0.5)),
# the ids in the vlxblk arm's TRANSPOSED order (within each group of
# UPG = 4 nodes - one e32 m8 group holds 4 rows - round l's 4 ids are
# contiguous: position (b // UPG) * LP * UPG + l * UPG + (b % UPG) holds
# idx[b, l]; +16 zero ids of padding for the 32-B index loads that use 4),
# the same ids row-major for the vle baseline, and the EXPECTED OUTPUT as
# fp32 bit patterns: a bit-exact emulation of the kernels' lane-wise fp32
# accumulation (acc = f32(acc + row) in ascending l; a float64 add of two
# fp32 values is exact, so the cast back is exactly the hardware result),
# so the verifier compares bit patterns exactly.

import argparse
import pathlib
import numpy as np

ROW_D = 64
LP = 25
UPG = 256 // ROW_D  # nodes per e32 m8 group (256 lanes)


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


def hex32(x):
    return "0x{:08x}".format(int(x))


def f32_bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def emit(cfg, NB, NROWS, out_dir, seed=42):
    assert NB % UPG == 0, "NB must be a multiple of the nodes per group (4)"
    assert NROWS <= 65536, "u16 row ids"
    rng = np.random.default_rng(seed)

    T = rng.uniform(-0.5, 0.5, size=(NROWS, ROW_D)).astype(np.float32)
    idx = rng.integers(0, NROWS, size=(NB, LP), dtype=np.int64)  # idx[b, l]
    # vlxblk arm: transposed [group][round][unit] (+16 zero padding)
    idx_mem = idx.reshape(NB // UPG, UPG, LP).transpose(0, 2, 1).reshape(-1)
    assert idx_mem[(3 // UPG) * LP * UPG + 5 * UPG + (3 % UPG)] == idx[3, 5]
    idx_mem = np.concatenate([idx_mem, np.zeros(32, dtype=np.int64)]).astype(np.uint16)

    # bit-exact emulation of the lane-wise fp32 accumulation
    T64 = T.astype(np.float64)
    acc = T[idx[:, 0]].copy()  # first row: acc = row (vfmul by 1.0 / vmv)
    for l in range(1, LP):
        acc = (acc.astype(np.float64) + T64[idx[:, l]]).astype(np.float32)

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by gatheragg/script/gen_data.py\n"
         "// config: {} (NB={} ROW_D={} LP={} NROWS={}; table {} MiB)\n"
         "// Check: exact fp32 bit compare of out[b, :] against the emulation.\n\n"
         ).format(cfg, NB, ROW_D, LP, NROWS, NROWS * ROW_D * 4 // 2**20)
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NB;    // destinations (nodes)\n"
          "  unsigned int ROW_D; // row length (fp32): 64 = gnnagg (256-B rows)\n"
          "  unsigned int LP;    // rows pooled per destination (neighbours)\n"
          "  unsigned int NROWS; // table rows (NROWS x ROW_D fp32)\n"
          "} gatheragg_layer;\n\n")
    s += ("const gatheragg_layer ga_l = {{.NB = {}, .ROW_D = {}, .LP = {}, .NROWS = {}}};\n\n"
          ).format(NB, ROW_D, LP, NROWS)
    s += "// feature table, fp32 bit patterns (read as const float * by the kernels)\n"
    s += c_array("ga_tbl_bits", "uint32_t", f32_bits(T.reshape(-1)), hex32, align=128)
    s += "// row ids, vlxblk arm: TRANSPOSED [group][round][unit], UPG = 4, +16 padding\n"
    s += c_array("ga_idx", "uint16_t", idx_mem, "{}".format)
    s += "// row ids, vle baseline: row-major [node][round]\n"
    s += c_array("ga_idx_rows", "uint16_t", idx.reshape(-1).astype(np.uint16), "{}".format)
    s += "// expected output out[b*ROW_D + d], fp32 bit patterns (bit-exact emulation)\n"
    s += c_array("ga_expected_bits", "const uint32_t", f32_bits(acc.reshape(-1)), hex32,
                 data_section=False)
    s += ("static float ga_out[{}] __attribute__((section(\".data\"), aligned(128)));\n"
          ).format(NB * ROW_D)
    path = out_dir / "data_{}.h".format(cfg)
    path.write_text(s)
    print("wrote data_{}.h  (NB={} ROW_D={} LP={} NROWS={}, |out|max={:.3f}, header {:.1f} MB)".format(
        cfg, NB, ROW_D, LP, NROWS, np.abs(acc).max(), path.stat().st_size / 1e6))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--NB", type=int, nargs="+", default=[64, 2048])
    p.add_argument("--NROWS", type=int, default=16384)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for nb in args.NB:
        emit("d64_nb{}".format(nb), nb, args.NROWS, out_dir)
