#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# slsint8 data generator: emits data/data_n<NROWS>.h for the int8 deployed
# SparseLengthsSum config (paper: sls/int8), split layout:
#   table : NROWS x ROW_D u8 rows (32-B blocks)  -> 2 MiB, filled ON-CORE
#   sb    : NROWS x (fp16 scale, fp16 bias)      -> 256 KiB, filled ON-CORE
#   idx   : NB x LP u16 row ids                  -> literal (seed 42)
# out_b[d] = f32( sum_l  s[id_bl] * table[id_bl, d] ) + sum_l bias[id_bl]
#
# The big arrays are NOT emitted literally: the core fills them from the
# closed-form patterns below with the vector helper in include/bench_fill.h
# (a scalar head, then doubling vector copies); this script reproduces both
# fills bit-exactly. All scale/bias values are exact dyadic rationals
# (multiples of 2^-10 / 2^-8), so the on-core fp16 arrays and this script
# agree bit-for-bit, and u8 -> f32 conversion is exact.

import argparse
import pathlib
import numpy as np

HEAD = 3904     # table fill head, BYTES (= BF_HEAD_BYTES = 122 rows of 32 B)
SB_HEAD = 5040  # scale/bias fill head, ROWS: a multiple of the 315-row
                # (lcm(15, 63)) pattern period and of 16 rows (= 64 B, the
                # copy alignment bench_fill_rep needs), so the tiled array
                # equals the closed form for every row of the table.


def table(nrows, row_d):
    # core: t[i] = (i * 37) & 255 for i < HEAD, then bench_fill_rep(period HEAD)
    i = np.arange(HEAD, dtype=np.int64)
    head = ((i * 37) & 255).astype(np.uint8)
    n = nrows * row_d
    if n <= HEAD:
        return head[:n].reshape(nrows, row_d)
    reps = (n + HEAD - 1) // HEAD
    return np.tile(head, reps)[:n].reshape(nrows, row_d)


def scale_bias(nrows):
    # core: rows r < SB_HEAD written from the closed form, then tiled with
    # period SB_HEAD rows; SB_HEAD % 315 == 0 makes the tiling the identity
    # on the closed form, so evaluate it directly for every row.
    assert SB_HEAD % 315 == 0 and SB_HEAD % 16 == 0
    r = np.arange(nrows, dtype=np.int64)
    scale = ((1 + (r % 15)) / 1024.0).astype(np.float16)   # [2^-10, 15 * 2^-10]
    bias = (((r % 63) - 31) / 256.0).astype(np.float16)    # [-31/256, 31/256]
    return scale, bias


def c_float(v):
    # float literal that is always valid C: "{:.9g}" alone yields "1368" for
    # integral values, and "1368f" is not a number.
    s = "{:.9g}".format(v)
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return s + "f"


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


def emit(cfg, NB, LP, ROW_D, NROWS, out_dir):
    rng = np.random.default_rng(42)
    idx = rng.integers(0, NROWS, size=NB * LP, dtype=np.int64)
    tbl = table(NROWS, ROW_D).astype(np.float64)
    scale, bias = scale_bias(NROWS)
    s64 = scale.astype(np.float64)
    b64 = bias.astype(np.float64)
    ids = idx.reshape(NB, LP)

    # Bit-exact emulation of the kernel's fp32 arithmetic, per output
    # element, in kernel order (ascending l within a bag). Both arms run
    # the same op sequence per lookup:
    #   v12 = f32(u8 row)                        exact (<= 255)
    #   acc = f32(fs * v12 + acc)                vfmacc.vf = one rounding
    #   bias = f32(bias + fb)                    scalar C add
    # and per bag  out = f32(acc + bias)         vfadd.vf
    # A float64 op followed by a cast to float32 is exactly the RNE-rounded
    # fp32 result (the fs * row product has <= 19 significant bits and the
    # exponent gap to acc is < 29 bits, so the float64 FMA is itself exact).
    acc = np.zeros((NB, ROW_D), dtype=np.float32)
    bsum = np.zeros(NB, dtype=np.float32)
    for l in range(LP):
        id_l = ids[:, l]
        rows = tbl[id_l]                                   # (NB, ROW_D) exact ints
        fs = s64[id_l][:, None]
        acc = (fs * rows + acc.astype(np.float64)).astype(np.float32)       # vfmacc.vf
        bsum = (bsum.astype(np.float64) + b64[id_l]).astype(np.float32)     # bias += fb
    out = (acc.astype(np.float64) + bsum.astype(np.float64)[:, None]).astype(np.float32)  # vfadd.vf

    # Per-bag checksums: float64 sum of the ROW_D fp32 outputs. The core
    # sums each bag in f32 (error ~ ROW_D * 2^-24 relative) and compares
    # with tolerance 1e-4 * |chk| + 1e-3. Every bag is verified.
    checksum = out.astype(np.float64).sum(axis=1)

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by slsint8/script/gen_data.py\n"
         "// config: {}\n\n".format(cfg))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NB;      // bags\n"
          "  unsigned int LP;      // lookups per bag\n"
          "  unsigned int ROW_D;   // u8 elements per table row (32 -> 32-B blocks)\n"
          "  unsigned int NROWS;   // table rows (u16 id ceiling)\n"
          "  unsigned int HEAD;    // table fill head (bytes)\n"
          "  unsigned int SB_HEAD; // scale/bias fill head (rows)\n"
          "} slsint8_layer;\n\n")
    s += ("const slsint8_layer sl_l = {{.NB = {}, .LP = {}, .ROW_D = {}, .NROWS = {}, "
          ".HEAD = {}, .SB_HEAD = {}}};\n\n").format(NB, LP, ROW_D, NROWS, HEAD, SB_HEAD)
    s += c_array("sl_idx", "uint16_t", idx, "{}")
    s += c_array("sl_checksum", "const float", [c_float(v) for v in checksum], "{}",
                 data_section=False)
    # table, scale/bias and output buffers: sized here, filled on-core
    s += ("static uint8_t sl_tbl[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static __fp16 sl_sb[{1}] __attribute__((section(\".data\"), aligned(64)));\n"
          "static float sl_out[{2}] __attribute__((section(\".data\"), aligned(128)));\n"
          ).format(NROWS * ROW_D, NROWS * 2, NB * ROW_D)
    (out_dir / "data_{}.h".format(cfg)).write_text(s)
    print("wrote data_{}.h  (NB={} LP={} ROW_D={} NROWS={}, |checksum| in [{:.3f}, {:.3f}])".format(
        cfg, NB, LP, ROW_D, NROWS, np.abs(checksum).min(), np.abs(checksum).max()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--NB", type=int, default=2048)
    p.add_argument("--LP", type=int, default=40)
    p.add_argument("--ROW_D", type=int, default=32)
    p.add_argument("--NROWS", type=int, default=65536)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    emit("n{}".format(args.NROWS), args.NB, args.LP, args.ROW_D, args.NROWS, out_dir)
