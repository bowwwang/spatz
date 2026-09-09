#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# sls data generator: plain pooling (no scale/bias, no dequant), accumulate in
# the table dtype (no conversion). Paper points: sls-1 = fp16 rows of 16
# elements (32-B blocks), sls-2 = fp32 rows of 32 elements (128-B blocks).
# The kernel is the memory-access pattern the paper highlights:
#   out_b[d] = sum_l table[id_bl][d]   (fp16 in, fp16 out)
#   table : NROWS x ROW_D fp16                (literal)
#   idx   : NB x LP u16 row ids (seed 42)     (literal)
#   expected out[NB][8][ROW_D]: the 8 chunk-slot partial sums per bag (no
#   final reduction), fp16 in ascending chunk order — identical in both arms
# LP must be a multiple of 8 (8 rows per gather). Debug config: --NB 64.

import argparse
import pathlib
import numpy as np


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
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


def emit(cfg, NB, LP, ROW_D, NROWS, out_dir, dtype, dbg_every=0):
    assert LP % 8 == 0, "LP must be a multiple of 8 (8 rows per gather)"
    rng = np.random.default_rng(42)
    idx = rng.integers(0, NROWS, size=NB * LP, dtype=np.int64)
    # rows: exact dyadic values in [-0.5, 0.5), every row distinct; fp16 (sls-1,
    # 32-B rows) or fp32 (sls-2, 128-B rows)
    npt = np.float16 if dtype == "fp16" else np.float32
    r = np.arange(NROWS, dtype=np.int64)[:, None]
    d = np.arange(ROW_D, dtype=np.int64)[None, :]
    tbl = ((((r * 37 + d * 11 + (r >> 4)) & 1023) - 512) / 1024.0).astype(npt)
    ids = idx.reshape(NB, LP)
    t64 = tbl.astype(np.float64)
    def f16(x):                      # correctly rounded to the table dtype (RNE), as the hardware
        return x.astype(npt)
    # Output = the 8 chunk-slot partial sums per bag (no final reduction —
    # a real slide-based reduction stalled the VLSU; user ruling 2026-09-09):
    # out[b][i][d] = sum_c t[id_{b, 8c+i}][d], accumulated in fp16 in ascending
    # c. Both arms produce exactly this (vlxblk: 128-lane accumulator,
    # slot i = lanes 16i..; vle: 8 accumulators, row l -> acc l % 8), so one
    # bit-exact expected array serves both.
    rows = t64[ids]                                  # (NB, LP, ROW_D)
    nch = LP // 8
    acc = f16(rows[:, 0:8, :]).reshape(NB, 8 * ROW_D)          # chunk 0: vmv (init)
    for c in range(1, nch):
        acc = f16(acc.astype(np.float64) + rows[:, 8*c:8*c+8, :].reshape(NB, 8 * ROW_D))
    expected = acc.astype(np.float64).reshape(NB * 8 * ROW_D)

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by slsint8/script/gen_data.py\n"
         "// config: {}\n\n".format(cfg))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NB;        // bags\n"
          "  unsigned int LP;        // lookups per bag (multiple of 8)\n"
          "  unsigned int ROW_D;     // elements per table row (fp16: 16 -> 32 B; fp32: 32 -> 128 B)\n"
          "  unsigned int NROWS;     // table rows\n"
          "  unsigned int OUT_D;     // outputs per bag = 8 slots x ROW_D\n"
          "  unsigned int DBG_EVERY; // progress print every N bags (0 = off; -dbg target only)\n"
          "} sls_layer;\n\n")
    s += ("const sls_layer sl_l = {{.NB = {}, .LP = {}, .ROW_D = {}, .NROWS = {}, .OUT_D = {}, .DBG_EVERY = {}}};\n\n"
          ).format(NB, LP, ROW_D, NROWS, 8 * ROW_D, dbg_every)
    if dtype == "fp16":
        s += "// table[NROWS][ROW_D] fp16 (32-B rows) as raw bit patterns (exact, compact);\n"
        s += "// the main casts to const __fp16 *\n"
        s += c_array("sl_tbl_bits", "uint16_t", tbl.reshape(NROWS * ROW_D).view(np.uint16), "0x{:04x}", align=128)
    else:
        s += "// table[NROWS][ROW_D] fp32 (128-B rows) as raw bit patterns (exact, compact);\n"
        s += "// the main casts to const float *\n"
        s += c_array("sl_tbl_bits", "uint32_t", tbl.reshape(NROWS * ROW_D).view(np.uint32), "0x{:08x}", align=128)
    s += "// idx[NB][LP] u16 row ids (+16 zero padding: the vlxblk arm loads 16 ids per\n// chunk = 32 B, full bus width, and uses 8)\n"
    s += c_array("sl_idx", "uint16_t", np.concatenate([idx, np.zeros(16, dtype=np.int64)]), "{}")
    s += "// expected out[NB][8][ROW_D]: 8 chunk-slot partial sums per bag ({} results)\n".format(dtype)
    s += c_array("sl_expected", "const float", expected, "{:.9e}f", data_section=False)
    ctype = "__fp16" if dtype == "fp16" else "float"
    s += ("static {1} sl_out[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          ).format(NB * 8 * ROW_D, ctype)
    (out_dir / "data_{}.h".format(cfg)).write_text(s)
    print("wrote data_{}.h  (NB={} LP={} ROW_D={} NROWS={}, |expected| max {:.3f}, header {:.1f} MB)".format(
        cfg, NB, LP, ROW_D, NROWS, np.abs(expected).max(), len(s) / 1048576))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--NB", type=int, default=2048)
    p.add_argument("--LP", type=int, default=40)
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--ROW_D", type=int, default=0, help="default 16 (fp16) / 32 (fp32)")
    p.add_argument("--NROWS", type=int, nargs="+", default=[4096])
    p.add_argument("--dbg", action="store_true", help="also emit the _dbg header (DBG_EVERY = NB/8)")
    p.add_argument("--tag", default="", help="config-name suffix, e.g. _nb2048 for paper-size runs")
    args = p.parse_args()
    row_d = args.ROW_D or (16 if args.dtype == "fp16" else 32)
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for n in args.NROWS:
        emit("{}_n{}{}".format(args.dtype, n, args.tag), args.NB, args.LP, row_d, n, out_dir, args.dtype)
        if args.dbg:
            emit("{}_n{}_dbg".format(args.dtype, n), args.NB, args.LP, row_d, n, out_dir, args.dtype, dbg_every=max(1, args.NB // 8))
