#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# radix-scatter data generator: emits data/data_n<NREC>.h
#   paper config (BENCHMARK_PLAN 2026-09-07): NREC=65536 16-B records
#   (RD=4 x e32), FANOUT=256 partitions (8 radix bits, the SIGMOD'15
#   regime), 1 MiB src + 1 MiB dst; n8192 is the quick development config.
#
# A REAL radix partition of random keys: bucket = key >> (32 - 8), the
# histogram + exclusive prefix sum give the bucket base offsets, and each
# record's slot is its bucket base plus its rank among the records of that
# bucket in input order (the two-pass formulation with precomputed
# offsets; the offset computation is untimed setup for both arms). The
# slot array is a permutation with sequential writes inside each bucket -
# the locality structure of the real algorithm.
#
# Every array the kernels read is emitted literally (no on-core data
# generation): rs_src (record word 0 = key, words 1..3 random payload),
# rs_srcT (column-major staging of the same payload, baseline arm only:
# strided vector loads corrupt one lane under misses, erratum #2),
# rs_slot (u16, +64 zero padding), rs_dst zero. Check (exact, integer):
# rs_dst[rs_slot[r]*RD + d] == rs_src[r*RD + d] for every sampled record
# (every 4th + the last), so no expected array is needed.

import argparse
import pathlib
import numpy as np

RD = 4  # e32 elements per record (16-B records)


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


def radix_slots(keys, fanout_log2):
    bucket = (keys >> np.uint32(32 - fanout_log2)).astype(np.int64)
    fanout = 1 << fanout_log2
    hist = np.bincount(bucket, minlength=fanout)
    base = np.concatenate([[0], np.cumsum(hist)[:-1]])  # exclusive prefix sum
    order = np.argsort(bucket, kind="stable")           # input order within a bucket
    rank = np.empty(len(keys), dtype=np.int64)
    rank[order] = np.arange(len(keys)) - base[bucket[order]]
    return (base[bucket] + rank).astype(np.uint16), hist


def emit(nrec, fanout_log2, out_dir, seed=42):
    fanout = 1 << fanout_log2
    assert nrec <= 65536, "u16 slot ids"
    assert nrec % 128 == 0, "kernels run two 64-record chunks per loop iteration"
    rng = np.random.default_rng(seed)

    keys = rng.integers(0, 1 << 32, size=nrec, dtype=np.uint64).astype(np.uint32)
    slot, hist = radix_slots(keys, fanout_log2)
    assert len(np.unique(slot)) == nrec, "slots must be a permutation"

    src = np.empty((nrec, RD), dtype=np.uint32)
    src[:, 0] = keys
    src[:, 1:] = rng.integers(0, 1 << 32, size=(nrec, RD - 1), dtype=np.uint64).astype(np.uint32)
    srcT = src.T.reshape(-1)

    # validate the check formula on the full image
    image = np.zeros((nrec, RD), dtype=np.uint32)
    image[slot] = src
    assert (image[slot] == src).all()

    cfg = "n{}".format(nrec)
    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by radix-scatter/script/gen_data.py\n"
         "// config: {} (NREC={}, RD={}, FANOUT={}; bucket sizes {}..{})\n"
         "// Check: rs_dst[rs_slot[r]*RD + d] == rs_src[r*RD + d], exact integers.\n\n"
         ).format(cfg, nrec, RD, fanout, hist.min(), hist.max())
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NREC;        // records (16 B each = RD x e32)\n"
          "  unsigned int RD;          // e32 elements per record\n"
          "  unsigned int FANOUT_LOG2; // radix bits; FANOUT = 1 << FANOUT_LOG2\n"
          "} radix_layer;\n\n")
    s += ("const radix_layer rs_l = {{.NREC = {}, .RD = {}, .FANOUT_LOG2 = {}}};\n\n"
          ).format(nrec, RD, fanout_log2)
    s += "// source records, row-major: word 0 = key, words 1..3 payload\n"
    s += c_array("rs_src", "uint32_t", src.reshape(-1), hex32, align=128)
    s += "// destination image, zero\n"
    s += ("static uint32_t rs_dst[{}] __attribute__((section(\".data\"), aligned(128)));\n\n"
          ).format(nrec * RD)
    s += "// partition slot of each record (+64 zero padding)\n"
    s += c_array("rs_slot", "uint16_t", np.concatenate([slot, np.zeros(64, dtype=np.uint16)]),
                 "{}".format, align=64)
    s += "// column-major staging of the same records (baseline arm only)\n"
    s += c_array("rs_srcT", "uint32_t", srcT, hex32, align=128)
    path = out_dir / "data_{}.h".format(cfg)
    path.write_text(s)
    print("wrote data_{}.h  (NREC={} RD={} FANOUT={}, dst={} KiB, buckets {}..{} records, header {:.1f} MB)".format(
        cfg, nrec, RD, fanout, nrec * RD * 4 // 1024, hist.min(), hist.max(), path.stat().st_size / 1e6))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nrec", type=int, nargs="+", default=[8192, 65536],
                   help="record counts to emit (one header each)")
    p.add_argument("--fanout-log2", type=int, default=8)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for nrec in args.nrec:
        emit(nrec, args.fanout_log2, out_dir)
