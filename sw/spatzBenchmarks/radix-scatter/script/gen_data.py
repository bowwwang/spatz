#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# radix-scatter data generator: emits data/data_n<NREC>.h for the paper
# config (BENCHMARK_PLAN 2026-09-07): NREC=65536 16-B records (RD=4 x e32),
# FANOUT=256 partitions (8 radix bits, the SIGMOD'15 regime), 1 MiB dst.
#
# Everything is closed-form and exact (integers), so nothing is random and
# the header stays small: it carries the layer struct and the two 128-word
# iota seeds of the on-core vector fills. The BIG arrays (src 1 MiB, dst
# 1 MiB, slots 128 KiB, baseline staging 1 MiB) are NOT emitted; main
# fills them on-core with vector loops that this script mirrors exactly:
#
#   slot[r] = (r & (F-1)) * (NREC/F) + r/F    radix partition of synthetic
#             keys that assign records round-robin to the F partitions
#             (bucket = r mod F); the histogram + prefix-sum offsets then
#             collapse to this closed form: a permutation with per-bucket
#             sequential runs of NREC/F records (the real locality
#             structure, 4 records per 64-B line).
#   src[i]  = i                                unique payload words
#   srcT[d * NREC + r] = 4r + d = src[r*4 + d] column-major staging of the
#             SAME payload for the vsoxei32 baseline (erratum #2: strided
#             vector loads corrupt one lane under cache misses).
#   dst     = 0
#
# Check (exact, integer): dst[slot(r) * RD + d] == r * RD + d for every
# sampled record r (every 4th + the last; sampling bounds the scalar-core
# reference cost, a structural scatter bug hits sampled records too). The
# checker recomputes slot(r) from the closed form in C, so no expected
# array is needed; this script validates the same formula on the full
# image.

import argparse
import pathlib
import numpy as np

RD = 4            # e32 elements per record (16-B records)
FILL_CHUNK = 128  # elements per on-core fill iteration (e16 m4 / e32 m8)
SEED_LEN = 128    # iota seeds of the vector fills


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays must not carry the .data section attribute: a const
    # object in .data is a section type conflict.
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


def slots_closed_form(nrec, fanout_log2):
    r = np.arange(nrec, dtype=np.uint32)
    fanout = 1 << fanout_log2
    return ((r & (fanout - 1)) * (nrec // fanout) + (r >> fanout_log2)).astype(np.uint16)


def slots_on_core(nrec, fanout_log2, seed16):
    # Mirror of main's vector loop, chunk by chunk in u16 arithmetic:
    #   r = seed + c; bucket = r & (F-1); slot = (bucket << (log2 NREC -
    #   log2 F)) | (r >> log2 F)
    nrec_log2 = nrec.bit_length() - 1
    fanout = 1 << fanout_log2
    out = np.empty(nrec, dtype=np.uint16)
    for c in range(0, nrec, FILL_CHUNK):
        r = (seed16.astype(np.uint32) + c).astype(np.uint16)
        bucket = r & np.uint16(fanout - 1)
        out[c:c + FILL_CHUNK] = ((bucket << np.uint16(nrec_log2 - fanout_log2))
                                 | (r >> np.uint16(fanout_log2))).astype(np.uint16)
    return out


def emit(nrec, fanout_log2, out_dir):
    fanout = 1 << fanout_log2
    assert nrec & (nrec - 1) == 0 and nrec >= fanout, \
        "NREC must be a power of two and >= FANOUT (closed-form slots)"
    assert nrec <= 65536, "u16 slot ids"
    assert nrec % FILL_CHUNK == 0, "on-core fills run in 128-element chunks"
    assert nrec % 64 == 0, "kernels run 64 records per chunk"

    seed16 = np.arange(SEED_LEN, dtype=np.uint16)
    seed32 = np.arange(SEED_LEN, dtype=np.uint32)

    slot = slots_closed_form(nrec, fanout_log2)
    assert (slots_on_core(nrec, fanout_log2, seed16) == slot).all()
    assert len(np.unique(slot)) == nrec, "slots must be a permutation"

    # on-core fills: src[i] = i (e32 m8 chunks of seed + c), srcT = 4r + d
    src = np.arange(nrec * RD, dtype=np.uint32)
    src_core = np.concatenate([seed32 + c for c in range(0, nrec * RD, FILL_CHUNK)])
    assert (src_core == src).all()
    srcT = np.concatenate([((seed32 + c) << 2) + d
                           for d in range(RD) for c in range(0, nrec, FILL_CHUNK)])
    assert (srcT == src.reshape(nrec, RD).T.reshape(-1)).all()

    # full expected image and the C checker's formula on the sample set
    expected = np.zeros((nrec, RD), dtype=np.uint32)
    expected[slot] = src.reshape(nrec, RD)
    sample = np.append(np.arange(0, nrec, 4), nrec - 1)
    exp_rows = (sample[:, None] * RD + np.arange(RD)[None, :]).astype(np.uint32)
    assert (expected[slot[sample]] == exp_rows).all()

    cfg = "n{}".format(nrec)
    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by radix-scatter/script/gen_data.py\n"
         "// config: {} (NREC={}, RD={}, FANOUT={})\n"
         "//\n"
         "// Data contract (all closed-form, exact integers):\n"
         "//   rs_slot[r] = (r & (F-1)) * (NREC/F) + r/F   generated ON-CORE by main\n"
         "//                (vector loop, mirrored in gen_data.py); a permutation\n"
         "//                with per-bucket sequential runs of NREC/F records\n"
         "//   rs_src[i]  = i                                filled on-core\n"
         "//   rs_srcT[d*NREC + r] = 4r + d                  baseline arm only, on-core\n"
         "//   rs_dst     = 0\n"
         "// Check: rs_dst[slot(r)*RD + d] == r*RD + d, exact, sampled every 4th\n"
         "// record + the last; the checker recomputes slot(r) from the closed form.\n\n"
         ).format(cfg, nrec, RD, fanout)
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NREC;        // records (16 B each = RD x e32)\n"
          "  unsigned int RD;          // e32 elements per record\n"
          "  unsigned int FANOUT_LOG2; // radix bits; FANOUT = 1 << FANOUT_LOG2\n"
          "} radix_layer;\n\n")
    s += ("const radix_layer rs_l = {{.NREC = {}, .RD = {}, .FANOUT_LOG2 = {}}};\n\n"
          ).format(nrec, RD, fanout_log2)
    # record buffers: sized here, filled on-core (declaration order kept from
    # the verified layout: src, dst, slot, then the seeds)
    s += ("static uint32_t rs_src[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static uint32_t rs_dst[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "// +16 elements of slack kept from the verified layout\n"
          "static uint16_t rs_slot[{1}] __attribute__((section(\".data\"), aligned(64)));\n\n"
          ).format(nrec * RD, nrec + 16)
    # iota seeds of the on-core vector fills
    s += c_array("rs_seed16", "uint16_t", seed16, "{}", align=64)
    s += c_array("rs_seed32", "uint32_t", seed32, "{}", align=128)
    (out_dir / "data_{}.h".format(cfg)).write_text(s)
    print("wrote data_{}.h  (NREC={} RD={} FANOUT={}, dst={} KiB, slots={} KiB)".format(
        cfg, nrec, RD, fanout, nrec * RD * 4 // 1024, nrec * 2 // 1024))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nrec", type=int, nargs="+", default=[65536],
                   help="record counts to emit (one header each)")
    p.add_argument("--fanout-log2", type=int, default=8)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for nrec in args.nrec:
        emit(nrec, args.fanout_log2, out_dir)
