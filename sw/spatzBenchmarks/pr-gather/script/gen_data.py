#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# pr-gather data generator: emits data/data_n<NV>_a<NACT>.h
#   NV   : vertices, contrib table NV x fp64 (paper: 65,536 -> 512 KiB > L1)
#   NACT : destination vertices processed in the timed tile (paper: 4,096;
#          512 is the quick development config), multiple of 16
#   DEG  : uniform in-degree 16 (GAP/Graph500 edgefactor)
# out[v] = base + damp * sum_{e < DEG} contrib[nbr[v*DEG + e]],
# base = 0.15 / NV, damp = 0.85 (fp64).
#
# Every array the kernel reads is emitted literally (no on-core data
# generation): contrib as uint64 bit patterns, the u16 neighbor ids, the
# 16-lane reduction seed vector (lane 0 = base, others 0) and the expected
# outputs as uint64 bit patterns. Expected values follow the vlxblk
# kernel's operation order: each contribution scaled by damp (fp64
# rounding), then the ordered sum seeded with base. Spatz's vfredosum
# associates internally, so the verifier compares in integer ULPs with a
# 2^20-ULP (~2.3e-10 relative) bound; a single wrong index perturbs a sum
# by >= ~1e-2 relative, far outside that bound.

import argparse
import pathlib
import numpy as np

DAMP = 0.85
SEED_LANES = 16  # one e64 register at VLEN=1024


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays must not carry the .data section attribute (a const
    # object in .data is a section type conflict).
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


def hex64(x):
    return "0x{:016x}".format(int(x))


def emit(NV, NACT, DEG, out_dir, seed=42):
    assert NV & (NV - 1) == 0 and NV <= 65536, "NV: power of two, ids must fit u16"
    assert DEG == 16, "kernel assumes deg 16 (one e64 register per neighborhood)"
    assert NACT % 16 == 0, "NACT must be a multiple of 16 (two 8-vertex pipeline rounds)"
    cfg = "n{}_a{}".format(NV, NACT)
    rng = np.random.default_rng(seed)

    # PageRank contributions rank/deg with rank ~ U(0.5, 1.5)/NV
    contrib = (rng.uniform(0.5, 1.5, NV) / NV / DEG).astype(np.float64)
    nbr = rng.integers(0, NV, size=(NACT, DEG)).astype(np.uint16)
    base = 0.15 / NV

    # kernel order: prod = fp64(damp * c), s = base; s = fp64(s + prod_e)
    prod = np.float64(DAMP) * contrib[nbr]  # (NACT, DEG), rounded per element
    s = np.full(NACT, base, dtype=np.float64)
    for e in range(DEG):
        s = s + prod[:, e]
    expected = s

    seedv = np.zeros(SEED_LANES, dtype=np.float64)
    seedv[0] = base

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
            "} pr_layer;\n\n")
    hdr += "const pr_layer pr_l = {{.NV = {}, .NACT = {}, .DEG = {}}};\n\n".format(NV, NACT, DEG)
    hdr += "const double pr_damp = {!r};\n\n".format(DAMP)
    hdr += "// reduction seed vector: lane 0 = base = 0.15 / NV, lanes 1..15 = 0\n"
    hdr += c_array("pr_seed", "double", seedv, repr, align=128)
    hdr += "// contributions, fp64 bit patterns (read as const double * by the kernels)\n"
    hdr += c_array("pr_contrib_bits", "uint64_t", contrib.view(np.uint64), hex64, align=128)
    hdr += "// neighbor ids, NACT x DEG, + 128 zero ids so a pipeline may issue\n"
    hdr += "// its last gathers past the end (they read contrib[0], unused)\n"
    nbr_pad = np.concatenate([nbr.reshape(-1), np.zeros(128, dtype=np.uint16)])
    hdr += c_array("pr_nbr", "uint16_t", nbr_pad, "{}".format)
    hdr += "// expected outputs, fp64 bit patterns\n"
    hdr += c_array("pr_expected_bits", "const uint64_t", expected.view(np.uint64), hex64,
                   data_section=False)
    hdr += ("static double pr_out[{}] __attribute__((section(\".data\"), aligned(64)));\n"
            ).format(NACT)
    path = out_dir / "data_{}.h".format(cfg)
    path.write_text(hdr)
    print("wrote data_{}.h  (NV={} NACT={} DEG={}, expected in [{:.4g}, {:.4g}], header {:.1f} MB)".format(
        cfg, NV, NACT, DEG, expected.min(), expected.max(), path.stat().st_size / 1e6))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--NV", type=int, default=65536)
    p.add_argument("--NACT", type=int, nargs="+", default=[512, 4096])
    p.add_argument("--DEG", type=int, default=16)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for nact in args.NACT:
        emit(args.NV, nact, args.DEG, out_dir)
