#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# nbforce data generator: emits data/data_adh.h for the paper config
#   adh : NC_DOM = 23,750 j-clusters (ADH, 95k atoms / 4) resident as four
#         parallel X4 fp32 field arrays; timed tile of NC_TILE = 256
#         i-clusters; LIST = 96 j-clusters per i-cluster drawn from a
#         +-WINDOW = 256 id locality window (spatial-sort model); cut2 = 2.0.
#
# The field arrays (4 x 380 KiB) are NOT emitted literally: the core fills
# them from the closed-form pattern (head of HEAD f32, tiled) with the
# vector helper in include/bench_fill.h; this script reproduces that fill
# exactly (all values are exact multiples of 2^-8, so float32 == float64).
# The u16 pair list (48 KiB) is emitted literally; the rvv arm expands it
# on-core (untimed) into the per-element u32 index array. The expected
# forces (NC_TILE x 4 atoms x 3 components) are the mdg_iatom_body formula
# evaluated in float64.

import argparse
import pathlib
import numpy as np

HEAD = 976  # f32 elements in the fill head = BF_HEAD_BYTES / 4


def field(n, mult):
    i = np.arange(HEAD, dtype=np.uint64)
    head = (((i * np.uint64(mult)) & np.uint64(1023)).astype(np.float64)
            / 256.0).astype(np.float32)
    if n <= HEAD:
        return head[:n]
    reps = (n + HEAD - 1) // HEAD
    return np.tile(head, reps)[:n]


def pairlist(nc_dom, nc_tile, lst, window):
    # Mirrors the original on-core rule: 9-bit hash of the pair slot,
    # offset in [-WINDOW, -1] u [1, WINDOW], j = c + off wrapped at 0.
    c = np.repeat(np.arange(nc_tile, dtype=np.int64), lst)
    slot = np.arange(nc_tile * lst, dtype=np.uint64)
    h = ((slot * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(23)
    off = (h & np.uint64(2 * window - 1)).astype(np.int64) - window
    off = np.where(off >= 0, off + 1, off)
    j = c + off
    j = np.where(j < 0, j + nc_dom, j)
    assert (j >= 0).all() and (j < nc_dom).all() and (j != c).all()
    assert j.max() < 65536
    return j.astype(np.uint16)


def expected_forces(X, Y, Z, Q, plist, nc_tile, lst, cut2):
    # mdg_iatom_body in float64 for every (i-cluster, i-atom, component):
    #   d = i - j; r2 = dx^2 + dy^2 + dz^2; w = max(cut2 - r2, 0);
    #   s = w^2 * qj; f += s * d   (summed over all LIST*4 j-atoms)
    X4 = X.astype(np.float64).reshape(-1, 4)
    Y4 = Y.astype(np.float64).reshape(-1, 4)
    Z4 = Z.astype(np.float64).reshape(-1, 4)
    Q4 = Q.astype(np.float64).reshape(-1, 4)
    L = plist.reshape(nc_tile, lst).astype(np.int64)
    xj = X4[L].reshape(nc_tile, lst * 4)  # (T, LIST*4)
    yj = Y4[L].reshape(nc_tile, lst * 4)
    zj = Z4[L].reshape(nc_tile, lst * 4)
    qj = Q4[L].reshape(nc_tile, lst * 4)
    dx = X4[:nc_tile][:, :, None] - xj[:, None, :]  # (T, 4, LIST*4)
    dy = Y4[:nc_tile][:, :, None] - yj[:, None, :]
    dz = Z4[:nc_tile][:, :, None] - zj[:, None, :]
    r2 = dx * dx + dy * dy + dz * dz
    w = np.maximum(cut2 - r2, 0.0)
    s = w * w * qj[:, None, :]
    fx = (s * dx).sum(axis=-1)  # (T, 4)
    fy = (s * dy).sum(axis=-1)
    fz = (s * dz).sum(axis=-1)
    # index c*12 + 3*a + d
    return np.stack([fx, fy, fz], axis=-1).reshape(-1), (w > 0).sum()


def c_float(v):
    # float literal that is always valid C: "{:.9g}" prints integral values
    # without a decimal point ("2"), and "2f" is not a C constant.
    s = "{:.9g}".format(v)
    if "." not in s and "e" not in s and "n" not in s:
        s += ".0"
    return s + "f"


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays (expected output) must not carry the .data section
    # attribute: a const object in .data is a section type conflict.
    attr = ("__attribute__((section(\".data\"), aligned({})))".format(align)
            if data_section else "__attribute__((aligned({})))".format(align))
    out = "static {} {}[{}] {} = {{\n".format(ctype, name, len(vals), attr)
    line = "   "
    for v in vals:
        s = " " + (fmt(v) if callable(fmt) else fmt.format(v)) + ","
        if len(line) + len(s) > 78:
            out += line + "\n"
            line = "   "
        line += s
    out += line.rstrip(",") + "};\n\n"
    return out


def emit(cfg, nc_dom, nc_tile, lst, window, cut2, out_dir):
    n_atoms = nc_dom * 4
    X = field(n_atoms, 37)
    Y = field(n_atoms, 53)
    Z = field(n_atoms, 71)
    Q = field(n_atoms, 89)
    plist = pairlist(nc_dom, nc_tile, lst, window)
    expected, n_in_cut = expected_forces(X, Y, Z, Q, plist, nc_tile, lst, cut2)

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by nbforce/script/gen_data.py\n"
         "// config: {}\n\n".format(cfg))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NC_DOM;  // domain j-clusters (ADH: 95,000 atoms / 4)\n"
          "  unsigned int NC_TILE; // timed i-clusters (T in the paper table)\n"
          "  unsigned int LIST;    // j-clusters per i-cluster (P; Pall'13-derived)\n"
          "  unsigned int WINDOW;  // pair-list locality window (+-ids)\n"
          "  unsigned int HEAD;    // field fill head length (f32 elements)\n"
          "  float CUT2;           // cutoff^2 of the simplified pair force\n"
          "} nbforce_layer;\n\n")
    s += ("const nbforce_layer nb_l = {{.NC_DOM = {}, .NC_TILE = {}, .LIST = {}, "
          ".WINDOW = {}, .HEAD = {}, .CUT2 = {}}};\n\n").format(
              nc_dom, nc_tile, lst, window, HEAD, c_float(cut2))
    s += "// pair list: LIST u16 j-cluster ids per i-cluster (row c = pairs of i-cluster c)\n"
    s += c_array("nb_list", "uint16_t", plist, "{}")
    s += ("// expected forces, fo[c*12 + 3*a + d] (float64 reference of "
          "mdg_iatom_body)\n")
    s += c_array("nb_expected", "const float", expected, c_float, data_section=False)
    # field arrays + output buffer: sized here, filled on-core
    s += ("// field arrays (x, y, z, q): filled on-core, see main-*.c fill_field\n"
          "static float nb_x[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static float nb_y[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static float nb_z[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static float nb_q[{0}] __attribute__((section(\".data\"), aligned(128)));\n"
          "static float nb_f[{1}] __attribute__((section(\".data\"), aligned(64)));\n"
          ).format(n_atoms, nc_tile * 12)
    (out_dir / "data_{}.h".format(cfg)).write_text(s)
    print("wrote data_{}.h  (NC_DOM={} NC_TILE={} LIST={} WINDOW={} cut2={}, "
          "|expected|max={:.3f}, pairs in cutoff={}/{})".format(
              cfg, nc_dom, nc_tile, lst, window, cut2, np.abs(expected).max(),
              n_in_cut, nc_tile * 4 * lst * 4))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nc_dom", type=int, default=23750)
    p.add_argument("--nc_tile", type=int, default=256)
    p.add_argument("--list", type=int, default=96)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--cut2", type=float, default=2.0)
    args = p.parse_args()
    np.random.seed(42)  # convention; the data is fully closed-form
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    emit("adh", args.nc_dom, args.nc_tile, args.list, args.window, args.cut2,
         out_dir)
