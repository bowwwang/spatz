#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# nbforce data generator: emits data/data_<cfg>.h
#   adh     : NC_DOM = 23,750 j-clusters (ADH, 95k atoms / 4) resident as
#             four parallel X4 fp32 field arrays (x, y, z, q; 380 KiB each);
#             timed tile NC_TILE = 256 i-clusters; LIST = 96 j-clusters per
#             i-cluster drawn from a +-WINDOW = 256 id locality window
#             (spatial-sort model, MD ruling 2026-09-07); cut2 = 2.0.
#   adh_t32 : same domain, NC_TILE = 32 (quick development config).
#
# Geometry: cluster c sits in cell (c % 16, (c // 16) % 16, c // 256) of a
# unit-cell grid (spatially sorted ids: the +-256 window is about one
# z-layer), its 4 atoms jittered +-0.3 around the cell centre; charges
# q ~ U(0.5, 1.5). With cut2 = 2.0 only spatial neighbours interact, the
# rest is clamped to zero by max(cut2 - r2, 0) - the kernel's work is the
# same either way (FMA-only, no branches).
#
# Every array the kernels read is emitted literally (no on-core data
# generation): the four fields as fp32 bit patterns, the u16 pair list
# (+64 zero padding: the pipeline over-reads 32 ids per 64-B index load
# and prefetches one chunk past the end), the expanded per-element u32
# index array of the plain-RVV baseline, and the expected forces (float64
# reference of the pair-force formula) as fp32 bit patterns.
#
# Pair force (simplified, FMA-only, 17 FLOP/pair), per i-atom and j-atom:
#   d = ri - rj; r2 = dx^2 + dy^2 + dz^2; w = max(cut2 - r2, 0);
#   s = w^2 * qj; f_i += s * d    (summed over LIST*4 j-atoms)

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


def hex32(x):
    return "0x{:08x}".format(int(x))


def f32_bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


def geometry(nc_dom, rng):
    c = np.arange(nc_dom)
    cell = np.stack([c % 16, (c // 16) % 16, c // 256], axis=1).astype(np.float64)
    centre = cell + 0.5
    jit = rng.uniform(-0.3, 0.3, size=(nc_dom, 4, 3))
    pos = centre[:, None, :] + jit  # (NC, 4 atoms, xyz)
    X = pos[:, :, 0].reshape(-1).astype(np.float32)
    Y = pos[:, :, 1].reshape(-1).astype(np.float32)
    Z = pos[:, :, 2].reshape(-1).astype(np.float32)
    Q = rng.uniform(0.5, 1.5, size=nc_dom * 4).astype(np.float32)
    return X, Y, Z, Q


def pairlist(nc_dom, nc_tile, lst, window, rng):
    # j within +-window ids of i, j != i, wrapped into the domain
    c = np.repeat(np.arange(nc_tile, dtype=np.int64), lst)
    off = rng.integers(1, window + 1, size=nc_tile * lst) * rng.choice([-1, 1], size=nc_tile * lst)
    j = (c + off) % nc_dom
    assert (j != c).all() and j.max() < 65536
    return j.astype(np.uint16)


def expected_forces(X, Y, Z, Q, plist, nc_tile, lst, cut2):
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
    return np.stack([fx, fy, fz], axis=-1).reshape(-1), int((w > 0).sum())


def emit(cfg, nc_dom, nc_tile, lst, window, cut2, out_dir, seed=42):
    assert nc_dom <= 65536, "u16 cluster ids"
    assert lst % 16 == 0, "LIST must be a multiple of 16 (two 8-cluster chunks per loop iteration)"
    rng = np.random.default_rng(seed)
    X, Y, Z, Q = geometry(nc_dom, rng)
    plist = pairlist(nc_dom, nc_tile, lst, window, rng)
    expected, n_in_cut = expected_forces(X, Y, Z, Q, plist, nc_tile, lst, cut2)
    # baseline: per-element u32 indices (cluster*4 + lane)
    plist_exp = (plist.astype(np.uint32)[:, None] * 4 + np.arange(4, dtype=np.uint32)[None, :]).reshape(-1)

    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by nbforce/script/gen_data.py\n"
         "// config: {} (pairs inside the cutoff: {} of {})\n\n".format(
             cfg, n_in_cut, nc_tile * 4 * lst * 4))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int NC_DOM;    // domain j-clusters (ADH: 95,000 atoms / 4)\n"
          "  unsigned int NC_TILE;   // timed i-clusters (T in the paper table)\n"
          "  unsigned int LIST;      // j-clusters per i-cluster (P; Pall'13-derived)\n"
          "  unsigned int WINDOW;    // pair-list locality window (+-ids)\n"
          "  uint32_t     CUT2_BITS; // cutoff^2 of the pair force, fp32 bits\n"
          "} nbforce_layer;\n\n")
    s += ("const nbforce_layer nb_l = {{.NC_DOM = {}, .NC_TILE = {}, .LIST = {}, "
          ".WINDOW = {}, .CUT2_BITS = {}}};\n\n").format(
              nc_dom, nc_tile, lst, window, hex32(f32_bits([cut2])[0]))
    s += "// field arrays (X4 layout: 4 atoms per cluster), fp32 bit patterns\n"
    s += c_array("nb_x_bits", "uint32_t", f32_bits(X), hex32, align=128)
    s += c_array("nb_y_bits", "uint32_t", f32_bits(Y), hex32, align=128)
    s += c_array("nb_z_bits", "uint32_t", f32_bits(Z), hex32, align=128)
    s += c_array("nb_q_bits", "uint32_t", f32_bits(Q), hex32, align=128)
    s += "// pair list: LIST u16 j-cluster ids per i-cluster (+64 zero padding)\n"
    s += c_array("nb_list", "uint16_t", np.concatenate([plist, np.zeros(64, dtype=np.uint16)]),
                 "{}".format)
    s += "// baseline: expanded per-element u32 indices (cluster*4 + lane)\n"
    s += c_array("nb_list_exp", "uint32_t", plist_exp, "{}".format, align=128)
    s += "// expected forces fo[c*12 + 3*a + d], fp32 bit patterns (float64 reference)\n"
    s += c_array("nb_expected_bits", "const uint32_t", f32_bits(expected), hex32,
                 data_section=False)
    s += ("static float nb_f[{}] __attribute__((section(\".data\"), aligned(64)));\n"
          ).format(nc_tile * 12)
    path = out_dir / "data_{}.h".format(cfg)
    path.write_text(s)
    print("wrote data_{}.h  (NC_DOM={} NC_TILE={} LIST={} WINDOW={} cut2={}, |f|max={:.3f}, "
          "pairs in cutoff={}/{}, header {:.1f} MB)".format(
              cfg, nc_dom, nc_tile, lst, window, cut2, np.abs(expected).max(),
              n_in_cut, nc_tile * 4 * lst * 4, path.stat().st_size / 1e6))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nc_dom", type=int, default=23750)
    p.add_argument("--nc_tile", type=int, nargs="+", default=[32, 256])
    p.add_argument("--list", type=int, default=96)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--cut2", type=float, default=2.0)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    for t in args.nc_tile:
        cfg = "adh" if t == 256 else "adh_t{}".format(t)
        emit(cfg, args.nc_dom, t, args.list, args.window, args.cut2, out_dir)
