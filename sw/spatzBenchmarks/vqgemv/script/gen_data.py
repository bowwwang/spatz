#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# vqgemv data generator: emits data/data_<cfg>.h for the two paper configs
#   aqlm : 2 codebooks x 256 entries x 8  fp16 (16-B blocks, u8 indices)
#   vptq : 2 codebooks x 4096 entries x 16 fp16 (32-B blocks, u16 indices)
# c[N] = sum_k a[k] * scale[k] * (cb0[idx0[k,g], d] + cb1[idx1[k,g], d])
#
# K must be even (the pipelined vlxblk kernel processes two k-rows per
# iteration); the index arrays carry 256 elements of tail padding for its
# explicit-EEW index over-read.
#
# EVERYTHING the kernel reads is emitted literally (activations, scales,
# indices, both codebooks) plus the expected output — the header is the
# single source of truth, the mains contain no data generation. Values are
# exact dyadic rationals (multiples of 2^-10 / 2^-7) so the fp16 literals
# round-trip exactly. VPTQ codebooks = 2 x 65,536 fp16 (~2.6 MB of source,
# git-ignored like all data headers).

import argparse
import pathlib
import numpy as np

HEAD = 1952  # period of the codebook pattern (elements), kept for value continuity


def codebook(cbe, mult):
    # closed-form fp16 pattern: ((i*mult) & 1023) - 512) / 1024, period HEAD
    i = np.arange(HEAD, dtype=np.uint64)
    head = ((((i * mult) & 1023).astype(np.int64) - 512) / 1024.0).astype(np.float16)
    if cbe <= HEAD:
        return head[:cbe]
    reps = (cbe + HEAD - 1) // HEAD
    return np.tile(head, reps)[:cbe]


def c_array(name, ctype, vals, fmt, align=64, data_section=True):
    # const arrays (expected output) must not carry the .data section
    # attribute: a const object in .data is a section type conflict.
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


def emit(cfg, K, N, CB_D, CBN, idx_bytes, out_dir):
    groups = N // CB_D
    rng_k = np.arange(K, dtype=np.int64)
    a = ((20 + rng_k % 51) / 1024.0).astype(np.float16)          # [0.0195, 0.069]
    scales = ((64 + 2 * (rng_k % 7)) / 128.0).astype(np.float16)  # [0.5, 0.594]
    ii = np.arange(K * groups, dtype=np.int64)
    idx0 = (ii * 179 + 3) % CBN
    idx1 = (ii * 83 + 17) % CBN
    cb0 = codebook(CBN * CB_D, 37)
    cb1 = codebook(CBN * CB_D, 53)

    # Bit-exact emulation of the kernel's fp16 arithmetic. Each output lane
    # accumulates independently in ascending k, and both arms execute the
    # same op sequence per k: w = f16(cb0 + cb1); w = f16(w * scale);
    # acc = f16(a * w + acc) (vfmacc = single-rounding FMA). A float64 op
    # followed by a cast to float16 is exactly the RNE-rounded fp16 result,
    # so the expected values match the hardware to the bit and the verifier
    # can be tight enough to catch ONE wrong gathered entry.
    W0 = cb0.astype(np.float64).reshape(CBN, CB_D)[idx0].reshape(K, groups, CB_D)
    W1 = cb1.astype(np.float64).reshape(CBN, CB_D)[idx1].reshape(K, groups, CB_D)
    a64 = a.astype(np.float64)
    s64 = scales.astype(np.float64)

    # Both arms execute the same per-lane fp16 sequence per k (the pipelined
    # vlxblk kernel only reorders ACROSS k-rounds, never within a lane):
    #   w = f16(cb0+cb1) [vfadd]; w = f16(w*scale) [vfmul.vf]; acc = f16(a*w + acc) [vfmacc]
    acc = np.zeros(N, dtype=np.float16)
    for k in range(K):
        w = (W0[k] + W1[k]).reshape(N).astype(np.float16)
        w = (w.astype(np.float64) * s64[k]).astype(np.float16)
        acc = (a64[k] * w.astype(np.float64) + acc.astype(np.float64)).astype(np.float16)
    expected = acc.astype(np.float64)

    idx_ctype = "uint8_t" if idx_bytes == 1 else "uint16_t"
    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by vqgemv/script/gen_data.py\n"
         "// config: {}\n\n".format(cfg))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int K;         // accumulation dimension\n"
          "  unsigned int N;         // output dimension\n"
          "  unsigned int CB_D;      // codebook entry length (elements)\n"
          "  unsigned int CBN;       // codebook entries per table\n"
          "  unsigned int IDX_BYTES; // 1 = u8 indices (vlxblkei8), 2 = u16 (vlxblkei16)\n"
          "} vqgemv_layer;\n\n")
    s += ("const vqgemv_layer vq_l = {{.K = {}, .N = {}, .CB_D = {}, .CBN = {}, "
          ".IDX_BYTES = {}}};\n\n").format(K, N, CB_D, CBN, idx_bytes)
    s += c_array("vq_a", "__fp16", a.astype(np.float64), "(__fp16){:.10e}")
    s += c_array("vq_scales", "__fp16", scales.astype(np.float64), "(__fp16){:.10e}")
    # +256 zero elements of tail padding: the pipelined vlxblk kernel loads
    # indices under the group vtype (explicit EEW) and over-reads gvl <= 256
    # elements past the current k-row (never used, must be addressable).
    pad = np.zeros(256, dtype=np.int64)
    s += c_array("vq_idx0", idx_ctype, np.concatenate([idx0, pad]), "{}")
    s += c_array("vq_idx1", idx_ctype, np.concatenate([idx1, pad]), "{}")
    # ".9e" always carries an exponent -> a valid float literal even for
    # integral values ("{:.9g}f" would emit "2f")
    s += c_array("vq_expected", "const float", expected, "{:.9e}f", data_section=False)
    # codebooks: literal (CBN x CB_D fp16 each)
    s += c_array("vq_cb0", "__fp16", cb0.astype(np.float64), "(__fp16){:.10e}")
    s += c_array("vq_cb1", "__fp16", cb1.astype(np.float64), "(__fp16){:.10e}")
    # output buffer (zero-initialized static); +16 elements: the rvv arm's
    # per-group stores are padded to 32 B
    s += ("static __fp16 vq_c[{0}] __attribute__((section(\".data\"), aligned(64)));\n"
          ).format(N + 16)
    (out_dir / "data_{}.h".format(cfg)).write_text(s)
    print("wrote data_{}.h  (K={} N={} CB_D={} CBN={} idx={}B, |expected|max={:.3f})".format(
        cfg, K, N, CB_D, CBN, idx_bytes, np.abs(expected).max()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--N", type=int, default=128)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    emit("aqlm", args.K, args.N, CB_D=8, CBN=256, idx_bytes=1, out_dir=out_dir)
    emit("vptq", args.K, args.N, CB_D=16, CBN=4096, idx_bytes=2, out_dir=out_dir)
