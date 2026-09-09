#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# vqgemm data generator: emits data/data_<cfg>.h for the two paper configs
#   aqlm : 2 codebooks x 256 entries x 8  fp16 (16-B blocks, u8 indices)
#   vptq : 2 codebooks x 4096 entries x 16 fp16 (32-B blocks, u16 indices)
# C[M,N] = sum_k A[m,k] * scale[k] * (cb0[idx0[k,g], d] + cb1[idx1[k,g], d])
# (n = g*CB_D + d), i.e. the vqgemv decode fused into an M-row GEMM.
#
# EVERYTHING the kernels read is emitted literally (A, scales, indices,
# both codebooks) plus the expected output; the mains contain no data
# generation. Values are exact dyadic rationals so the fp16 literals
# round-trip exactly. Both arms are software-pipelined two-round kernels:
# K and groups (= N/CB_D) must be even; the index arrays carry 256
# elements of tail padding for the vlxblk arm's explicit-EEW over-read.

import argparse
import pathlib
import numpy as np

HEAD = 1952  # period of the codebook pattern (elements), kept for value continuity


def codebook(cbe, mult):
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


def emit(cfg, M, N, K, CB_D, CBN, idx_bytes, out_dir):
    groups = N // CB_D
    rng_k = np.arange(K, dtype=np.int64)
    scales = ((64 + 2 * (rng_k % 7)) / 128.0).astype(np.float16)  # [0.5, 0.594]
    ia = np.arange(M * K, dtype=np.int64)
    # A row-major [M, K]; (i*41) % 51 with 41 coprime to 51 makes every row
    # of a 4-row tile distinct, so a swapped/duplicated row accumulator is
    # caught by the exact check.
    a = ((20 + (ia * 41) % 51) / 1024.0).astype(np.float16).reshape(M, K)  # [0.0195, 0.069]
    ii = np.arange(K * groups, dtype=np.int64)
    idx0 = (ii * 179 + 3) % CBN
    idx1 = (ii * 83 + 17) % CBN
    cb0 = codebook(CBN * CB_D, 37)
    cb1 = codebook(CBN * CB_D, 53)

    # Bit-exact emulation of the kernel's fp16 arithmetic. Every output
    # element (m, n) accumulates independently in ascending k, and both arms
    # execute the same op sequence per k on lane n: w = f16(cb0 + cb1);
    # w = f16(w * scale[k]); acc[m] = f16(a[m,k] * w + acc[m]) (vfmacc =
    # single-rounding FMA). The m-tiling (4 vfmacc.vf per k into 4 m4
    # accumulators) does not change any per-element op order, so the
    # per-element emulation is identical to vqgemv's. A float64 op followed
    # by a cast to float16 is exactly the RNE-rounded fp16 result (all
    # intermediates fit in 53 bits), so the expected values match the
    # hardware to the bit and the verifier can be tight enough to catch ONE
    # wrong gathered entry.
    W0 = cb0.astype(np.float64).reshape(CBN, CB_D)[idx0].reshape(K, N)
    W1 = cb1.astype(np.float64).reshape(CBN, CB_D)[idx1].reshape(K, N)
    s64 = scales.astype(np.float64)
    w = (W0 + W1).astype(np.float16)                               # vfadd.vv
    w = (w.astype(np.float64) * s64[:, None]).astype(np.float16)   # vfmul.vf
    w64 = w.astype(np.float64)                                     # [K, N]
    a64 = a.astype(np.float64)                                     # [M, K]
    acc = np.zeros((M, N), dtype=np.float16)
    for k in range(K):
        acc = (a64[:, k:k + 1] * w64[k][None, :] + acc.astype(np.float64)).astype(np.float16)  # vfmacc.vf
    expected = acc.astype(np.float64).reshape(M * N)

    idx_ctype = "uint8_t" if idx_bytes == 1 else "uint16_t"
    s = ("// Copyright 2026 ETH Zurich and University of Bologna.\n"
         "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
         "// SPDX-License-Identifier: Apache-2.0\n\n"
         "// This file was generated automatically by vqgemm/script/gen_data.py\n"
         "// config: {}\n\n".format(cfg))
    s += "#include <stdint.h>\n\n"
    s += ("typedef struct {\n"
          "  unsigned int M;         // output rows (rows of A)\n"
          "  unsigned int N;         // output columns (decoded W columns)\n"
          "  unsigned int K;         // accumulation dimension (A columns = W rows)\n"
          "  unsigned int CB_D;      // codebook entry length (elements)\n"
          "  unsigned int CBN;       // codebook entries per table\n"
          "  unsigned int IDX_BYTES; // 1 = u8 indices (vlxblkei8), 2 = u16 (vlxblkei16)\n"
          "} vqgemm_layer;\n\n")
    s += ("const vqgemm_layer vq_l = {{.M = {}, .N = {}, .K = {}, .CB_D = {}, .CBN = {}, "
          ".IDX_BYTES = {}}};\n\n").format(M, N, K, CB_D, CBN, idx_bytes)
    s += "// A[M][K], row-major\n"
    s += c_array("vq_a", "__fp16", a.reshape(M * K).astype(np.float64), "(__fp16){:.10e}")
    s += c_array("vq_scales", "__fp16", scales.astype(np.float64), "(__fp16){:.10e}")
    s += "// idx[K][groups], row-major (entry numbers)\n"
    # +256 zero elements of tail padding: the pipelined vlxblk kernel loads
    # indices under the tile vtype (explicit EEW) and over-reads up to N
    # elements past the current k-row (never used, must be addressable).
    pad = np.zeros(256, dtype=np.int64)
    s += c_array("vq_idx0", idx_ctype, np.concatenate([idx0, pad]), "{}")
    s += c_array("vq_idx1", idx_ctype, np.concatenate([idx1, pad]), "{}")
    s += "// expected C[M][N], row-major (fp16 results widened to float)\n"
    s += c_array("vq_expected", "const float", expected, "{:.9e}f", data_section=False)
    # codebooks: literal (CBN x CB_D fp16 each)
    s += c_array("vq_cb0", "__fp16", cb0.astype(np.float64), "(__fp16){:.10e}")
    s += c_array("vq_cb1", "__fp16", cb1.astype(np.float64), "(__fp16){:.10e}")
    # output (zero-initialized static) + the rvv arm's DOUBLE-BUFFERED decode
    # scratch rows (k parity), each +16 elements for the padded 32-B stores
    s += ("static __fp16 vq_c[{0}] __attribute__((section(\".data\"), aligned(64)));\n"
          "static __fp16 vq_wrow[{1}] __attribute__((section(\".data\"), aligned(64)));\n"
          ).format(M * N, 2 * (N + 16))
    (out_dir / "data_{}.h".format(cfg)).write_text(s)
    print("wrote data_{}.h  (M={} N={} K={} CB_D={} CBN={} idx={}B, |expected|max={:.3f})".format(
        cfg, M, N, K, CB_D, CBN, idx_bytes, np.abs(expected).max()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--K", type=int, default=128)
    args = p.parse_args()
    out_dir = pathlib.Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    emit("aqlm", args.M, args.N, args.K, CB_D=8, CBN=256, idx_bytes=1, out_dir=out_dir)
    emit("vptq", args.M, args.N, args.K, CB_D=16, CBN=4096, idx_bytes=2, out_dir=out_dir)
