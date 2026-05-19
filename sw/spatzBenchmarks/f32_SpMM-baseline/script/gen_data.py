#!/usr/bin/env python3
# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Bowen Wang <bowwang@iis.ee.ethz.ch>
#
# Emit `data/data_spmm.h` from a hjson config (`script/spmm.json`).
# Mirrors the structure of `hp-fmatmul/script/gen_data.py` and the sister
# `sp-SpMV/script/gen_data.py`: parse a hjson config, emit a layer
# instance + named data arrays. No torch dependency.

import argparse
import pathlib
import random
import hjson
from typing import List, Tuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_format(fmt: str) -> Tuple[int, int]:
    fmt = fmt.strip().lower()
    mapping = {"2_to_4": (2, 4), "1_to_4": (1, 4)}
    if fmt not in mapping:
        raise ValueError(
            f"Unsupported format {fmt!r}. Use one of: {', '.join(mapping.keys())}")
    return mapping[fmt]


def pack_indices_row(indices: List[int], idx_width: int) -> List[int]:
    """Pack a row of small indices into uint32 words (little-endian within word)."""
    total_bits = len(indices) * idx_width
    num_words = ceil_div(total_bits, 32)
    words = [0] * num_words

    bitpos = 0
    mask = (1 << idx_width) - 1
    for v in indices:
        v &= mask
        w = bitpos // 32
        o = bitpos % 32
        if o + idx_width <= 32:
            words[w] |= (v << o)
        else:
            lo_bits = 32 - o
            hi_bits = idx_width - lo_bits
            lo_mask = (1 << lo_bits) - 1
            words[w] |= ((v & lo_mask) << o)
            words[w + 1] |= (v >> lo_bits) & ((1 << hi_bits) - 1)
        bitpos += idx_width
    return [w & 0xFFFFFFFF for w in words]


def c_array_floats(name: str, arr: List[float], per_line: int = 8) -> str:
    out = [f'static const float {name}[{len(arr)}] __attribute__((aligned(64))) = {{']
    for i, v in enumerate(arr):
        if i % per_line == 0:
            out.append('  ')
        out[-1] += f'{v:.8e}f'
        if i != len(arr) - 1:
            out[-1] += ', '
        if (i % per_line == per_line - 1) and (i != len(arr) - 1):
            out.append('')
    out.append('\n};\n')
    return '\n'.join(out)


def c_array_u32(name: str, arr: List[int], per_line: int = 6) -> str:
    out = [f'static const uint32_t {name}[{len(arr)}] __attribute__((aligned(64))) = {{']
    for i, v in enumerate(arr):
        if i % per_line == 0:
            out.append('  ')
        out[-1] += f'0x{v:08x}u'
        if i != len(arr) - 1:
            out[-1] += ', '
        if (i % per_line == per_line - 1) and (i != len(arr) - 1):
            out.append('')
    out.append('\n};\n')
    return '\n'.join(out)


def emit_spmm_layer(name: str, **kwargs) -> str:
    s = ''
    s += '#include "layer.h"\n\n'
    s += f'const spmm_layer {name}_l = {{\n'
    s += f'\t.M                  = {kwargs["M"]},\n'
    s += f'\t.N                  = {kwargs["N"]},\n'
    s += f'\t.P                  = {kwargs["P"]},\n'
    s += f'\t.P_W                = {kwargs["P_W"]},\n'
    s += f'\t.N_SPARSE           = {kwargs["n_sparse"]},\n'
    s += f'\t.M_SPARSE           = {kwargs["m_sparse"]},\n'
    s += f'\t.IDX_WIDTH          = {kwargs["idx_width"]},\n'
    s += f'\t.NM_INDEX_ROW_WORDS = {kwargs["row_words"]},\n'
    s += f'\t.NM_INDEX_WORDS     = {kwargs["total_index_words"]}\n'
    s += '};\n\n'
    s += c_array_floats(f'{name}_a_dram',        kwargs['a'])
    s += '\n'
    s += c_array_floats(f'{name}_w_dram',        kwargs['w'])
    s += '\n'
    s += c_array_u32(   f'{name}_nm_index_dram', kwargs['nm_index_words'])
    s += '\n'
    s += c_array_floats(f'{name}_golden_dram',   kwargs['golden'])
    return s


def gen_spmm(param):
    n_sparse, m_sparse = parse_format(param['format'])
    idx_width = (m_sparse - 1).bit_length()

    M   = int(param['M'])
    N   = int(param['N'])
    P_W = int(param['P_W'])
    seed = int(param['seed'])

    if M <= 0:
        raise ValueError('M must be positive.')
    if N <= 0:
        raise ValueError('N must be positive.')
    if m_sparse != 4:
        raise ValueError('This script assumes m=4 (formats *_to_4).')
    if P_W % n_sparse != 0:
        raise ValueError(
            f'Require P_W divisible by n for {n_sparse}:{m_sparse}. '
            f'Got P_W={P_W}, n={n_sparse}.')
    if (P_W * idx_width) % 32 != 0:
        raise ValueError(
            f'Require (P_W * IDX_WIDTH) multiple of 32 for u32-aligned packed '
            f'indices.  Got P_W={P_W}, IDX_WIDTH={idx_width} -> '
            f'{P_W * idx_width} bits.')

    blocks = P_W // n_sparse
    P      = blocks * m_sparse

    rng = random.Random(seed)
    # Activation matrix A (row-major): M rows, N cols
    a = [rng.uniform(-3.0, 3.0) for _ in range(M * N)]
    # Sparse weight matrix W (compact, row-major): N rows, P_W nnz each
    w = [rng.uniform(-2.0, 2.0) for _ in range(N * P_W)]

    # Per sparse row, per dense block, choose n unique positions of 0..m-1.
    nm_index_words: List[int] = []
    all_indices: List[List[int]] = []
    for _row in range(N):
        row_idx: List[int] = []
        for _blk in range(blocks):
            picks = rng.sample(range(m_sparse), n_sparse)
            picks.sort()
            row_idx.extend(picks)
        all_indices.append(row_idx)
        nm_index_words.extend(pack_indices_row(row_idx, idx_width))

    row_words   = ceil_div(P_W * idx_width, 32)
    total_words = N * row_words
    assert len(nm_index_words) == total_words

    # Golden:  C[i*P + p] += A[i*N + k] * W[k*P_W + j]
    golden = [0.0] * (M * P)
    for i in range(M):
        base_a = i * N
        base_c = i * P
        for k in range(N):
            act = a[base_a + k]
            base_w = k * P_W
            idx_row = all_indices[k]
            for j in range(P_W):
                blk = j // n_sparse
                lane = idx_row[j]
                out_pos = m_sparse * blk + lane
                golden[base_c + out_pos] += act * w[base_w + j]

    return {
        'format':    param['format'],
        'M':         M,
        'N':         N,
        'P':         P,
        'P_W':       P_W,
        'n_sparse':  n_sparse,
        'm_sparse':  m_sparse,
        'idx_width': idx_width,
        'row_words': row_words,
        'total_index_words': total_words,
        'seed':      seed,
        'a':         a,
        'w':         w,
        'nm_index_words': nm_index_words,
        'golden':    golden,
    }


def emit_header_file(kernel: str, **kwargs):
    out_dir = pathlib.Path(__file__).parent.parent / 'data'
    if kernel == 'SpMM':
        out = out_dir / 'data_spmm.h'
        body = emit_spmm_layer('spmm', **kwargs)
    else:
        raise ValueError(f'Unknown kernel: {kernel!r}')

    header = (
        '// Copyright 2025 ETH Zurich and University of Bologna.\n'
        '// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n'
        '// SPDX-License-Identifier: Apache-2.0\n\n'
        '// This file was generated automatically.\n'
        f'// SpMM {kwargs["n_sparse"]}:{kwargs["m_sparse"]} fp32: '
        f'M={kwargs["M"]}, N={kwargs["N"]}, P_W={kwargs["P_W"]}, P={kwargs["P"]}, '
        f'IDX_WIDTH={kwargs["idx_width"]}, '
        f'row_words={kwargs["row_words"]}, seed={kwargs["seed"]}\n\n'
        '#pragma once\n\n'
        '#include <stdint.h>\n\n'
    )
    with out.open('w') as f:
        f.write(header + body)
    print(f'Wrote {out}')


def main():
    parser = argparse.ArgumentParser(description='Generate data for sp-SpMM')
    parser.add_argument('-c', '--cfg', type=pathlib.Path, required=True,
                        help='hjson config (e.g., script/spmm.json)')
    args = parser.parse_args()

    with args.cfg.open() as f:
        param = hjson.loads(f.read())

    if param['kernel'] == 'SpMM':
        kwargs = gen_spmm(param)
        emit_header_file('SpMM', **kwargs)
    else:
        raise ValueError(f'Unknown kernel: {param["kernel"]!r}')


if __name__ == '__main__':
    main()
