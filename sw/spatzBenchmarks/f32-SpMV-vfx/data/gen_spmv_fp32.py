#!/usr/bin/env python3
# Generates data_vfxmacc.h for SpMV fp32 tests (activation fp32, weights fp32)
# Supports sparse formats: 2:4 and 1:4

import argparse
import random
from typing import List, Tuple


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def parse_format(fmt: str) -> Tuple[int, int]:
    fmt = fmt.strip().lower()
    mapping = {
        "2_to_4": (2, 4),
        "1_to_4": (1, 4),
    }
    if fmt not in mapping:
        raise ValueError(f"Unsupported --format {fmt!r}. Use one of: {', '.join(mapping.keys())}")
    return mapping[fmt]


def pack_indices_row(indices: List[int], idx_width: int) -> List[int]:
    """
    Pack one row of indices into uint32 words (row-aligned).
    Little-endian packing: idx0 in lowest bits of word0.
    """
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
    out = [f"static const float {name}[{len(arr)}] __attribute__((aligned(64))) = {{"]  # DMA-friendly
    for i, v in enumerate(arr):
        if i % per_line == 0:
            out.append("  ")
        out[-1] += f"{v:.8e}f"
        if i != len(arr) - 1:
            out[-1] += ", "
        if (i % per_line == per_line - 1) and (i != len(arr) - 1):
            out.append("")
    out.append("\n};\n")
    return "\n".join(out)


def c_array_u32(name: str, arr: List[int], per_line: int = 6) -> str:
    out = [f"static const uint32_t {name}[{len(arr)}] __attribute__((aligned(64))) = {{"]  # DMA-friendly
    for i, v in enumerate(arr):
        if i % per_line == 0:
            out.append("  ")
        out[-1] += f"0x{v:08x}u"
        if i != len(arr) - 1:
            out[-1] += ", "
        if (i % per_line == per_line - 1) and (i != len(arr) - 1):
            out.append("")
    out.append("\n};\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", type=str, required=True, choices=["1_to_4", "2_to_4"],
                    help="Sparse format: 1_to_4 or 2_to_4")
    ap.add_argument("--N", type=int, default=16, help="Activation length (rows), default 16")
    ap.add_argument("--p_w", type=int, default=32, help="Compressed width per row (nnz per row), default 32")
    ap.add_argument("--seed", type=int, default=1, help="Random seed, default 1")
    ap.add_argument("--out", type=str, default="data_vfxmacc.h", help="Output header filename")
    args = ap.parse_args()

    # Parse format
    n, m = parse_format(args.format)
    idx_width = (m - 1).bit_length()  # m=4 -> 2 bits

    N = args.N
    P_W = args.p_w

    if N <= 0:
        raise ValueError("N must be positive.")
    if m != 4:
        raise ValueError("This script currently assumes m=4 (formats *_to_4).")

    # Dense width per row implied by n:m and compact width P_W:
    # Each block of m outputs contains n nonzeros, so:
    #   blocks = P_W / n   (must be integer)
    #   P      = blocks * m = (P_W/n) * m
    if P_W % n != 0:
        raise ValueError(f"Require P_W divisible by n for {n}:{m}. Got P_W={P_W}, n={n}.")

    # Keep your old "u32-friendly" constraint (optional, but preserves old test assumptions).
    # With idx_width=2, requiring P_W multiple of 16 makes (P_W*2) bits a multiple of 32.
    if (P_W * idx_width) % 32 != 0:
        raise ValueError(
            f"Require (P_W * IDX_WIDTH) multiple of 32 for simple u32-aligned packed-index addressing "
            f"in current C test. Got P_W={P_W}, IDX_WIDTH={idx_width} -> {P_W*idx_width} bits."
        )

    blocks = P_W // n
    P = blocks * m

    rng = random.Random(args.seed)

    # activation vector
    a = [rng.uniform(-3.0, 3.0) for _ in range(N)]

    # weight matrix (compact, row-major): N rows, P_W nnz each
    w = [rng.uniform(-2.0, 2.0) for _ in range(N * P_W)]

    # index matrix:
    # per row, for each dense block choose n unique positions (0..m-1),
    # append them to row_idx (length P_W)
    nm_index_words: List[int] = []
    all_indices: List[List[int]] = []

    for r in range(N):
        row_idx: List[int] = []
        for _blk in range(blocks):
            # choose n unique positions within the m-lane block
            picks = rng.sample(range(m), n)
            picks.sort()
            row_idx.extend(picks)
        assert len(row_idx) == P_W
        all_indices.append(row_idx)
        nm_index_words.extend(pack_indices_row(row_idx, idx_width))

    row_words = ceil_div(P_W * idx_width, 32)
    total_words = N * row_words
    assert len(nm_index_words) == total_words

    # golden dense output (length P)
    # compact index j belongs to block (j // n), selects one of m positions
    golden = [0.0] * P
    for r in range(N):
        act = a[r]
        base_w = r * P_W
        for j in range(P_W):
            blk = j // n
            pos_in_blk = all_indices[r][j]   # 0..m-1
            out_pos = m * blk + pos_in_blk   # 0..P-1
            golden[out_pos] += act * w[base_w + j]

    guard = "DATA_VFXMACC_H"
    header = []
    header.append("// Auto-generated by gen_spmv_fp32.py")
    header.append(
        f"// SpMV {n}:{m} fp32: N={N}, P_W={P_W}, P={P}, IDX_WIDTH={idx_width}, row_words={row_words}, seed={args.seed}"
    )
    header.append("")
    header.append(f"#ifndef {guard}")
    header.append(f"#define {guard}")
    header.append("")
    header.append("#include <stdint.h>")
    header.append("")
    header.append("// Sparse format")
    header.append(f"#define N_SPARSE   ({n})")
    header.append(f"#define M_SPARSE   ({m})")
    header.append(f"#define IDX_WIDTH  ({idx_width})")
    header.append("")
    header.append("// SpMV dimensions")
    header.append(f"#define N      ({N})")
    header.append(f"#define P_W    ({P_W})")
    header.append(f"#define P      ({P})")
    header.append("")
    header.append("// Packed-index storage")
    header.append(f"#define NM_INDEX_ROW_WORDS ({row_words})")
    header.append(f"#define NM_INDEX_WORDS     ({total_words})")
    header.append("")
    header.append(c_array_floats("a_dram", a))
    header.append(c_array_floats("w_dram", w))
    header.append(c_array_u32("nm_index_dram", nm_index_words))
    header.append(c_array_floats("golden_dram", golden))
    header.append(f"#endif // {guard}\n")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header))

    print(f"Wrote {args.out}")
    print(
        f"  format={args.format} (n={n}, m={m}), N={N}, P_W={P_W}, P={P}, "
        f"NM_INDEX_ROW_WORDS={row_words}, NM_INDEX_WORDS={total_words}"
    )


if __name__ == "__main__":
    main()
