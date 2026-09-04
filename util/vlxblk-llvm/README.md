# VLXBLK LLVM 14 toolchain

The `vlxblkei8/16.v`, `vsxblkei8/16.v` and `vsetblklen` mnemonics are Spatz
custom instructions on custom-1. This directory provides the MC-layer
(assembler) support so kernels use real mnemonics instead of `.word`.

The compiler is otherwise ennest's exact runtime toolchain — **clang 14.0.6
from `mp-17/llvm-project @ 91be672`** — so all normal codegen is
byte-identical to the shared prebuilt. Only the RISC-V assembler gained the
encodings; there is no codegen/lowering change.

## Just use the prebuilt (any IIS machine)

The build is installed, network-visible, at:

    /usr/scratch2/risa/bowwang/llvm14-vlxblk/install    (clang 14.0.6)

`util/Makefrag` already points `LLVM_INSTALL_DIR` here. `GCC_INSTALL_DIR`
(libc/crt sysroot) is unchanged. Nothing else to do — build the cluster sw
as usual.

## Build it from source (other filesystems / to audit)

    bash build-llvm14-vlxblk.sh

Edit the paths at the top first (ROOT, host CC/CXX). It shallow-fetches
`mp-17/llvm-project @ 91be672`, applies `vlxblk-mc-layer.patch`, builds
clang+lld (RISCV, Release, ninja, ~30 min, ~2 GB), installs to
`$ROOT/install`, creates the `llvm-ranlib`/`strip`/`readelf` multicall
aliases, and verifies the five mnemonics assemble to the expected
encodings. No newlib build — libc/crt come from `GCC_INSTALL_DIR`.

Then point the cluster build at it:

    make ... LLVM_INSTALL_DIR=<your>/install

## The patch

`vlxblk-mc-layer.patch` (57 lines against `mp-17@91be672`): two format
classes in `RISCVInstrFormatsV.td` and two instruction classes + five defs
in `RISCVInstrInfoV.td`. Encodings (custom-1, 0101011): loads funct7
**0x0C**, stores **0x0D**, vsetblklen **0x0F**; funct3 selects index EEW
(000=EI8, 101=EI16). These match the RTL and the `riscv-opcodes` file
(`util/opcodes-vlxblk_CUSTOM`) exactly.

## Verified

`llvm-mc -show-encoding` gives the same words as the old `.word` macros
(e.g. `vlxblkei16.v v8,(x6),v2` -> 0x1823542b). With this toolchain the
cluster sw builds clean, the ISA gate passes 13/13, and cycle counts are
bit-identical to the prebuilt (vqdecode-vlxblk-t16 = 54901).
