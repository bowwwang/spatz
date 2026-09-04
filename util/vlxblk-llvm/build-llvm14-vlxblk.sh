#!/usr/bin/env bash
set -uo pipefail
ROOT=/scratch2/bowwang/llvm14-vlxblk
export PATH=/scratch2/bowwang/spatz-ennest/.toolshim:/usr/local/anaconda3-2023.07/bin:$PATH
CC=/usr/pack/gcc-11.2.0-af/linux-x64/bin/gcc
CXX=/usr/pack/gcc-11.2.0-af/linux-x64/bin/g++
mkdir -p $ROOT/llvm-project/build && cd $ROOT/llvm-project/build
if [ ! -f build.ninja ]; then
  cmake -G Ninja \
    -DCMAKE_INSTALL_PREFIX=$ROOT/install \
    -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
    -DLLVM_OPTIMIZED_TABLEGEN=True \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_TARGETS_TO_BUILD="RISCV" \
    -DLLVM_DEFAULT_TARGET_TRIPLE=riscv32-unknown-elf \
    -DLLVM_ENABLE_LLD=False -DLLVM_APPEND_VC_REV=OFF \
    -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DCMAKE_BUILD_TYPE=Release ../llvm > $ROOT/cmake.log 2>&1
  rc=$?; if [ $rc -ne 0 ]; then echo "CMAKE FAIL"; tail -15 $ROOT/cmake.log; exit 1; fi
fi
echo "CMAKE OK"
COMPS="clang lld llvm-objdump llvm-objcopy llvm-ar llvm-nm llvm-readobj llvm-mc llvm-size clang-resource-headers"
ninja $COMPS > $ROOT/ninja.log 2>&1
rc=$?; if [ $rc -ne 0 ]; then echo "BUILD FAIL rc=$rc"; grep -iE "error:|FAILED" $ROOT/ninja.log | head -15; exit 1; fi
echo "BUILD OK"
for c in $COMPS; do ninja install-$c >> $ROOT/install.log 2>&1; done
# Multicall aliases the per-component install does not create:
( cd $ROOT/install/bin && ln -sf llvm-ar llvm-ranlib && ln -sf llvm-objcopy llvm-strip && ln -sf llvm-readobj llvm-readelf )
echo "INSTALL OK: $($ROOT/install/bin/clang --version | head -1)"
# --- MC-layer verification: mnemonics -> encodings, compare to .word values ---
cd $ROOT
cat > mctest.s << 'ASM'
vsetblklen t0
vlxblkei8.v  v16, (t2), v28
vlxblkei16.v v8,  (t1), v2
vsxblkei8.v  v8,  (t1), v2
vsxblkei16.v v8,  (t1), v2
ASM
$ROOT/install/bin/llvm-mc -triple=riscv32 -mattr=+v -show-encoding mctest.s > mctest.out 2>&1
echo "=== MC encodings (new mnemonics):"; cat mctest.out
echo LLVM14_BUILD_DONE
