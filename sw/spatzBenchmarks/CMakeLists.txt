# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

# Defines
set(SNRT_NFPU_PER_CORE "0" CACHE STRING "Number of FPUs per Spatz")

# Allow spatzBenchmarks to be built as a standalone library.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    set(CMAKE_TOOLCHAIN_FILE toolchain-gcc CACHE STRING "Toolchain to use")

    project(Benchmarks LANGUAGES C ASM)
    include(SnitchUtilities)

    # Build the runtime.
    add_subdirectory(../snRuntime snRuntime)
endif()

include_directories(include)
include_directories(${SNRUNTIME_INCLUDE_DIRS})

add_compile_options(-O3 -g -ffunction-sections)


# Macro to regenerate the golden values and compile a module
macro(add_spatz_test_noParam name file)
  set(target_name ${name})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

macro(add_spatz_test_oneParam name file param1)
  set(target_name ${name}_M${param1})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}.h" SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

macro(add_spatz_test_twoParam name file param1 param2)
  set(target_name ${name}_M${param1}_N${param2})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${param2}.h" SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

macro(add_spatz_test_threeParam name file param1 param2 param3)
  set(target_name ${name}_M${param1}_N${param2}_K${param3})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${param2}_${param3}.h" SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

# Parametrized SpMV variant: (format, N, P_W). Selects data header
# `data/data_spmv_<fmt>_N<N>_PW<PW>.h`. Target name suffixed with the shape.
macro(add_spatz_test_spmv name file fmt N P_W)
  set(target_name ${name}_${fmt}_N${N}_PW${P_W})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC
    DATAHEADER="data/data_spmv_${fmt}_N${N}_PW${P_W}.h"
    SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

# Parametrized SpMM variant: (format, M, N, P_W). Selects data header
# `data/data_spmm_<fmt>_M<M>_N<N>_PW<PW>.h`. Target name suffixed with the shape.
macro(add_spatz_test_spmm name file fmt M N P_W)
  set(target_name ${name}_${fmt}_M${M}_N${N}_PW${P_W})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC
    DATAHEADER="data/data_spmm_${fmt}_M${M}_N${N}_PW${P_W}.h"
    SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

# Benchmark library
add_library(benchmark benchmark/benchmark.c)

# Kernels
# add_library(dp-fmatmul dp-fmatmul/kernel/dp-fmatmul.c)
# add_library(sp-fmatmul sp-fmatmul/kernel/sp-fmatmul.c)
# add_library(hp-fmatmul hp-fmatmul/kernel/hp-fmatmul.c)

# add_library(widening-hp-fmatmul widening-hp-fmatmul/kernel/widening-fmatmul.c)
# add_library(widening-bp-fmatmul widening-bp-fmatmul/kernel/widening-fmatmul.c)

# add_library(sdotp-hp-fmatmul sdotp-hp-fmatmul/kernel/sdotp-fmatmul.c)
# add_library(sdotp-bp-fmatmul sdotp-bp-fmatmul/kernel/sdotp-fmatmul.c)

# add_library(dp-faxpy dp-faxpy/kernel/faxpy.c)

# add_library(dp-fdotp dp-fdotp/kernel/fdotp.c)

# add_library(dp-fconv2d dp-fconv2d/kernel/fconv2d.c)

# add_library(dp-fft dp-fft/kernel/fft.c)
# add_library(sp-fft sp-fft/kernel/fft.c)

# add_library(f32-SpMV-vfx-gated f32-SpMV-vfx-gated/data/data_vfxmacc.c)
# add_library(f32-SpMM-vfx-gated f32-SpMM-vfx-gated/data/data_vfxmacc.c)

# Tests
enable_testing()
set(SNITCH_TEST_PREFIX spatzBenchmarks-)

# add_spatz_test_threeParam(dp-fmatmul dp-fmatmul/main.c 64  64  64 )

# add_spatz_test_threeParam(sp-fmatmul sp-fmatmul/main.c 64  64  64 )
# add_spatz_test_threeParam(sp-fmatmul sp-fmatmul/main.c 64  128 64 )

# add_spatz_test_threeParam(hp-fmatmul hp-fmatmul/main.c 64  64  64 )
# add_spatz_test_threeParam(hp-fmatmul hp-fmatmul/main.c 64  128 64 )
# add_spatz_test_threeParam(hp-fmatmul hp-fmatmul/main.c 128 128 128)

# add_spatz_test_threeParam(widening-hp-fmatmul widening-hp-fmatmul/main.c 64  64  64 )
# add_spatz_test_threeParam(widening-hp-fmatmul widening-hp-fmatmul/main.c 64  128 64 )
# add_spatz_test_threeParam(widening-hp-fmatmul widening-hp-fmatmul/main.c 128 128 128)

# add_spatz_test_threeParam(widening-bp-fmatmul widening-bp-fmatmul/main.c 64  64  64 )
# add_spatz_test_threeParam(widening-bp-fmatmul widening-bp-fmatmul/main.c 64  128 64 )
# add_spatz_test_threeParam(widening-bp-fmatmul widening-bp-fmatmul/main.c 128 128 128)
# add_spatz_test_threeParam(widening-bp-fmatmul widening-bp-fmatmul/main.c 128 256 128)

# add_spatz_test_threeParam(sdotp-hp-fmatmul sdotp-hp-fmatmul/main.c 64  64  64 )
# add_spatz_test_threeParam(sdotp-hp-fmatmul sdotp-hp-fmatmul/main.c 64  128 64 )
# add_spatz_test_threeParam(sdotp-hp-fmatmul sdotp-hp-fmatmul/main.c 128 128 128)

# add_spatz_test_threeParam(sdotp-bp-fmatmul sdotp-bp-fmatmul/main.c 64  64  64 )
# add_spatz_test_threeParam(sdotp-bp-fmatmul sdotp-bp-fmatmul/main.c 64  128 64 )
# add_spatz_test_threeParam(sdotp-bp-fmatmul sdotp-bp-fmatmul/main.c 128 128 128)
# add_spatz_test_threeParam(sdotp-bp-fmatmul sdotp-bp-fmatmul/main.c 128 256 128)

# add_spatz_test_oneParam(dp-faxpy dp-faxpy/main.c 256)
# add_spatz_test_oneParam(dp-faxpy dp-faxpy/main.c 1024)

# add_spatz_test_oneParam(dp-fdotp dp-fdotp/main.c 128)
# add_spatz_test_oneParam(dp-fdotp dp-fdotp/main.c 4096)

# add_spatz_test_threeParam(dp-fconv2d dp-fconv2d/main.c 32 32 7)
# add_spatz_test_threeParam(dp-fconv2d dp-fconv2d/main.c 64 64 7)

# add_spatz_test_twoParam(dp-fft dp-fft/main.c 128 2)

# add_spatz_test_twoParam(sp-fft sp-fft/main.c 256 2)
# add_spatz_test_twoParam(sp-fft sp-fft/main.c 512 2)

# sp-SpMV / sp-SpMM are parametrized by sparsity format + dimensions.
# Each shape is registered TWICE — once as Ventaglio (vfx) and once as
# `-baseline` (RVV reference, compiled with -DUSE_BASELINE). Data headers
# are committed under data/ and regenerated via script/gen_all.sh.
#
#   SpMV : (format, N, P_W)
#   SpMM : (format, M, N, P_W)
#
# To add a shape: edit script/gen_all.sh in the benchmark, run it, then
# add the matching add_spatz_test_spm{v,m}(...) lines below.

# --- SpMV variants (N=32, P_W=128) ---
add_spatz_test_spmv(sp-SpMV          sp-SpMV/main.c 1to4 32 128)
add_spatz_test_spmv(sp-SpMV-baseline sp-SpMV/main.c 1to4 32 128)
target_compile_definitions(test-${SNITCH_TEST_PREFIX}sp-SpMV-baseline_1to4_N32_PW128
  PUBLIC USE_BASELINE)

add_spatz_test_spmv(sp-SpMV          sp-SpMV/main.c 2to4 32 128)
add_spatz_test_spmv(sp-SpMV-baseline sp-SpMV/main.c 2to4 32 128)
target_compile_definitions(test-${SNITCH_TEST_PREFIX}sp-SpMV-baseline_2to4_N32_PW128
  PUBLIC USE_BASELINE)

# --- SpMM variants (M=8, N=32, P_W=128) ---
add_spatz_test_spmm(sp-SpMM          sp-SpMM/main.c 1to4 8 32 128)
add_spatz_test_spmm(sp-SpMM-baseline sp-SpMM/main.c 1to4 8 32 128)
target_compile_definitions(test-${SNITCH_TEST_PREFIX}sp-SpMM-baseline_1to4_M8_N32_PW128
  PUBLIC USE_BASELINE)

add_spatz_test_spmm(sp-SpMM          sp-SpMM/main.c 2to4 8 32 128)
add_spatz_test_spmm(sp-SpMM-baseline sp-SpMM/main.c 2to4 8 32 128)
target_compile_definitions(test-${SNITCH_TEST_PREFIX}sp-SpMM-baseline_2to4_M8_N32_PW128
  PUBLIC USE_BASELINE)

# add_spatz_test_noParam(f32-SpMV-vfx-gated f32-SpMV-vfx-gated/main.c)
# add_spatz_test_noParam(f32-SpMM-vfx-gated f32-SpMM-vfx-gated/main.c)