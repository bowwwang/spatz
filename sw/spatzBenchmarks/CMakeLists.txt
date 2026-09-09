# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13)

# Allow spatzBenchmarks to be built as a standalone library.
if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake)
    set(CMAKE_TOOLCHAIN_FILE toolchain-gcc CACHE STRING "Toolchain to use")

    project(Benchmarks LANGUAGES C ASM)
    include(SnitchUtilities)

    # Build the runtime.
    add_subdirectory(../snRuntime snRuntime)
endif()

# default value to use SPM
if(NOT DEFINED USE_CACHE)
  set(USE_CACHE 0)
endif()

# Validate that USE_CACHE is either 0 or 1
if(NOT USE_CACHE MATCHES "^[01]$")
  message(FATAL_ERROR "USE_CACHE must be either 0 or 1. Current value: ${USE_CACHE}")
endif()

# default value to set iteration
if(NOT DEFINED MEAS_1ITER)
  set(MEAS_1ITER 1)
endif()

# Validate that MEAS_1ITER is either 0 or 1
if(NOT MEAS_1ITER MATCHES "^[01]$")
  message(FATAL_ERROR "MEAS_1ITER must be either 0 or 1. Current value: ${MEAS_1ITER}")
endif()

# default value to print with real UART
if(NOT DEFINED PRINT_UART)
  set(PRINT_UART 1)
endif()

# Validate that PRINT_UART is either 0 or 1
if(NOT PRINT_UART MATCHES "^[01]$")
  message(FATAL_ERROR "PRINT_UART must be either 0 or 1. Current value: ${PRINT_UART}")
endif()

include_directories(include)
include_directories(${SNRUNTIME_INCLUDE_DIRS})

add_compile_options(-O3 -g -ffunction-sections)

option(PRINT_CHECK "Enable Printing" OFF)

# Macro to regenerate the golden values and compile a module
# if(PRINT_CHECK)
add_definitions(-DPRINT_RESULT)
macro(add_spatz_test_oneParam name file param1)
  set(target_name ${name}_M${param1})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} MEAS_1ITER=${MEAS_1ITER} PRINT_UART=${PRINT_UART})
endmacro()

macro(add_spatz_test_twoParam name file param1 param2)
  set(target_name ${name}_M${param1}_N${param2})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${param2}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} MEAS_1ITER=${MEAS_1ITER} PRINT_UART=${PRINT_UART})
endmacro()

macro(add_spatz_test_threeParam name file param1 param2 param3)
  set(target_name ${name}_M${param1}_N${param2}_K${param3})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${param2}_${param3}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} MEAS_1ITER=${MEAS_1ITER} PRINT_UART=${PRINT_UART})
endmacro()

# Use data type as another parameter
macro(add_spatz_test_oneParam_type name file param1 type)
  set(target_name ${name}_M${param1})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${type}.h" USE_CACHE=${USE_CACHE} PREC=${type} SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

add_definitions(-DPRINT_RESULT)
macro(add_spatz_test_twoParam_type name file param1 param2 type)
  set(target_name ${name}_M${param1}_N${param2})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${param2}_${type}.h" USE_CACHE=${USE_CACHE} PREC=${type} SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

add_definitions(-DPRINT_RESULT)
macro(add_spatz_test_threeParam_type name file param1 param2 param3 type)
  set(target_name ${name}_M${param1}_N${param2}_K${param3})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${param2}_${param3}_${type}.h" USE_CACHE=${USE_CACHE} PREC=${type} SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

macro(add_spatz_test_threeParam_type_lmul name file param1 param2 param3 type lmul)
  set(target_name ${name}_M${param1}_N${param2}_K${param3}_LMUL${lmul})
  add_snitch_test(${target_name} ${file})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${param1}_${param2}_${param3}_${type}.h" USE_CACHE=${USE_CACHE} PREC=${type} SPMV_LMUL=${lmul} SNRT_NFPU_PER_CORE=${SNRT_NFPU_PER_CORE})
endmacro()

# Benchmark library
add_library(benchmark benchmark/benchmark.c)

# Kernels
add_library(dp-fmatmul dp-fmatmul/kernel/dp-fmatmul.c)
add_library(sp-fmatmul sp-fmatmul/kernel/sp-fmatmul.c)
add_library(hp-fmatmul hp-fmatmul/kernel/hp-fmatmul.c)

add_spatz_test_twoParam_type(dp-gemv gemv/main.c 64  128 64)

add_spatz_test_threeParam_type(dp-sa-gemv sa_gemv/main.c 256 128 16 64)
add_spatz_test_threeParam_type(hp-sa-gemv sa_gemv/main.c 128 4096 512 16)

add_library(dp-faxpy dp-faxpy/kernel/faxpy.c)

add_library(dp-fdotp dp-fdotp/kernel/fdotp.c)
add_library(dp-dbfdotp dp-db-fdotp/kernel/db-fdotp.c)

add_library(dp-fconv2d dp-fconv2d/kernel/fconv2d.c)

add_library(dp-fft dp-fft/kernel/fft.c)
add_library(sp-fft sp-fft/kernel/fft.c)
add_library(dp-db-fft dp-db-fft/kernel/db-fft.c)

add_library(linked-list linked-list/kernel/llist.c)

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

# vqdecode: VQ codebook decode, table-size sweep across the L1 cache
macro(add_vqdecode variant vname tblkb)
  set(target_name vqdecode-${vname}-t${tblkb})
  add_snitch_test(${target_name} vqdecode/main.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC VARIANT=${variant} TBL_KB=${tblkb} PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_vqdecode(1 vlxblk 16)
add_vqdecode(1 vlxblk 32)
add_vqdecode(1 vlxblk 64)
add_vqdecode(1 vlxblk 128)
add_vqdecode(1 vlxblk 256)
add_vqdecode(1 vlxblk 512)
add_vqdecode(1 vlxblk 1024)
add_vqdecode(1 vlxblk 2048)
# dictionary decode (benchmark #2): 16-B fixed-slot entries, same kernel
macro(add_vqdecode_d variant vname dictd tblkb)
  set(target_name vqdecode-${vname}-d${dictd}-t${tblkb})
  add_snitch_test(${target_name} vqdecode/main.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC VARIANT=${variant} DICT_D=${dictd} TBL_KB=${tblkb} PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_vqdecode_d(1 vlxblk 4 16)
add_vqdecode_d(1 vlxblk 4 64)
add_vqdecode_d(1 vlxblk 4 256)
add_vqdecode_d(1 vlxblk 4 1024)
add_vqdecode_d(0 rvv 4 16)
add_vqdecode_d(0 rvv 4 64)
add_vqdecode(0 rvv 16)
add_vqdecode(0 rvv 32)
add_vqdecode(0 rvv 64)

# attn-vagg: sparse-attention V-aggregation (256-B fp16 rows = m4 @ e16).
# sp-fmatmul layout: one main per arm (main-vlxblk.c / main-vle.c), kernel
# in attn-vagg/kernel/, config from the generated DATAHEADER
# (attn-vagg/script/gen_data.py -> data/data_nq<NQ>.h; the header carries
# the NQ/TOPK/HD/NTOK parameters, so no size macros).
# Paper config (BENCHMARK_PLAN 2026-09-07): 16K-token V cache = 4 MiB,
# TOPK=1024 budget, NQ=16.
# Data: script/gen_data.py -> data/data_nq<NQ>.h with ALL arrays literal
# (V pool, ids, scores, per-arm expected outputs); nq16 = paper config,
# nq2 = quick development config.
macro(add_attnvagg arm cfg)
  set(target_name attn-vagg-${arm}-${cfg})
  add_snitch_test(${target_name} attn-vagg/main-${arm}.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_attnvagg(vlxblk nq2)
add_attnvagg(vlxblk nq16)
add_attnvagg(vle nq16)

# gatheragg = paper row "gnnagg": GNN pull-mode neighbour aggregation,
# 256-B fp32 rows (d64), 4-MiB table. sp-fmatmul layout: one main per arm
# (main-vlxblk.c / main-vle.c), kernels in gatheragg/kernel/, config from
# the generated DATAHEADER (gatheragg/script/gen_data.py ->
# data/data_<cfg>.h; ALL arrays literal - table, ids, expected output).
# Paper config (BENCHMARK_PLAN 2026-09-07 ruling): NB=2048 nodes x LP=25
# (GraphSAGE S1), NROWS=16384; d64_nb64 is the quick development config.
# (The former d32 configs are superseded by the sls kernels.)
macro(add_gatheragg arm cfg)
  set(target_name gatheragg-${arm}-${cfg})
  add_snitch_test(${target_name} gatheragg/main-${arm}.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_gatheragg(vlxblk d64_nb64)
add_gatheragg(vlxblk d64_nb2048)
add_gatheragg(vle d64_nb2048)

# isaprobe: instruction-level hang probes (debug aid, not a benchmark)
add_snitch_test(isaprobe isaprobe/main.c)
target_link_libraries(test-${SNITCH_TEST_PREFIX}isaprobe benchmark ${SNITCH_RUNTIME})
target_compile_definitions(test-${SNITCH_TEST_PREFIX}isaprobe PUBLIC PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})

# sls: SparseLengthsSum / embedding-bag pooling (paper points sls-1 = fp16 rows,
# 32-B blocks; sls-2 = fp32 rows, 128-B blocks). sp-fmatmul layout: one main per
# arm and dtype (sls/main-vlxblk.c fp16, main-vlxblk32.c fp32, main-vle.c fp16
# baseline), kernels in sls/kernel/, config from the generated DATAHEADER
# (sls/script/gen_data.py --dtype fp16|fp32 --NROWS ... -> data/data_<dtype>_n<rows>.h).
# NB=64 bags while developing; paper NB=2048.
macro(add_sls arm mainfile cfg)
  set(target_name sls-${arm}-${cfg})
  add_snitch_test(${target_name} sls/${mainfile})
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
# sls-1 (fp16, 32-B rows): L1-resident 128 KiB table and the paper's 2 MiB table
add_sls(vlxblk main-vlxblk.c fp16_n4096)
add_sls(vlxblk main-vlxblk.c fp16_n65536)
add_sls(vle main-vle.c fp16_n4096)
add_sls(vle main-vle.c fp16_n65536)
# sls-2 (fp32, 128-B rows): L1-resident 128 KiB table and the paper's 2 MiB table
add_sls(vlxblk32 main-vlxblk32.c fp32_n1024)
# paper-size runs (NB=2048): generate with `--NB 2048 --tag _nb2048`
add_sls(vlxblk main-vlxblk.c fp16_n65536_nb2048)
add_sls(vlxblk32 main-vlxblk32.c fp32_n16384_nb2048)
add_sls(vlxblk32 main-vlxblk32.c fp32_n16384)
# hang-locator builds (DBG_EVERY progress print; never for measurement)
add_sls(vlxblk main-vlxblk.c fp16_n4096_dbg)
add_sls(vle main-vle.c fp16_n4096_dbg)

# radix-scatter: DB partitioning record scatter (vsxblk vs vsoxei32 baseline).
# sp-fmatmul layout: one main per arm (main-vsxblk.c / main-vsuxei.c), kernels
# in radix-scatter/kernel/, config from the generated DATAHEADER
# (radix-scatter/script/gen_data.py -> data/data_n<nrec>.h; the header carries
# the NREC/RD/FANOUT_LOG2 parameters and the fill seeds, so no size macros;
# slots/src/dst are built on-core from the closed forms the generator mirrors).
# Paper config (BENCHMARK_PLAN 2026-09-07): 65,536 records = u16 slot
# ceiling, 1 MiB dst, F=256 realistic radix slots. The baseline arm is the
# column-decomposed vsoxei32 kernel; its main/target keep the historical
# "vsuxei" name.
macro(add_radix arm cfg)
  set(target_name radix-scatter-${arm}-${cfg})
  add_snitch_test(${target_name} radix-scatter/main-${arm}.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
# Data: script/gen_data.py -> data/data_n<nrec>.h, a real radix partition
# of random keys (F=256) with ALL arrays literal (src, srcT, slots, zero
# dst); n8192 is the quick development config, n65536 the paper config.
add_radix(vsxblk n8192)
add_radix(vsxblk n65536)
add_radix(vsuxei n65536)

# pr-gather: PageRank pull, D=1 x e64 (index-semantics regime point).
# sp-fmatmul layout: one main per arm (main-vlxblk.c / main-vluxei.c),
# kernels in pr-gather/kernel/, config from the generated DATAHEADER
# (pr-gather/script/gen_data.py -> data/data_n<nv>_a<nact>.h; the header
# carries NV/NACT/DEG, damp, the seed vector, ALL input arrays literally
# and the expected output, so no size macros and no on-core datagen).
# The baseline arm keeps its historical `vluxei` file/target name (batch
# vluxei32 hangs on the first missing gather (erratum #3) and misses are
# intrinsic here (512 KiB contrib > L1).
# Paper config (BENCHMARK_PLAN 2026-09-07): NV=65,536, NACT=4,096, DEG=16;
# a512 is the quick development config on the same table.
macro(add_prg arm cfg)
  set(target_name pr-gather-${arm}-${cfg})
  add_snitch_test(${target_name} pr-gather/main-${arm}.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_prg(vlxblk n65536_a512)
add_prg(vlxblk n8192_a512)   # L1-resident reference (64 KiB table)
add_prg(vlxblk n65536_a4096)
add_prg(vluxei n65536_a4096)

# vqgemv: FUSED VQ-GEMV (AQLM/VPTQ 2-codebook decode fused into GEMV).
# sp-fmatmul layout: one main per arm (main-vlxblk.c / main-rvv.c), kernel
# in vqgemv/kernel/, config from the generated DATAHEADER
# (vqgemv/script/gen_data.py -> data/data_<cfg>.h; the header carries the
# K/N/CB_D/CBN parameters, so no size macros).
# Paper configs (BENCHMARK_PLAN 2026-09-07 ruling): P=N=128 tile,
# aqlm = AQLM 2x8 (CB_D=8, CBN=256, u8 idx), vptq = VPTQ v16 (CB_D=16,
# CBN=4096, u16 idx).
macro(add_vqgemv arm cfg)
  set(target_name vqgemv-${arm}-${cfg})
  add_snitch_test(${target_name} vqgemv/main-${arm}.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_vqgemv(vlxblk aqlm)
add_vqgemv(rvv aqlm)
add_vqgemv(vlxblk vptq)
add_vqgemv(rvv vptq)

# "# vqgemm: fused VQ-GEMM, m-tiled (paper configs, M=K=N=128)" through the
# four add_vqgemm_cfg(...) lines). Target names are unchanged:
#   vqgemm-vlxblk-aqlm  vqgemm-rvv-aqlm  vqgemm-vlxblk-vptq  vqgemm-rvv-vptq
# Regenerate the data headers first: python3 vqgemm/script/gen_data.py

# vqgemm: fused VQ-GEMM, m-tiled (AQLM/VPTQ 2-codebook decode fused into
# the matrix product; 4 output rows per tile share one decoded row).
# sp-fmatmul layout like vqgemv: one main per arm (main-vlxblk.c /
# main-rvv.c), kernels in vqgemm/kernel/, config from the generated
# DATAHEADER (vqgemm/script/gen_data.py -> data/data_<cfg>.h; the header
# carries the M/N/K/CB_D/CBN parameters, so no size macros).
# Paper configs (M=N=K=128): aqlm = AQLM 2x8 (CB_D=8, CBN=256, u8 idx),
# vptq = VPTQ v16 (CB_D=16, CBN=4096, u16 idx).
macro(add_vqgemm arm cfg)
  set(target_name vqgemm-${arm}-${cfg})
  add_snitch_test(${target_name} vqgemm/main-${arm}.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_vqgemm(vlxblk aqlm)
add_vqgemm(rvv aqlm)
add_vqgemm(vlxblk vptq)
add_vqgemm(rvv vptq)

# nbforce: GROMACS cluster-pair force, ADH-sized domain (paper config).
# sp-fmatmul layout: one main per arm (main-vlxblk.c / main-rvv.c), kernel
# in nbforce/kernel/, config from the generated DATAHEADER
# (nbforce/script/gen_data.py -> data/data_<cfg>.h; the header carries the
# NC_DOM/NC_TILE/LIST/WINDOW/CUT2 parameters and ALL arrays literally:
# the four X4 field arrays, the pair list, the baseline's expanded index
# array and the expected forces; no size macros, no on-core datagen).
# adh = paper tile (256 i-clusters), adh_t32 = quick development tile.
macro(add_nbforce arm cfg)
  set(target_name nbforce-${arm}-${cfg})
  add_snitch_test(${target_name} nbforce/main-${arm}.c)
  target_link_libraries(test-${SNITCH_TEST_PREFIX}${target_name} benchmark ${SNITCH_RUNTIME})
  target_compile_definitions(test-${SNITCH_TEST_PREFIX}${target_name} PUBLIC DATAHEADER="data/data_${cfg}.h" PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
endmacro()
add_nbforce(vlxblk adh_t32)
add_nbforce(vlxblk adh)
add_nbforce(rvv adh)

# VLXBLK ISA suite through the benchmark harness (riscvTests glue lacks set_eoc)
add_snitch_test(vlxblk-isa vlxblk-isa/main.c)
target_link_libraries(test-${SNITCH_TEST_PREFIX}vlxblk-isa benchmark ${SNITCH_RUNTIME})
target_compile_definitions(test-${SNITCH_TEST_PREFIX}vlxblk-isa PUBLIC PRINT_RESULT USE_CACHE=${USE_CACHE} PRINT_UART=${PRINT_UART})
target_include_directories(test-${SNITCH_TEST_PREFIX}vlxblk-isa PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../riscvTests/isa/macros/vector)

add_spatz_test_oneParam(dp-faxpy dp-faxpy/main.c 256)
add_spatz_test_oneParam(dp-faxpy dp-faxpy/main.c 1024)

add_spatz_test_oneParam(dp-fdotp dp-fdotp/main.c 128)
add_spatz_test_oneParam(dp-fdotp dp-fdotp/main.c 4096)
add_spatz_test_oneParam(dp-fdotp dp-fdotp/main.c 32768)
add_spatz_test_oneParam(dp-db-fdotp dp-db-fdotp/main.c 8192)
add_spatz_test_oneParam(dp-db-fdotp dp-db-fdotp/main.c 32768)

add_spatz_test_threeParam(dp-fconv2d dp-fconv2d/main.c 32 32 7)
add_spatz_test_threeParam(dp-fconv2d dp-fconv2d/main.c 64 64 7)

add_spatz_test_twoParam(dp-fft dp-fft/main.c 128 2)
add_spatz_test_twoParam(dp-fft dp-fft/main.c 1024 2)
add_spatz_test_twoParam(dp-fft dp-fft/main.c 2048 2)

add_spatz_test_threeParam(dp-db-fft dp-db-fft/main.c 256 4 2)
add_spatz_test_threeParam(dp-db-fft dp-db-fft/main.c 1024 4 2)
add_spatz_test_threeParam(dp-db-fft dp-db-fft/main.c 2048 8 2)

# add_spatz_test_twoParam(sp-fft sp-fft/main.c 256 2)
# add_spatz_test_twoParam(sp-fft sp-fft/main.c 512 2)

# add_spatz_test_threeParam(dp-mxfmatmul dp-mxfmatmul/main.c 64 64 64)

# add_spatz_test_twoParam(linked-list linked-list/main.c 64 8)
# add_spatz_test_twoParam(linked-list linked-list/main.c 256 16)

# Tests for DATE
# Vanilla fmatmul
# add_spatz_test_threeParam(dp-fmatmul-8x2vl dp-fmatmul-8x2vl/main.c 64 64 64)
# add_spatz_test_threeParam(dp-fmatmul-8x2vl dp-fmatmul-8x2vl/main.c 32 32 32)
# add_spatz_test_threeParam(dp-fmatmul-8x2vl dp-fmatmul-8x2vl/main.c 16 16 16)
add_spatz_test_threeParam(dp-fmatmul-4x4vl dp-fmatmul-4x4vl/main.c 64 64 64)
# add_spatz_test_threeParam(dp-fmatmul-4x4vl dp-fmatmul-4x4vl/main.c 32 32 32)
# add_spatz_test_threeParam(dp-fmatmul-4x4vl dp-fmatmul-4x4vl/main.c 16 16 16)
add_spatz_test_threeParam(dp-fmatmul-4x4vl dp-fmatmul-4x4vl/main.c 128 128 128)
add_spatz_test_threeParam(dp-fmatmul-4x4vl dp-fmatmul-4x4vl/main.c 128 256 128)
add_spatz_test_threeParam(dp-fmatmul-4x4vl dp-fmatmul-4x4vl/main.c 256 256 256)
# # MXU matmul
# add_spatz_test_threeParam(dp-mxfmatmul-m8n4k4-b4 dp-mxfmatmul-m8n4k4-b4/main.c 64 64 64)
# add_spatz_test_threeParam(dp-mxfmatmul-m8n4k4-b4 dp-mxfmatmul-m8n4k4-b4/main.c 32 32 32)
# add_spatz_test_threeParam(dp-mxfmatmul-m8n4k4-b4 dp-mxfmatmul-m8n4k4-b4/main.c 16 16 16)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k4-b4 dp-mxfmatmul-m4n4k4-b4/main.c 64 64 64)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k4-b4 dp-mxfmatmul-m4n4k4-b4/main.c 32 32 32)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k4-b4 dp-mxfmatmul-m4n4k4-b4/main.c 16 16 16)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k8-b4 dp-mxfmatmul-m4n4k8-b4/main.c 64 64 64)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k8-b4 dp-mxfmatmul-m4n4k8-b4/main.c 32 32 32)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k8-b4 dp-mxfmatmul-m4n4k8-b4/main.c 16 16 16)
# add_spatz_test_threeParam(dp-mxfmatmul-m8n4k4-b2 dp-mxfmatmul-m8n4k4-b2/main.c 64 64 64)
# add_spatz_test_threeParam(dp-mxfmatmul-m8n4k4-b2 dp-mxfmatmul-m8n4k4-b2/main.c 32 32 32)
# add_spatz_test_threeParam(dp-mxfmatmul-m8n4k4-b2 dp-mxfmatmul-m8n4k4-b2/main.c 16 16 16)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k4-b2 dp-mxfmatmul-m4n4k4-b2/main.c 64 64 64)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k4-b2 dp-mxfmatmul-m4n4k4-b2/main.c 32 32 32)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k4-b2 dp-mxfmatmul-m4n4k4-b2/main.c 16 16 16)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k8-b2 dp-mxfmatmul-m4n4k8-b2/main.c 64 64 64)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k8-b2 dp-mxfmatmul-m4n4k8-b2/main.c 32 32 32)
# add_spatz_test_threeParam(dp-mxfmatmul-m4n4k8-b2 dp-mxfmatmul-m4n4k8-b2/main.c 16 16 16)
# Double buffering
add_spatz_test_threeParam(dp-db-fmatmul-4x4vl dp-db-fmatmul-4x4vl/main.c 128 128 128)
add_spatz_test_threeParam(dp-db-fmatmul-4x4vl dp-db-fmatmul-4x4vl/main.c 128 256 128)
add_spatz_test_threeParam(dp-db-fmatmul-4x4vl dp-db-fmatmul-4x4vl/main.c 256 256 256)
