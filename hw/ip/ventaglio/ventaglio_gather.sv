// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Bowen Wang, ETH Zurich
//
// Ventaglio Gather datapath

module ventaglio_gather
  import spatz_pkg::*;
  #(
    parameter int unsigned NarrowDataWidth = 256,
    parameter int unsigned WideDataWidth   = 1024
  ) (
    input  logic       clk_i,
    input  logic       rst_ni,
    input  logic       testmode_i,
    // Read ports -- to VFU    (narrow)
    input  vrf_addr_t  raddr_i,
    input  logic       re_i,
    output vrf_data_t  rdata_o,
    output logic       rvalid_o,
    // Read ports -- to Buffer (wide)

  );