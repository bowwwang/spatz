// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Bowen Wang, ETH Zurich
//
// Standalone Gather/Scatter datapath

module ventaglio
  import spatz_pkg::*;
  #(
    parameter int unsigned NarrowDataWidth = 256,
    parameter int unsigned WideDataWidth   = 1024
  ) (
    input  logic                         clk_i,
    input  logic                         rst_ni,
    input  logic                         testmode_i,
    // Write ports
    input  vrf_addr_t  waddr_i,
    input  vrf_data_t  wdata_i,
    input  logic       we_i,
    input  vrf_be_t    wbe_i,
    output logic       wvalid_o,
    // Read ports
    input  vrf_addr_t  raddr_i,
    input  logic       re_i,
    output vrf_data_t  rdata_o,
    output logic       rvalid_o
  );


  /******************************/
  /*      Scatter DataPath      */ 
  /******************************/

  /******************************/
  /*           Buffer           */ 
  /******************************/

  /******************************/
  /*      Gather  DataPath      */ 
  /******************************/