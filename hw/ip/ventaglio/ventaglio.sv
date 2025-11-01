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
    input  logic            clk_i,
    input  logic            rst_ni,
    input  logic            testmode_i,
    // configs
    input  logic 	[7:0] 	vtg_mode_i,
    input  logic 	[7:0] 	vtg_ratio_i,
    // index ports
    input  vrf_data_t  		vtg_index_i,
    // Write ports
    input  vrf_addr_t  		waddr_i,
    input  vrf_data_t  		wdata_i,
    input  logic       		we_i,
    input  vrf_be_t    		wbe_i,
    output logic       		wvalid_o,
    // Read ports
    input  vrf_addr_t  		raddr_i,
    input  logic       		re_i,
    output vrf_data_t  		rdata_o,
    output logic       		rvalid_o
  );

  /******************************/
  /*           Types            */ 
  /******************************/

  // In VRF we address bank words, in VTG we address channels
  // VTG access granularity is channel
  typedef logic [$clog2(VTGNrWordsPerBanks)-1:0] vtg_addr_t;

  /******************************/
  /*          Signals           */ 
  /******************************/
  // scatter --> buffer
  elen_t [VTGNrChannels-1:0][VTGNrBanksPerChannel-1:0] wdara_post_scatter;

  // buffer --> gather
  elen_t     [VTGNrChannels-1:0][VTGNrBanksPerChannel-1:0]  rdara_pre_gather;
  vtg_addr_t [VTGNrChannels-1:0][VTGNrReadPortsPerBank-1:0] raddr;

  /******************************/
  /*      Scatter DataPath      */ 
  /******************************/


  /******************************/
  /*           Buffer           */ 
  /******************************/
  for (genvar channel = 0; channel < VTGNrChannels; channel++) begin : gen_vtg_channels
    for (genvar bank = 0; bank < N_FU; bank++) begin: gen_vtg_banks
      elen_t [VTGNrReadPortsPerBank-1:0] rdata_int;

      for (genvar port = 0; port < VTGNrReadPortsPerBank; port++) begin: gen_rdata_assignment
        assign rdara_pre_gather[channel][port][ELEN*bank +: ELEN] = rdata_int[port];
      end

      ventaglio_regfile #(
        .NrReadPorts(VTGNrReadPortsPerBank),
        .NrWords    (VTGNrWordsPerBanks   ),
        .WordWidth  (ELEN              )
      ) i_vtg_vregfile (
        .clk_i     (clk_i                        ),
        .rst_ni    (rst_ni                       ),
        .testmode_i(testmode_i                   ),
        .waddr_i   (waddr[bank]                  ),
        .wdata_i   (wdata[bank][ELEN*cut +: ELEN]),
        .we_i      (we[bank]                     ),
        .wbe_i     (wbe[bank][ELENB*cut +: ELENB]),
        .raddr_i   (raddr[bank]                  ),
        .rdata_o   (rdata_int                    )
      );
    end
  end

  /******************************/
  /*      Gather  DataPath      */ 
  /******************************/