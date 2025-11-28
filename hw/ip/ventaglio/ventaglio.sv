// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Bowen Wang, ETH Zurich
//
// Standalone Gather/Scatter datapath
// 

module ventaglio
  import spatz_pkg::*;
  #(
    parameter int unsigned NrReadPorts  = 1,
    parameter int unsigned NrWritePorts = 1,

    parameter int unsigned NarrowDataWidth = VTGChannelWidth,
    parameter int unsigned WideDataWidth   = VTGChannelWidth * VENTAGLIO_WFACTOR
  ) (
    input  logic            clk_i,
    input  logic            rst_ni,
    input  logic            testmode_i,
    // configs
    // input  logic 	[7:0] 	vtg_mode_i,
    // input  logic 	[7:0] 	vtg_ratio_i,
    // index ports
    // input  vrf_data_t  		vtg_index_i,

    // Slave Write ports
    input  vrf_addr_t  [NrWritePorts-1:0]		waddr_i,
    input  vrf_data_t  [NrWritePorts-1:0]		wdata_i,
    input  logic       [NrWritePorts-1:0]		we_i,
    input  vrf_be_t    [NrWritePorts-1:0]		wbe_i,
    output logic       [NrWritePorts-1:0]		wvalid_o,
    // Slave Read ports
    input  vrf_addr_t  [NrReadPorts-1:0]		raddr_i,
    input  logic       [NrReadPorts-1:0]		re_i,
    output vrf_data_t  [NrReadPorts-1:0]		rdata_o,
    output logic       [NrReadPorts-1:0]		rvalid_o
  );

  /******************************/
  /*           Types            */ 
  /******************************/

  // We current assume one r/w port from VRF
  localparam int unsigned VRF_WD = 0;
  localparam int unsigned VRF_RD = 0;

  // The input address has type `vrf_addr_t`
  // It is used to address `NRVREG * NrWordsPerVector` words
  // Word is represented as `N_FU * ELEN`, is the bandwidth VPU or VLSU comsume
  // In VTL, each channel provide `N_FU * ELEN` bandwidth as well, so it is channel addressable

  // We currently utilized a interleaved address scheme
  // Word 0             -> Channel 0 Row 0
  // Word 1             -> Channel 1 Row 0
  // Word VTGNrChannels -> Channel 0 Row 1 ...


  // `f_channel` function extract the channel index
  function automatic logic [$clog2(VTGNrChannels)-1:0] f_channel(vrf_addr_t addr);
    f_channel = addr[$clog2(VTGNrChannels)-1:0];
  endfunction: f_channel

  // `f_row` function extract the word (row) index within one channel
  function automatic logic [$clog2(VTGNrWordsPerChannel)-1:0] f_row(vrf_addr_t addr);
    f_row = addr[$clog2(VTGNrWordsPerChannel * VTGNrChannels)-1:$clog2(VTGNrChannels)];
  endfunction: f_row

  // In VRF we address bank words, in VTG we address channels
  // VTG access granularity is channel
  typedef logic [$clog2(VTGNrWordsPerChannel)-1:0] vtg_row_addr_t;

  /******************************/
  /*          Signals           */ 
  /******************************/

  // write signals
  // TODO: support more write port to make it more general
  vtg_row_addr_t          [VTGNrChannels-1:0] waddr;
  ventaglio_narrow_data_t [VTGNrChannels-1:0] wdata;
  logic                   [VTGNrChannels-1:0] we;
  ventaglio_narrow_be_t   [VTGNrChannels-1:0] wbe;

  // read signals
  vtg_row_addr_t          [VTGNrChannels-1:0][VTGNrReadPortsPerBank-1:0] raddr;
  ventaglio_narrow_data_t [VTGNrChannels-1:0][VTGNrReadPortsPerBank-1:0] rdata;

  // write mapping 
  logic [VTGNrChannels-1:0][NrWritePorts-1:0] write_request;
  always_comb begin: gen_write_request
    for (int channel = 0; channel < VTGNrChannels; channel++) begin
      for (int port = 0; port < NrWritePorts; port++) begin
        write_request[channel][port] = we_i[port] && f_channel(waddr_i[port]) == channel;

      end
    end
  end: gen_write_request

  always_comb begin : proc_write
    waddr    = '0;
    wdata    = '0;
    we       = '0;
    wbe      = '0;
    wvalid_o = '0;

    for (int unsigned channel = 0; channel < VTGNrChannels; channel++) begin
      if (write_request[channel][VRF_WD]) begin
        waddr[channel]         = f_row(waddr_i[VRF_WD]);
        wdata[channel]         = wdata_i[VRF_WD];
        we[channel]            = 1'b1;
        wbe[channel]           = wbe_i[VRF_WD];
        wvalid_o[VRF_WD]       = 1'b1;
      end 
    end
  end : proc_write

  // read mapping
  logic [VTGNrChannels-1:0][NrReadPorts-1:0] read_request;
  always_comb begin: gen_read_request
    for (int channel = 0; channel < VTGNrChannels; channel++) begin
      for (int port = 0; port < NrReadPorts; port++) begin
        read_request[channel][port] = re_i[port] && f_channel(raddr_i[port]) == channel;
      end
    end
  end: gen_read_request

  always_comb begin : proc_read
    raddr    = '0;
    rvalid_o = '0;
    rdata_o  = 'x;

    for (int unsigned channel = 0; channel < VTGNrChannels; channel++) begin
      if (read_request[channel][VRF_RD]) begin
        raddr[channel][0]    = f_row(raddr_i[VRF_RD]);
        rdata_o[VRF_RD]      = rdata[channel][0];
        rvalid_o[VRF_RD]     = 1'b1;
      end 
    end
  end



  // scatter --> buffer
  // elen_t [VTGNrChannels-1:0][VTGNrBanksPerChannel-1:0] wdara_post_scatter;

  // buffer --> gather
  // elen_t     [VTGNrChannels-1:0][VTGNrBanksPerChannel-1:0]  rdara_pre_gather;
  // vtg_addr_t [VTGNrChannels-1:0][VTGNrReadPortsPerBank-1:0] raddr;

  /******************************/
  /*      Scatter DataPath      */ 
  /******************************/


  /******************************/
  /*           Buffer           */ 
  /******************************/

  // Buffer has `VTGNrChannels` channels
  // Each channel is divided into `N_FU` banks, whose width is `ELEN`
  // In this way, each channel can provide the same bandwidth as the VLSU and VPU
  for (genvar channel = 0; channel < VTGNrChannels; channel++) begin : gen_vtg_channels
    for (genvar bank = 0; bank < N_FU; bank++) begin: gen_vtg_banks
      elen_t [VTGNrReadPortsPerBank-1:0] rdata_int;

      for (genvar port = 0; port < VTGNrReadPortsPerBank; port++) begin: gen_rdata_assignment
        // assign rdara_pre_gather[channel][port][ELEN*bank +: ELEN] = rdata_int[port];
        assign rdata[channel][port][ELEN*bank +: ELEN] = rdata_int[port];
      end

      ventaglio_regfile #(
        .NrReadPorts(VTGNrReadPortsPerBank),
        .NrWords    (VTGNrWordsPerChannel ),
        .WordWidth  (ELEN                 )
      ) i_vtg_vregfile (
        .clk_i     (clk_i                            ),
        .rst_ni    (rst_ni                           ),
        .testmode_i(testmode_i                       ),
        .waddr_i   (waddr[channel]                   ),
        .wdata_i   (wdata[channel][ELEN*bank +: ELEN]),
        .we_i      (we[channel]                      ),
        .wbe_i     (wbe[channel][ELENB*bank +: ELENB]    ),
        .raddr_i   (raddr[channel]                  ),
        .rdata_o   (rdata_int                    )
      );
    end
  end

  /******************************/
  /*      Gather  DataPath      */ 
  /******************************/

endmodule : ventaglio