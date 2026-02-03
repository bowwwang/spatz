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

    // Spatz request
    input  spatz_req_t       spatz_req_i,
    input  logic             spatz_req_valid_i,
    output logic             spatz_req_ready_o,
    input  logic             spatz_vfu_req_ready_i,
    input  logic             vtl_index_preload_valid_i,
    // VTL response
    output logic             vtl_rsp_valid_o,
    output vsldu_rsp_t       vtl_rsp_o,
    // VFU response
    input  logic             vfu_rsp_valid_i,
    input  vfu_rsp_t         vfu_rsp_i,
    // Slave Write ports
    input  vrf_addr_t  [NrWritePorts-1:0]		waddr_i,
    input  vrf_data_t  [NrWritePorts-1:0]		wdata_i,
    input  logic       [NrWritePorts-1:0]		we_i,
    input  vrf_be_t    [NrWritePorts-1:0]		wbe_i,
    output logic       [NrWritePorts-1:0]		wvalid_o,
    input  logic       [NrWritePorts-1:0]   wscatter_en_i,
    // Slave Read ports
    input  vrf_addr_t  [NrReadPorts-1:0]		raddr_i,
    input  logic       [NrReadPorts-1:0]		re_i,
    output vrf_data_t  [NrReadPorts-1:0]		rdata_o,
    output logic       [NrReadPorts-1:0]		rvalid_o,
    input  logic       [NrReadPorts-1:0]    rgather_en_i,

    // Master VRF interface
    output vrf_addr_t                       vrf_waddr_o,
    output vrf_data_t                       vrf_wdata_o,
    output logic                            vrf_we_o,
    output vrf_be_t                         vrf_wbe_o,
    input  logic                            vrf_wvalid_i,
    output spatz_id_t  [1:0]                vrf_id_o,
    output vrf_addr_t                       vrf_raddr_o,
    output logic                            vrf_re_o,
    input  vrf_data_t                       vrf_rdata_i,
    input  logic                            vrf_rvalid_i
  );

// Include FF
`include "common_cells/registers.svh"
  
  ///////////////////////
  //  Operation queue  //
  ///////////////////////

  spatz_req_t spatz_req_d;

  spatz_req_t spatz_req;
  logic       spatz_req_valid;
  logic       spatz_req_ready;



  spill_register #(
    .T(spatz_req_t)
  ) i_operation_queue (
    .clk_i  (clk_i                                                                        ),
    .rst_ni (rst_ni                                                                       ),
    .data_i (spatz_req_d                                                                  ),
    .valid_i(spatz_req_valid_i && spatz_req_i.ex_unit == VFU && spatz_req_i.op_vtl.use_vtl), // currently we only care about instructions with sc/ga
    .ready_o(spatz_req_ready_o                                                            ),
    .data_o (spatz_req                                                                    ),
    .valid_o(spatz_req_valid                                                              ),
    .ready_i(spatz_req_ready                                                              )
  );


  always_comb begin : proc_spatz_req
    spatz_req_d = spatz_req_i;
  end

  /******************************/
  /*       State Handler        */ 
  /******************************/
  // Currently running instructions
  logic [NrParallelInstructions-1:0] running_d, running_q;
  `FF(running_q, running_d, '0)

  // New instruction
  logic new_vtl_request_d, new_vtl_request_q;
  assign new_vtl_request_d = spatz_req_valid && !running_q[spatz_req.id];
  `FF(new_vtl_request_q, new_vtl_request_d, '0)

  always_comb begin : proc_vtl_state
    running_d = running_q;
    spatz_req_ready  = 1'b0;
    // Operation queue is ready for the new instruction if non is in process
    spatz_req_ready = !spatz_req_valid;

    // A new spatz_req is recieved
    if (new_vtl_request_d) begin
      running_d[spatz_req.id] = 1'b1; // mark the instruction as running
    end

    // VTL serve as the co-unit for VPU -- sync the status
    if (running_q[vfu_rsp_i.id] && vfu_rsp_valid_i) begin // VFU finished this instruciton 
      running_d[vfu_rsp_i.id] = 1'b0;  // mark the instruction as finished
    end
    if (spatz_vfu_req_ready_i) begin
      spatz_req_ready         = 1'b1;
    end
  end 

  //////////////////////////
  //  Read Index Request  //
  //////////////////////////

  // Vector register file counter signals for index
  logic      vreg_idx_counter_en;
  vrf_addr_t vreg_idx_counter_d;
  vrf_addr_t vreg_idx_counter_q;
  `FF(vreg_idx_counter_q, vreg_idx_counter_d, '0)

  logic index_valid_d, index_valid_q;
  `FF(index_valid_q, index_valid_d, '0)

  // naive index valid logic handling
  always_comb begin : proc_index_valid
    index_valid_d = index_valid_q;
    if (vrf_rvalid_i) begin 
      // VALID when a new index is available from read
      index_valid_d = 1'b1;
    end 
    if (spatz_vfu_req_ready_i) begin 
      // INVALID when VFU has a new req to process 
      index_valid_d = 1'b0;
    end
  end 

  always_comb begin : proc_idx_counter
    vreg_idx_counter_d = vreg_idx_counter_q;
    if (vreg_idx_counter_en) begin
      vreg_idx_counter_d = vreg_idx_counter_q + 1'b1;
    end
    if (running_q[vfu_rsp_i.id] && vfu_rsp_valid_i) begin // VFU finished this instruciton 
      vreg_idx_counter_d = '0;
    end
  end

  // `vidx` is the VRF id storing the indices
  // This information is extracted from VLX or VFXMACC instructions 
  // We cover two scenarios:
  // 1. A new index is loaded with VLX                           --> use VLX info to load index 
  // 2. Each VFXMACC operation requires more than 'VRFWordWidth' --> use VFX info to load index 

  // NOTE: Current Index loading solely depends on `vlx` info
  vreg_t vidx_d, vidx_q;
  `FF(vidx_q, vidx_d, '0)
  // One step buffer in case the previous index request does not finish
  vreg_t vidx_buf_d, vidx_buf_q;
  `FF(vidx_buf_q, vidx_buf_d, '0)

  always_comb begin : proc_idx_addr_gen
    vidx_d     = vidx_q;
    vidx_buf_d = vidx_buf_q;

    // A new VLX instruction received
    if (spatz_req_valid_i && spatz_req_i.op_vtl.is_load_idx) begin
      // No on-fly index req --> update vidx, else keep the old req
      vidx_d     = (vrf_re_o) ? vidx_q : spatz_req_i.op_vtl.old_vd;
      // buffer the vidx info
      vidx_buf_d = spatz_req_i.op_vtl.old_vd;
    end

    // Previous index req is done, and the buffered vidx is not the same as the previous vidx
    if (!vrf_re_o && vidx_q != vidx_buf_q) begin 
      // update vidx
      vidx_d = vidx_buf_q;
    end 

    // NOTE: Currently not used, but should used for instruction requests more than one idx beat
    // For VLXMACC instructions
    // if (new_vtl_request_d) begin 
    //   if (spatz_req.op_vtl.gather_vs1) begin
    //     vidx_d = spatz_req.vs1;
    //   end else if (spatz_req.op_vtl.gather_vs2) begin 
    //     vidx_d = spatz_req.vs2;
    //   end else if (spatz_req.op_vtl.gather_vd || spatz_req.op_vtl.scatter_vd) begin
    //     vidx_d = spatz_req.vs1; // bowwang: we now use vid(weight) to find vid(index) in controller
    //   end
    // end 

    // address generation
    vrf_raddr_o     = {vidx_q, $clog2(NrWordsPerVector)'(1'b0)} + vreg_idx_counter_q;
    if (vreg_idx_counter_en)
      vrf_raddr_o     = {vidx_q, $clog2(NrWordsPerVector)'(1'b0)} + vreg_idx_counter_q + 1'b1;
  end

  // Logic to issue the read request for loading indices
  // 1. When no gather/scatter is operating, and controller informed the index is ready
  // 2. TODO: When the current indices is depleted

  logic index_preload_valid_d, index_preload_valid_q;
  `FF(index_preload_valid_q, index_preload_valid_d, '0)

  always_comb begin 
    index_preload_valid_d = index_preload_valid_q;
    if (vtl_index_preload_valid_i) begin // controller informed the index is ready
      index_preload_valid_d = 1'b1;
    end 
    // we read back an index vector 
    // TODO: this is not correct!!! We do not consider the indices depletion scenario
    if (vrf_rvalid_i) begin 
      index_preload_valid_d = 1'b0;
    end 
  end 

  logic index_preload;
  // Nothing is running while the index is ready --> preload
  assign index_preload = !(|running_q) && index_preload_valid_q;
  // Have a valid operation, but no valid index
  assign index_load    = spatz_req_valid && !index_valid_q;

  assign vrf_re_o      = index_preload | index_load;
  // This is not required, sudo ID used
  // In controller, we do not check the dependency issued from VTL
  assign vrf_id_o[0]     = spatz_req.id;

  /******************************/
  /*       Index Buffer         */ 
  /******************************/
  vrf_data_t index_d, index_q;
  `FF(index_q, index_d, 'b0);

  assign index_d      = vrf_rvalid_i ? vrf_rdata_i : index_q;

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

  // signals for gather or scatter
  logic is_gather, is_scatter;
  logic gather_done;
  `FF(gather_done, spatz_vfu_req_ready_i&&spatz_req_valid, '0)

  // `rgather_en_i` and `wscatter_en_i` signals are attached in VRF bypass logic
  always_comb begin
    is_gather  = (spatz_req.op_vtl.gather_vd || spatz_req.op_vtl.gather_vs1 || spatz_req.op_vtl.gather_vs2) && rgather_en_i;
    is_scatter = spatz_req.op_vtl.scatter_vd && wscatter_en_i;
  end

  // write signals
  vtg_row_addr_t          [VTGNrChannels-1:0] waddr;
  ventaglio_narrow_data_t [VTGNrChannels-1:0] wdata;
  logic                   [VTGNrChannels-1:0] we;
  ventaglio_narrow_be_t   [VTGNrChannels-1:0] wbe;

  // read signals
  vtg_row_addr_t          [VTGNrChannels-1:0][VTGNrReadPortsPerBank-1:0] raddr;
  ventaglio_narrow_data_t [VTGNrChannels-1:0][VTGNrReadPortsPerBank-1:0] rdata;

  // write mapping 
  logic                    [VTGNrChannels-1:0][NrWritePorts-1:0] write_request;
  logic [VTGNrChannels-1:0][VTGNrChannels-1:0][NrWritePorts-1:0] scatter_write_request;

  // post scatter signals
  logic                    [VTGNrChannels-1:0][NrWritePorts-1:0] scatter_we;
  vrf_addr_t               [VTGNrChannels-1:0][NrWritePorts-1:0] scatter_waddr;
  vrf_data_t               [VTGNrChannels-1:0][NrWritePorts-1:0] scatter_wdata;
  vrf_be_t                 [VTGNrChannels-1:0][NrWritePorts-1:0] scatter_wbe;

  logic                    [VTGNrChannels-1:0][NrWritePorts-1:0] scatter_wvalid;

  logic                    [NrWritePorts-1:0] post_scatter_wvalid;

  always_comb begin: gen_write_request
    for (int channel = 0; channel < VTGNrChannels; channel++) begin
      for (int port = 0; port < NrWritePorts; port++) begin
        write_request[channel][port] = we_i[port] && f_channel(waddr_i[port]) == channel && !is_scatter;
      end
    end
    // scatter requests
    for (int b_channel = 0; b_channel < VTGNrChannels; b_channel++) begin   // channels from the bank side
      for (int s_channel = 0; s_channel < VTGNrChannels; s_channel++) begin // channels from the scatter datapath side 
        for (int port = 0; port < NrWritePorts; port++) begin
          scatter_write_request[b_channel][s_channel][port] = scatter_we[s_channel][port] && f_channel(scatter_waddr[s_channel][port]) == b_channel && is_scatter;
        end
      end 
    end
  end: gen_write_request

  always_comb begin : proc_write
    waddr          = '0;
    wdata          = '0;
    we             = '0;
    wbe            = '0;
    wvalid_o       = '0;
    scatter_wvalid = '0;

    if (!is_scatter) begin // priority 1: normal requests
      for (int unsigned channel = 0; channel < VTGNrChannels; channel++) begin
        if (write_request[channel][VRF_WD]) begin
          waddr[channel]         = f_row(waddr_i[VRF_WD]);
          wdata[channel]         = wdata_i[VRF_WD];
          we[channel]            = 1'b1;
          wbe[channel]           = wbe_i[VRF_WD];
          wvalid_o[VRF_WD]       = 1'b1;
        end 
      end
    end else begin // priority 2: scatter requests
      for (int unsigned b_channel = 0; b_channel < VTGNrChannels; b_channel++) begin   // channels from the bank side
        for (int unsigned s_channel = 0; s_channel < VTGNrChannels; s_channel++) begin // channels from the scatter datapath side
          if (scatter_write_request[b_channel][s_channel][VRF_WD]) begin
            waddr[b_channel]                  = f_row(scatter_waddr[s_channel][VRF_WD]);
            wdata[b_channel]                  = scatter_wdata[s_channel][VRF_WD];
            we[b_channel]                     = 1'b1;
            wbe[b_channel]                    = scatter_wbe[s_channel][VRF_WD];
            scatter_wvalid[s_channel][VRF_WD] = 1'b1;
          end
        end
      end
      // assign the scatter write valid signal to the output 
      wvalid_o[VRF_WD]     = post_scatter_wvalid[0];
    end 

  end : proc_write

  // read mapping

  // read_request: non-gather request signals
  // gathered_read_request: gather request signals
  logic [VTGNrChannels-1:0][NrReadPorts-1:0] read_request;

  logic [VTGNrChannels-1:0][VTGNrChannels-1:0][NrReadPorts-1:0] gather_read_request;
  logic                    [VTGNrChannels-1:0][NrReadPorts-1:0] gather_re;
  vrf_addr_t               [VTGNrChannels-1:0][NrReadPorts-1:0] gather_raddr;

  vrf_data_t               [VTGNrChannels-1:0][NrReadPorts-1:0] gather_rdata;
  logic                    [VTGNrChannels-1:0][NrReadPorts-1:0] gather_rvalid;

  // post gather signals
  vrf_data_t               [NrReadPorts-1:0] post_gather_rdata;
  logic                    [NrReadPorts-1:0] post_gather_rvalid;

  always_comb begin: gen_read_request
    // normal requests
    for (int channel = 0; channel < VTGNrChannels; channel++) begin
      for (int port = 0; port < NrReadPorts; port++) begin
        read_request[channel][port] = re_i[port] && f_channel(raddr_i[port]) == channel && !(is_gather);
      end
    end
    // gathered requests
    for (int b_channel = 0; b_channel < VTGNrChannels; b_channel++) begin   // channels from the bank side
      for (int g_channel = 0; g_channel < VTGNrChannels; g_channel++) begin // channels from the gather datapath side 
        for (int port = 0; port < NrReadPorts; port++) begin
          gather_read_request[b_channel][g_channel][port] = gather_re[g_channel][port] && f_channel(gather_raddr[g_channel][port]) == b_channel && is_gather;
        end
      end 
    end
  end: gen_read_request

  // this should be extended for gather
  always_comb begin : proc_read
    raddr    = '0;
    rvalid_o = '0;
    rdata_o  = 'x;

    gather_rvalid = '0;
    gather_rdata  = 'x;

    if (!is_gather) begin // priority 1: normal read requests
      for (int unsigned b_channel = 0; b_channel < VTGNrChannels; b_channel++) begin // channels from the bank side
        if (read_request[b_channel][VRF_RD]) begin
          raddr[b_channel][0]    = f_row(raddr_i[VRF_RD]);
          rdata_o[VRF_RD]      = rdata[b_channel][0];
          rvalid_o[VRF_RD]     = 1'b1;
        end
      end
    end else if (is_gather) begin // priority 2: gather accesses
      for (int unsigned b_channel = 0; b_channel < VTGNrChannels; b_channel++) begin // channels from the bank side
        for (int unsigned g_channel = 0; g_channel < VTGNrChannels; g_channel++) begin // channels from the gather datapath side
          if (gather_read_request[b_channel][g_channel][VRF_RD]) begin
            raddr[b_channel][0]                  = f_row(gather_raddr[g_channel][VRF_RD]);
            gather_rdata[g_channel][VRF_RD]      = rdata[b_channel][0];
            // gather_rvalid[g_channel][VRF_RD]     = 1'b1;
            gather_rvalid[g_channel][VRF_RD]     = index_valid_q;
          end
        end
      end
      // assign the gathered data to the output 
      rdata_o[VRF_RD]      = post_gather_rdata[0];
      rvalid_o[VRF_RD]     = post_gather_rvalid[0];
    end 
  end

  /******************************/
  /*      Scatter DataPath      */ 
  /******************************/
  ventaglio_scatter #(
    .NarrowDataWidth (NarrowDataWidth),
    .WideDataWidth   (WideDataWidth)
  ) i_vtl_scatter (
    .clk_i       (clk_i),
    .rst_ni      (rst_ni),
    .testmode_i  (testmode_i),
    // narrow ports
    .waddr_i     (waddr_i),
    .wdata_i     (wdata_i),
    .we_i        (we_i && is_scatter),
    .wbe_i       (wbe_i),
    .wvalid_o    (post_scatter_wvalid[0]),
    // wide ports
    .waddr_o     (scatter_waddr),
    .wdata_o     (scatter_wdata),
    .we_o        (scatter_we),
    .wbe_o       (scatter_wbe),
    .wvalid_i    (scatter_wvalid),
    // control 
    // TODO: it is better to implement as a counter to track
    .scatter_done_i      (!is_scatter),
    // controls
    .index_i       (index_q                ),
    .vtl_cfg_i     (spatz_req.op_vtl.sp_cfg),
    .index_valid_i (index_valid_q) // meaning a new read data available
  );


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

  ventaglio_gather #(
    .NarrowDataWidth (NarrowDataWidth),
    .WideDataWidth   (WideDataWidth)
  ) i_vtl_gather (
    .clk_i       (clk_i),
    .rst_ni      (rst_ni),
    .testmode_i  (testmode_i),
    // narrow ports
    .raddr_i     (raddr_i              ),
    .re_i        (re_i && is_gather    ),
    .rdata_o     (post_gather_rdata[0] ),
    .rvalid_o    (post_gather_rvalid[0]),
    // wide ports
    .raddr_o     (gather_raddr         ),
    .re_o        (gather_re            ),
    .rdata_i     (gather_rdata         ),
    .rvalid_i    (gather_rvalid        ),
    // control
    .gather_done_i(gather_done          ),
    // index cfg
    .index_i     (index_q              ),
    .vtl_cfg_i   (spatz_req.op_vtl.sp_cfg),
    .load_index_o(vreg_idx_counter_en)
  );

  /******************************/
  /*      Write Requests        */ 
  /******************************/

  assign vrf_we_o    = '0;
  assign vrf_wbe_o   = '0;
  assign vrf_waddr_o = '0;
  assign vrf_wdata_o = '0;



endmodule : ventaglio