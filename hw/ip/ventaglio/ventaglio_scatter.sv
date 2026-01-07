// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Bowen Wang, ETH Zurich
//
// Ventaglio Scatter datapath

module ventaglio_scatter 
  import spatz_pkg::*;
  import   vtl_pkg::*;
  #(
    parameter int unsigned NarrowDataWidth = VTGChannelWidth,
    parameter int unsigned WideDataWidth   = VTGChannelWidth * VENTAGLIO_WFACTOR

  ) (
    input  logic       							clk_i,
    input  logic       							rst_ni,
    input  logic       							testmode_i,
    // write ports -- from VFU  (narrow)
    input  vrf_addr_t                           waddr_i,
    input  vrf_data_t                           wdata_i,
    input  logic                                we_i,
    input  vrf_be_t                     		wbe_i,
    output logic                                wvalid_o,
    // write ports -- to Buffer (wide)
    output vrf_addr_t  [VENTAGLIO_WFACTOR-1:0]  waddr_o,
    output vrf_data_t  [VENTAGLIO_WFACTOR-1:0]  wdata_o,
    output logic       [VENTAGLIO_WFACTOR-1:0]  we_o,
    output vrf_be_t    [VENTAGLIO_WFACTOR-1:0] 	wbe_o,
    input  logic       [VENTAGLIO_WFACTOR-1:0]  wvalid_i,
    // control
    input  logic                                scatter_done_i,
    // index
    input  vrf_data_t                           index_i,
    input  sp_cfg_t 							vtl_cfg_i,
    input  logic                                new_scatter_request
  );

  // Include FF
  `include "common_cells/registers.svh"

  ////////////////////////////
  //     request logics     //
  ////////////////////////////
  vrf_addr_t waddr;

  always_comb begin : proc_write_addr
  	waddr   = '0;
  	waddr_o = '0;
  	we_o    = '0;
  	case (vtl_cfg_i.sp_cfg_ratio)
  		SP_RATIO_050: begin // use the first two port to send out requests
  			waddr = waddr_i << 1;
  			waddr_o[0] = {waddr[$clog2(NrVRFWords)-1:1], 1'b0};
			waddr_o[1] = {waddr[$clog2(NrVRFWords)-1:1], 1'b1};
			if (we_i)
				we_o[1:0]  = 2'b11;
  		end 

  		SP_RATIO_025: begin
  			waddr = waddr_i << 2;
  			waddr_o[0] = {waddr[$clog2(NrVRFWords)-1:2], 2'b00};
			waddr_o[1] = {waddr[$clog2(NrVRFWords)-1:2], 2'b01};
			waddr_o[2] = {waddr[$clog2(NrVRFWords)-1:2], 2'b10};
			waddr_o[3] = {waddr[$clog2(NrVRFWords)-1:2], 2'b11};
			if (we_i)
				we_o = '1;
  		end
  		default : waddr = waddr_i;
  	endcase
  end // proc_write_addr

  ////////////////////////////
  //     scatter signals    //
  ////////////////////////////

  logic [3:0] beat_cnt_d, beat_cnt_q;
  `FF(beat_cnt_q, beat_cnt_d, '0)

  // we check the last bit of req addr
  // if it changes, it's a new beat
  logic addr_last_bit_d, addr_last_bit_q;
  `FF(addr_last_bit_q, addr_last_bit_d, '0)

  logic beat_cnt_en;
  assign addr_last_bit_d = waddr_i[0];
  assign beat_cnt_en = (!scatter_done_i) && (addr_last_bit_d ^ addr_last_bit_q);

  localparam int unsigned NrEffElePerBlk  = 2; 
  localparam int unsigned NrElePerBlk     = 4; 

  localparam int unsigned IdxWidth        = 2;
  localparam int unsigned EleWidth        = 32;
  localparam int unsigned EleWidthB       = EleWidth / 8;

  localparam int unsigned NrBlksPerBeat   = VRFWordWidth / (NrEffElePerBlk * EleWidth);
  // localparam int unsigned NrBeatsPerInput = VRFWordWidth / (NrBlksPerBeat * NrEffElePerBlk * IdxWidth);
  localparam int unsigned NrBeatsPerInput =  EleWidth / IdxWidth;

  always_comb begin
  	beat_cnt_d   = beat_cnt_q;
  	if (beat_cnt_en) begin
  		if (beat_cnt_q == NrBeatsPerInput-1) begin
  			beat_cnt_d   = '0;
  		end else begin
  			beat_cnt_d = beat_cnt_q + 1'b1;
  		end
  	end
  	if (scatter_done_i) beat_cnt_d   = '0;
  end


  ////////////////////////////
  //     index buffer       //
  ////////////////////////////

  // the write rewuests has at least one cycle delay comparing with the read
  // so we need to buffer the previous/current index in case read requests refresh it
  // the index will update when 
  // 1. this is a new request, we have to sync with the init index
  // 2. the scatter operations exhaust the buffer index

  vrf_data_t index_d, index_q;
  `FF(index_q, index_d, 'b0);

  always_comb begin
  	index_d = index_q;
  	if ( (new_scatter_request && beat_cnt_q == '0) || beat_cnt_q == NrBeatsPerInput - 1) begin 
  		index_d = index_i;
  	end 
  end

  ////////////////////////////
  //     scatter logic      //
  ////////////////////////////

  // organize the index into blks and beats
  logic [NrBeatsPerInput-1:0][NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][IdxWidth-1:0] idx;
  for (genvar beat = 0; beat < NrBeatsPerInput; beat++) begin
  	for (genvar	blk = 0; blk < NrBlksPerBeat; blk++) begin 
  		for (genvar ele = 0; ele < NrEffElePerBlk; ele++) begin
  			assign idx[beat][blk][ele] = index_q[beat*NrBlksPerBeat*NrEffElePerBlk*IdxWidth + blk*NrEffElePerBlk*IdxWidth + ele*IdxWidth +: IdxWidth];
  		end
  	end 
  end

  // organize the pre-scatter data into blocks
  logic [NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][EleWidth-1:0]  wdata_pre_scatter;
  logic [NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][EleWidthB-1:0] wbe_pre_scatter;
  for (genvar blk = 0; blk < NrBlksPerBeat; blk++) begin
  	for (genvar ele = 0; ele < NrEffElePerBlk; ele++) begin
  		assign wdata_pre_scatter[blk][ele] = wdata_i[blk*NrEffElePerBlk*EleWidth + ele*EleWidth +: EleWidth];
  		assign wbe_pre_scatter[blk][ele]   = wbe_i[blk*NrEffElePerBlk*EleWidthB + ele*EleWidthB +: EleWidthB];
  	end
  end

  // scatter the effective elements to the corresponding slots in the flatten_wdata
  logic [NrBlksPerBeat-1:0][NrElePerBlk-1:0][EleWidth-1:0]  wdata_post_scatter;
  logic [NrBlksPerBeat-1:0][NrElePerBlk-1:0][EleWidthB-1:0] wbe_post_scatter;
  always_comb begin
  	wdata_post_scatter = 'x;
  	wbe_post_scatter   = '0;
  	for (int blk = 0; blk < NrBlksPerBeat; blk++) begin
	  	for (int ele = 0; ele < NrEffElePerBlk; ele++) begin
	  		wdata_post_scatter[blk][ idx[beat_cnt_d][blk][ele] ] = wdata_pre_scatter[blk][ele];
	  		wbe_post_scatter[blk][ idx[beat_cnt_d][blk][ele] ]   = wbe_pre_scatter[blk][ele];
	  	end
	  end
  end

  // flatten the post_scatter wdata
  logic [VRFWordWidth*2-1:0]  flatten_wdata;
  logic [VRFWordBWidth*2-1:0] flatten_wbe;
  for (genvar blk = 0; blk < NrBlksPerBeat; blk++) begin
  	for (genvar ele = 0; ele < NrElePerBlk; ele++) begin
  		assign flatten_wdata[blk*NrElePerBlk*EleWidth + ele*EleWidth +: EleWidth]  = wdata_post_scatter[blk][ele];
  		assign flatten_wbe[blk*NrElePerBlk*EleWidthB + ele*EleWidthB +: EleWidthB] = wbe_post_scatter[blk][ele];
  	end
  end

  // assign the flattened write output data to each channel
  for (genvar channel=0; channel < 2; channel++) begin
  	assign wdata_o[channel] = flatten_wdata[channel*VRFWordWidth +: VRFWordWidth];
  	assign wbe_o[channel]   = flatten_wbe[channel*VRFWordBWidth +: VRFWordBWidth];
  	// temp to pass lint
  	assign wdata_o[channel+2] = '0;
  	assign wbe_o[channel+2]   = '0;
  end

  // handle the valid signals
  assign wvalid_o = (vtl_cfg_i.sp_cfg_ratio == SP_RATIO_050) ? &wvalid_i[1:0] : &wvalid_i;


endmodule : ventaglio_scatter