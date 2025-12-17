// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Author: Bowen Wang, ETH Zurich
//
// Ventaglio Gather datapath

module ventaglio_gather 
  import spatz_pkg::*;
  import   vtl_pkg::*;
  #(
    parameter int unsigned NarrowDataWidth = VTGChannelWidth,
    parameter int unsigned WideDataWidth   = VTGChannelWidth * VENTAGLIO_WFACTOR

  ) (
    input  logic       							clk_i,
    input  logic       							rst_ni,
    input  logic       							testmode_i,
    // Read ports -- to VFU    (narrow)
    input  vrf_addr_t                           raddr_i,
    input  logic                                re_i,
    output vrf_data_t                           rdata_o,
    output logic                                rvalid_o,
    // Read ports -- from Buffer (wide)
    output vrf_addr_t  [VENTAGLIO_WFACTOR-1:0]  raddr_o,
    output logic       [VENTAGLIO_WFACTOR-1:0]  re_o,
    input  vrf_data_t  [VENTAGLIO_WFACTOR-1:0]  rdata_i,
    input  logic       [VENTAGLIO_WFACTOR-1:0]  rvalid_i,
    // control
    input  logic                                gather_done_i, 
    // index
    input  vrf_data_t                           index_i,
    input  sp_cfg_t 							vtl_cfg_i,
    output logic                                load_index_o
  );

  // Include FF
  `include "common_cells/registers.svh"

  ////////////////////////////
  //     request logics     //
  ////////////////////////////
  vrf_addr_t raddr;

  always_comb begin : proc_read_addr
  	raddr    = '0;
  	raddr_o  = '0;
  	re_o     = '0;
  	case (vtl_cfg_i.sp_cfg_ratio)
  		SP_RATIO_050: begin 
  			// one read request will map to 2 channels
  			raddr = raddr_i << 1;
			raddr_o[0] = {raddr[$clog2(NrVRFWords)-1:1], 1'b0};
			raddr_o[1] = {raddr[$clog2(NrVRFWords)-1:1], 1'b1};
			if (re_i)
				re_o[1:0] = 2'b11;
  		end 
  		SP_RATIO_025: begin 
  			raddr = raddr_i << 2;
  			raddr_o[0] = {raddr[$clog2(NrVRFWords)-1:2], 2'b00};
			raddr_o[1] = {raddr[$clog2(NrVRFWords)-1:2], 2'b01};
			raddr_o[2] = {raddr[$clog2(NrVRFWords)-1:2], 2'b10};
			raddr_o[3] = {raddr[$clog2(NrVRFWords)-1:2], 2'b11};
			if (re_i)
				re_o = '1;
		end 
  		default:      raddr = raddr_i;      // need to consider more cased
  	endcase // vtl_cfg_i.sp_cfg_ratio
  end // proc_read_addr


  ////////////////////////////
  //      gather logic      //
  ////////////////////////////

  // a 4-bit counter to track the requests
  // We support LMUL=8, so maximum we can have 16 beats
  logic [3:0] beat_cnt_d, beat_cnt_q;
  `FF(beat_cnt_q, beat_cnt_d, '0)

  // we check the last bit of req addr
  // if it changes, it's a new beat
  logic addr_last_bit_d, addr_last_bit_q;
  `FF(addr_last_bit_q, addr_last_bit_d, '0)

  logic beat_cnt_en;
  assign addr_last_bit_d = raddr_i[0];
  assign beat_cnt_en = (!gather_done_i) && (addr_last_bit_d ^ addr_last_bit_q);

  // progress counter (we need to use beat_cnt_d to select effective index chunk)
  // assign beat_cnt_d = (beat_cnt_en) ? beat_cnt_q + 1 : beat_cnt_q;

  // TODO: support more format
  // For testing, we focus on 16-bit ele and 2-bit index (2:4)
  // whcih means we are gathering 16 elements per 256-bit, and each has 2-bit index

  localparam int unsigned NrEffElePerBlk  = 2; 
  localparam int unsigned NrElePerBlk     = 4; 

  localparam int unsigned IdxWidth        = 2;
  localparam int unsigned EleWidth        = 32;

  localparam int unsigned NrBlksPerBeat   = VRFWordWidth / (NrEffElePerBlk * EleWidth);
  // localparam int unsigned NrBeatsPerInput = VRFWordWidth / (NrBlksPerBeat * NrEffElePerBlk * IdxWidth);
  localparam int unsigned NrBeatsPerInput =  EleWidth / IdxWidth;

  // handle the index loading 
  always_comb begin
  	load_index_o = 0;
  	beat_cnt_d   = beat_cnt_q;
  	if (beat_cnt_en) begin
  		if (beat_cnt_q == NrBeatsPerInput-1) begin
  			beat_cnt_d   = '0;
  			load_index_o = 1;
  		end else begin
  			beat_cnt_d = beat_cnt_q + 1'b1;
  		end
  	end
  	if (gather_done_i) beat_cnt_d   = '0;
  end

  logic [NrBeatsPerInput-1:0][NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][IdxWidth-1:0] idx;
  for (genvar beat = 0; beat < NrBeatsPerInput; beat++) begin
  	for (genvar	blk = 0; blk < NrBlksPerBeat; blk++) begin 
  		for (genvar ele = 0; ele < NrEffElePerBlk; ele++) begin
  			assign idx[beat][blk][ele] = index_i[beat*NrBlksPerBeat*NrEffElePerBlk*IdxWidth + blk*NrEffElePerBlk*IdxWidth + ele*IdxWidth +: IdxWidth];
  		end
  	end 
  end

  // flatten the read data(here we once again focus on 2:4)
  logic [VRFWordWidth*2-1:0] flatten_rdata;
  for (genvar channel=0; channel < 2; channel++) begin
  	assign flatten_rdata[channel*VRFWordWidth +: VRFWordWidth] = rdata_i[channel];
  end

  // organize the pre-gather data into blocks
  logic [NrBlksPerBeat-1:0][NrElePerBlk-1:0][EleWidth-1:0] rdata_pre_gather;
  for (genvar blk = 0; blk < NrBlksPerBeat; blk++) begin
  	for (genvar ele = 0; ele < NrElePerBlk; ele++) begin
  		assign rdata_pre_gather[blk][ele] = flatten_rdata[blk*NrElePerBlk*EleWidth + ele*EleWidth +: EleWidth];
  	end
  end

  // organize the post-gather data into blocks
  logic [NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][EleWidth-1:0] rdata_post_gather;
  for (genvar blk = 0; blk < NrBlksPerBeat; blk++) begin
  	for (genvar ele = 0; ele < NrEffElePerBlk; ele++) begin
  		assign rdata_post_gather[blk][ele] = rdata_pre_gather[blk][ idx[beat_cnt_d][blk][ele] ];
  		assign rdata_o[blk*NrEffElePerBlk*EleWidth + ele*EleWidth +: EleWidth] = rdata_post_gather[blk][ele];
  	end
  end

  // valid signal propogate 
  assign rvalid_o = (vtl_cfg_i.sp_cfg_ratio == SP_RATIO_050) ? &rvalid_i[1:0] : &rvalid_i;


endmodule : ventaglio_gather