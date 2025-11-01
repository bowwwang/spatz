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
    parameter int unsigned NarrowDataWidth 	= 256,
    parameter int unsigned WideDataWidth   	= 1024,
    parameter vtg_elemw_e  Width            = EW16,
    // do not touch 
    parameter int unsigned WidenRatio       = WideDataWidth / NarrowDataWidth,

  ) (
    input  logic       									clk_i,
    input  logic       									rst_ni,
    input  logic       									testmode_i,
    // Read ports -- to VFU    (narrow)
    output logic 				 [NarrowDataWidth-1:0]  rdata_o,
    // Read ports -- from Buffer (wide)
    input  logic [WidenRatio-1:0][NarrowDataWidth-1:0] 	rdata_i,
    // index
    input  vtg_ratio_e 									vtg_mode_i,
    input  vrf_data_t  									vtg_index_i
  );

  //////////////////
  //     size     //
  //////////////////

  localparam int unsigned EleWidth = (Width == EW32) ? 32 : 
  									 (Width == EW16) ? 16 : 
  									 (Width == EW8 ) ? 8  : 16;

  localparam int unsigned NumNarrowEle = NarrowDataWidth / EleWidth;
  localparam int unsigned NumWideEle   = WideDataWidth   / EleWidth;

  // index to address output vector
  localparam int unsigned WideIndexWidth   = (NumWideEle <= 1) ? 1 : $clog2(NumWideEle);

  function automatic int unsigned idx_bits_per_ele(vtg_ratio_e mode);
  	case(mode)
  		VTG_N1_M2: return 1;
  		VTG_N1_M4,
  		VTG_N2_M4: return 2;
  		default:   return 2;
  	endcase // mode
  endfunction

  function automatic int unsigned n_of_ntom (vtg_ratio_e mode);
  	case(mode)
  		VTG_N1_M2: return 1;
  		VTG_N1_M4: return 1;
  		VTG_N2_M4: return 2;
  		default:   return 0;
  	endcase // mode
  endfunction

  function automatic int unsigned m_of_ntom (vtg_ratio_e mode);
  	case(mode)
  		VTG_N1_M2: return 2;
  		VTG_N1_M4: return 4;
  		VTG_N2_M4: return 4;
  		default:   return 0;
  	endcase // mode
  endfunction

  function automatic logic [WideIndexWidth-1:0] get_wide_base_index (int unsigned e);
  	int unsigned n = n_of_ntom(vtg_mode_i);
  	int unsigned m = m_of_ntom(vtg_mode_i);
  	int unsigned base = e * (m / n);

  	logic [WideIndexWidth-1:0] wide_idx;
  	wide_idx = base[WideIndexWidth-1:0];
  	return wide_idx;
  endfunction

  ///////////////////////////////////////////////////
  // Reinterpret the input lanes as element arrays //
  ///////////////////////////////////////////////////

  typedef logic [EleWidth-1:0] ele_t;

  // lane[k] is the k-th slice of the wide input channel
  ele_t [WidenRatio-1:0][NumNarrowEle-1:0]    lanes_e;

  generate 
  	for (int unsigned l = 0; l < WidenRatio; l++) begin : gen_lanes
  	  for (int unsigned e = 0; e < NumNarrowEle; e++) begin : gen_lane_ele
  		localparam int unsigned lo = e  * EleWidth;
  		localparam int unsigned hi = lo + EleWidth -1;
  		assign lanes_e[l][e] = lanes[l][hi:lo];
  	  end
    end
  endgenerate

  ////////////////////////////////////////////////////////
  // Decode per-output-element indices from vtg_index_i //
  ////////////////////////////////////////////////////////

  // interral index signal
  // TODO (bowwang): potentially need to temperally store index
  logic vrf_data_t flat_index;
  assign flat_index = vtg_index_i;

  // TODO (bowwang): adapt to more index modes, currently only support x:2 and x:4 
  function automatic logic [1:0] idx2_of_ele (int unsigned e);
  	automatic int unsigned base = e * idx_bits_per_ele(vtg_mode_i);
  	// index interpretation 
  	case (vtg_mode_i)
  		VTG_N1_M2: return { 1'b0,               flat_index[base]};
  		VTG_N1_M4,
  		VTG_N2_M4: return { flat_index[base+1], flat_index[base]};
  		default:   return 2'b00;
  	endcase // vtg_mode_i
  endfunction


  // Output Packing
  ele_t [NumNarrowEle-1:0] out_ele;

  generate
  	for(int unsigned ele = 0; ele < NumNarrowEle; ele++) begin
  		logic [WideIndexWidth-1:0] idx_base, idx_wide;
  		logic [1:0]                idx_local;
  		int unsigned lane_sel, off_sel;

  		always_comb begin
  			idx_base   = get_wide_base_index(e);
  			idx_local  = idx2_of_ele(e);
  			idx_wide   = idx_base + idx_local;
  			lane_sel   = idx_wide / NumNarrowEle;
  			off_sel    = idx_wide % NumNarrowEle;
  			out_ele[e] = lanes_e[lane_sel][off_sel];
  		end
  	end
  endgenerate

  logic [NarrowDataWidth-1:0] out_bus;
  generate
  	for (int unsigned e = 0; e<NumNarrowEle; e++) begin
  		localparam int lo = e  * EleWidth;
  		localparam int hi = lo + EleWidth - 1;
  		assign out_bus[hi:lo] = out_ele[e];
  	end
  endgenerate

  assign rdata_o = out_bus;

endmodule