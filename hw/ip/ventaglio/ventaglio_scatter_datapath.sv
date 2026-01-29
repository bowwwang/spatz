module ventaglio_scatter_datapath
  import spatz_pkg::*;
  import vtl_pkg::*;
#(
  parameter int unsigned NrEffElePerBlk = 2,
  parameter int unsigned NrElePerBlk    = 4,
  parameter int unsigned NrCh           = 2,
  parameter int unsigned IdxWidth       = 2,
  parameter int unsigned EleWidth       = 32
) (
  input  logic                               clk_i,
  input  logic                               rst_ni,

  input  logic                               scatter_done_i,
  input  logic                               we_i,
  input  vrf_addr_t                          waddr_i,

  input  vrf_data_t                          wdata_i,
  input  vrf_be_t                            wbe_i,

  input  vrf_data_t                          index_i,
  input  logic                               index_valid_i,

  input  logic      [VENTAGLIO_WFACTOR-1:0]  wvalid_i,

  output vrf_data_t [VENTAGLIO_WFACTOR-1:0]  wdata_o,
  output vrf_be_t   [VENTAGLIO_WFACTOR-1:0]  wbe_o,
  output logic                               wvalid_o
);

  `include "common_cells/registers.svh"

  ////////////////////////////
  //     beat tracking      //
  ////////////////////////////
  logic [3:0] beat_cnt_d, beat_cnt_q;
  `FF(beat_cnt_q, beat_cnt_d, '0)

  logic addr_last_bit_d, addr_last_bit_q;
  `FF(addr_last_bit_q, addr_last_bit_d, '0)

  logic beat_cnt_en;
  assign addr_last_bit_d = waddr_i[0];
  // (mirrors gather style: only advance on actual traffic)
  assign beat_cnt_en = (!scatter_done_i) && (addr_last_bit_d ^ addr_last_bit_q) && we_i;

  localparam int unsigned EleWidthB       = EleWidth / 8;
  localparam int unsigned NrBlksPerBeat   = (VRFWordWidth * NrCh) / (NrElePerBlk * EleWidth);
  localparam int unsigned NrBeatsPerInput = EleWidth / IdxWidth;

  always_comb begin
    beat_cnt_d = beat_cnt_q;

    if (beat_cnt_en) begin
      if (beat_cnt_q == NrBeatsPerInput-1) begin
        beat_cnt_d = '0;
      end else begin
        beat_cnt_d = beat_cnt_q + 1'b1;
      end
    end

    if (scatter_done_i) beat_cnt_d = '0;
  end

  ////////////////////////////
  //      index buffer      //
  ////////////////////////////
  vrf_data_t index_d, index_q;
  `FF(index_q, index_d, '0)

  always_comb begin
    index_d = index_q;

    // preserve your original “refresh” conditions (write side is delayed)
    if ( index_valid_i
         && ( (beat_cnt_q == '0)
              || scatter_done_i
              || (beat_cnt_q == (NrBeatsPerInput-1)-1) ) ) begin
      index_d = index_i;
    end
  end

  ////////////////////////////
  //     scatter logic      //
  ////////////////////////////

  // Unpack indices
  logic [NrBeatsPerInput-1:0][NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][IdxWidth-1:0] idx;
  for (genvar beat = 0; beat < NrBeatsPerInput; beat++) begin
    for (genvar blk = 0; blk < NrBlksPerBeat; blk++) begin
      for (genvar ele = 0; ele < NrEffElePerBlk; ele++) begin
        assign idx[beat][blk][ele] =
          index_q[beat*NrBlksPerBeat*NrEffElePerBlk*IdxWidth +
                  blk*NrEffElePerBlk*IdxWidth +
                  ele*IdxWidth +: IdxWidth];
      end
    end
  end

  // Pre-scatter: slice incoming narrow data into [blk][eff-ele]
  logic [NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][EleWidth-1:0]  wdata_pre_scatter;
  logic [NrBlksPerBeat-1:0][NrEffElePerBlk-1:0][EleWidthB-1:0] wbe_pre_scatter;

  for (genvar blk = 0; blk < NrBlksPerBeat; blk++) begin
    for (genvar ele = 0; ele < NrEffElePerBlk; ele++) begin
      assign wdata_pre_scatter[blk][ele] =
        wdata_i[blk*NrEffElePerBlk*EleWidth + ele*EleWidth +: EleWidth];
      assign wbe_pre_scatter[blk][ele] =
        wbe_i[blk*NrEffElePerBlk*EleWidthB + ele*EleWidthB +: EleWidthB];
    end
  end

  // Post-scatter: place effective elements into full block slots [0..NrElePerBlk-1]
  logic [NrBlksPerBeat-1:0][NrElePerBlk-1:0][EleWidth-1:0]  wdata_post_scatter;
  logic [NrBlksPerBeat-1:0][NrElePerBlk-1:0][EleWidthB-1:0] wbe_post_scatter;

  always_comb begin
    wdata_post_scatter = '0;
    wbe_post_scatter   = '0;

    for (int blk = 0; blk < NrBlksPerBeat; blk++) begin
      for (int ele = 0; ele < NrEffElePerBlk; ele++) begin
        wdata_post_scatter[blk][ idx[beat_cnt_d][blk][ele] ] = wdata_pre_scatter[blk][ele];
        wbe_post_scatter[blk][ idx[beat_cnt_d][blk][ele] ]   = wbe_pre_scatter[blk][ele];
      end
    end
  end

  // Flatten to per-channel wide words
  logic [VRFWordWidth*NrCh-1:0]  flatten_wdata;
  logic [VRFWordBWidth*NrCh-1:0] flatten_wbe;

  for (genvar blk = 0; blk < NrBlksPerBeat; blk++) begin
    for (genvar ele = 0; ele < NrElePerBlk; ele++) begin
      assign flatten_wdata[blk*NrElePerBlk*EleWidth + ele*EleWidth +: EleWidth] =
        wdata_post_scatter[blk][ele];
      assign flatten_wbe[blk*NrElePerBlk*EleWidthB + ele*EleWidthB +: EleWidthB] =
        wbe_post_scatter[blk][ele];
    end
  end

  // Drive output arrays: first NrCh channels get data, rest are zero
  for (genvar ch = 0; ch < VENTAGLIO_WFACTOR; ch++) begin : gen_out
    if (ch < NrCh) begin
      assign wdata_o[ch] = flatten_wdata[ch*VRFWordWidth +: VRFWordWidth];
      assign wbe_o[ch]   = flatten_wbe  [ch*VRFWordBWidth +: VRFWordBWidth];
    end else begin
      assign wdata_o[ch] = '0;
      assign wbe_o[ch]   = '0;
    end
  end

  // Valid: AND only the channels this core uses
  assign wvalid_o = &wvalid_i[NrCh-1:0];

endmodule : ventaglio_scatter_datapath
