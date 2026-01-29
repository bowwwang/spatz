module ventaglio_scatter
  import spatz_pkg::*;
  import vtl_pkg::*;
#(
  parameter int unsigned NarrowDataWidth = VTGChannelWidth,
  parameter int unsigned WideDataWidth   = VTGChannelWidth * VENTAGLIO_WFACTOR
) (
  input  logic                               clk_i,
  input  logic                               rst_ni,
  input  logic                               testmode_i,

  // write ports -- from VFU (narrow)
  input  vrf_addr_t                          waddr_i,
  input  vrf_data_t                          wdata_i,
  input  logic                               we_i,
  input  vrf_be_t                            wbe_i,
  output logic                               wvalid_o,

  // write ports -- to Buffer (wide)
  output vrf_addr_t [VENTAGLIO_WFACTOR-1:0]  waddr_o,
  output vrf_data_t [VENTAGLIO_WFACTOR-1:0]  wdata_o,
  output logic      [VENTAGLIO_WFACTOR-1:0]  we_o,
  output vrf_be_t   [VENTAGLIO_WFACTOR-1:0]  wbe_o,
  input  logic      [VENTAGLIO_WFACTOR-1:0]  wvalid_i,

  // control
  input  logic                               scatter_done_i,

  // index
  input  vrf_data_t                          index_i,
  input  sp_cfg_t                            vtl_cfg_i,
  input  logic                               index_valid_i
);

  `include "common_cells/registers.svh"

  ////////////////////////////
  //     request logics     //
  ////////////////////////////
  vrf_addr_t waddr;

  always_comb begin : proc_write_addr
    waddr   = '0;
    waddr_o = '0;
    we_o    = '0;

    unique case (vtl_cfg_i.sp_cfg_ratio)
      SP_RATIO_050: begin
        // one write maps to 2 channels
        waddr      = waddr_i << 1;
        waddr_o[0] = {waddr[$clog2(NrVRFWords)-1:1], 1'b0};
        waddr_o[1] = {waddr[$clog2(NrVRFWords)-1:1], 1'b1};
        if (we_i) we_o[1:0] = 2'b11;
      end

      SP_RATIO_025: begin
        // one write maps to 4 channels
        waddr      = waddr_i << 2;
        waddr_o[0] = {waddr[$clog2(NrVRFWords)-1:2], 2'b00};
        waddr_o[1] = {waddr[$clog2(NrVRFWords)-1:2], 2'b01};
        waddr_o[2] = {waddr[$clog2(NrVRFWords)-1:2], 2'b10};
        waddr_o[3] = {waddr[$clog2(NrVRFWords)-1:2], 2'b11};
        if (we_i) we_o = '1;
      end

      default: begin
        waddr = waddr_i;
      end
    endcase
  end

  ////////////////////////////
  //  parameterized cores   //
  ////////////////////////////

  vrf_data_t [VENTAGLIO_WFACTOR-1:0] wdata_050, wdata_025;
  vrf_be_t   [VENTAGLIO_WFACTOR-1:0] wbe_050,   wbe_025;
  logic                               wvalid_050, wvalid_025;

  // 2:4 scatter (effective 2 elems per block, 2 channels used)
  ventaglio_scatter_datapath #(
    .NrEffElePerBlk(2),
    .NrElePerBlk   (4),
    .NrCh          (2),
    .IdxWidth      (2),
    .EleWidth      (32)
  ) i_scatter_2of4 (
    .clk_i,
    .rst_ni,
    .scatter_done_i,
    .we_i,
    .waddr_i,
    .wdata_i,
    .wbe_i,
    .index_i,
    .index_valid_i,
    .wvalid_i,
    .wdata_o  (wdata_050),
    .wbe_o    (wbe_050),
    .wvalid_o (wvalid_050)
  );

  // 1:4 scatter (effective 1 elem per block, 4 channels used)
  ventaglio_scatter_datapath #(
    .NrEffElePerBlk(1),  // key change for 1:4
    .NrElePerBlk   (4),
    .NrCh          (4),  // key change for 1:4
    .IdxWidth      (2),
    .EleWidth      (32)
  ) i_scatter_1of4 (
    .clk_i,
    .rst_ni,
    .scatter_done_i,
    .we_i,
    .waddr_i,
    .wdata_i,
    .wbe_i,
    .index_i,
    .index_valid_i,
    .wvalid_i,
    .wdata_o  (wdata_025),
    .wbe_o    (wbe_025),
    .wvalid_o (wvalid_025)
  );

  // Select active format
  always_comb begin
    unique case (vtl_cfg_i.sp_cfg_ratio)
      SP_RATIO_050: begin
        wdata_o  = wdata_050;
        wbe_o    = wbe_050;
        wvalid_o = wvalid_050;
      end

      SP_RATIO_025: begin
        wdata_o  = wdata_025;
        wbe_o    = wbe_025;
        wvalid_o = wvalid_025;
      end

      default: begin
        wdata_o  = '0;
        wbe_o    = '0;
        wvalid_o = 1'b0;
      end
    endcase
  end

endmodule : ventaglio_scatter
