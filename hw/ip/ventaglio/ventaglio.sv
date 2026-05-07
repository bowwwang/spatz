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
  import rvv_pkg::*;
  import vtl_pkg::*;
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



  // Spill admits any VTL op regardless of which ex_unit it routes to:
  //   - vfxmacc.vrf / vfxmul.vrf are dispatched as ex_unit=VFU.
  //   - vventclr is dispatched as ex_unit=SLD (bypasses the VFU entirely).
  // Both kinds need ventaglio's bank, so admission is gated solely on use_vtl.
  spill_register #(
    .T(spatz_req_t)
  ) i_operation_queue (
    .clk_i  (clk_i                                              ),
    .rst_ni (rst_ni                                             ),
    .data_i (spatz_req_d                                        ),
    .valid_i(spatz_req_valid_i && spatz_req_i.op_vtl.use_vtl    ),
    .ready_o(spatz_req_ready_o                                  ),
    .data_o (spatz_req                                          ),
    .valid_o(spatz_req_valid                                    ),
    .ready_i(spatz_req_ready                                    )
  );

  always_comb begin : proc_spatz_req
    spatz_req_d = spatz_req_i;
  end

  /******************************/
  /*       Clear Sequencer      */
  /******************************/
  // vventclr writes 0 to every cell of ventaglio's bank. Walks bank rows
  // 0 → VTGNrWordsPerChannel-1, all VTGNrChannels in parallel each cycle.
  // On the last row, asserts vtl_rsp_valid_o so the controller can retire
  // the op via the (formerly-VSLDU) SLD retirement path.
  typedef logic [$clog2(VTGNrWordsPerChannel)-1:0] vtg_row_addr_t;

  logic           clear_active_q,  clear_active_d;
  vtg_row_addr_t  clear_row_q,     clear_row_d;
  spatz_id_t      clear_id_q,      clear_id_d;
  `FF(clear_active_q, clear_active_d, 1'b0)
  `FF(clear_row_q,    clear_row_d,    '0)
  `FF(clear_id_q,     clear_id_d,     '0)

  logic clear_last_row;
  assign clear_last_row = (clear_row_q == vtg_row_addr_t'(VTGNrWordsPerChannel - 1));

  // True for the cycle that retires the clear op.
  logic clear_done;
  assign clear_done = clear_active_q && clear_last_row;

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

  always_comb begin : proc_clear
    clear_active_d = clear_active_q;
    clear_row_d    = clear_row_q;
    clear_id_d     = clear_id_q;

    // Start clearing when a clear_buffer op is freshly latched in the spill.
    if (!clear_active_q && new_vtl_request_d && spatz_req.op_vtl.clear_buffer) begin
      clear_active_d = 1'b1;
      clear_row_d    = '0;
      clear_id_d     = spatz_req.id;
    end else if (clear_active_q) begin
      if (clear_last_row) begin
        clear_active_d = 1'b0;
      end else begin
        clear_row_d = clear_row_q + 1'b1;
      end
    end
  end

  // Retire the clear op via the SLD response port.
  assign vtl_rsp_valid_o = clear_done;
  assign vtl_rsp_o.id    = clear_id_q;

  always_comb begin : proc_vtl_state
    running_d = running_q;
    spatz_req_ready = !spatz_req_valid;

    // A new spatz_req is received
    if (new_vtl_request_d) begin
      running_d[spatz_req.id] = 1'b1;
    end

    // VFU-routed ops retire via vfu_rsp.
    if (running_q[vfu_rsp_i.id] && vfu_rsp_valid_i) begin
      running_d[vfu_rsp_i.id] = 1'b0;
    end
    // vventclr retires via clear_done (vtl_rsp_valid_o).
    if (clear_done && running_q[clear_id_q]) begin
      running_d[clear_id_q] = 1'b0;
    end

    // Spill release:
    //   - For vventclr (clear_buffer=1): hold the spill until clearing is
    //     done. Ignores spatz_vfu_req_ready_i because vventclr never goes
    //     to the VFU.
    //   - For other VTL ops: release when the VFU latches the op.
    if (spatz_req_valid && spatz_req.op_vtl.clear_buffer) begin
      if (clear_done) spatz_req_ready = 1'b1;
    end else if (spatz_vfu_req_ready_i) begin
      spatz_req_ready = 1'b1;
    end
  end

  //////////////////////////////////
  //  Double-Buffered Index RAM   //
  //////////////////////////////////
  // Two-slot ping-pong index buffer. While the active op consumes one slot,
  // the spare slot is pre-filled with the next op's index (peeked from
  // spatz_req_i). When the VFU latches the next op (spatz_vfu_req_ready_i),
  // active_buf_q flips and the formerly-spare slot becomes active without
  // any reload bubble. Eliminates the index-load stall between back-to-back
  // sparse ops (vfxmacc.vrf / vfxmul.vrf in SpMM kernels).

  vrf_data_t  index_data_q       [0:1], index_data_d       [0:1];
  logic       index_buf_valid_q  [0:1], index_buf_valid_d  [0:1];
  spatz_id_t  index_buf_id_q     [0:1], index_buf_id_d     [0:1];
  `FF(index_data_q[0],      index_data_d[0],      '0)
  `FF(index_data_q[1],      index_data_d[1],      '0)
  `FF(index_buf_valid_q[0], index_buf_valid_d[0], 1'b0)
  `FF(index_buf_valid_q[1], index_buf_valid_d[1], 1'b0)
  `FF(index_buf_id_q[0],    index_buf_id_d[0],    '0)
  `FF(index_buf_id_q[1],    index_buf_id_d[1],    '0)

  // Pointer: which slot the active spatz_req consumes from.
  logic active_buf_q, active_buf_d;
  `FF(active_buf_q, active_buf_d, 1'b0)

  // Locked fetch state — once vrf_re_o is issued, latch the target slot/id/
  // vreg so a slow vrf response can't be misrouted if spatz_req_i changes
  // mid-flight (e.g. controller switches to a non-VTL op).
  logic       fetch_lock_q,   fetch_lock_d;
  logic       fetch_target_q, fetch_target_d;
  spatz_id_t  fetch_id_q,     fetch_id_d;
  vreg_t      fetch_vidx_q,   fetch_vidx_d;
  `FF(fetch_lock_q,   fetch_lock_d,   1'b0)
  `FF(fetch_target_q, fetch_target_d, 1'b0)
  `FF(fetch_id_q,     fetch_id_d,     '0)
  `FF(fetch_vidx_q,   fetch_vidx_d,   '0)

  // ── Composed view (replaces single index_q / index_valid_q) ──
  vrf_data_t  index_q;
  logic       index_valid_q;
  assign index_q       = index_data_q[active_buf_q];
  assign index_valid_q = index_buf_valid_q[active_buf_q] &&
                         (index_buf_id_q[active_buf_q] == spatz_req.id);

  // ── Next-op peek ──
  // Spill must be full (current op latched) AND controller must be offering
  // a different VTL op on the wires. If next == current, nothing to prefetch.
  logic       next_op_visible;
  spatz_id_t  next_id;
  vreg_t      next_idx_vreg;
  assign next_op_visible = spatz_req_valid && spatz_req_valid_i
                           && spatz_req_i.ex_unit == VFU
                           && spatz_req_i.op_vtl.use_vtl
                           && spatz_req_i.id != spatz_req.id;
  assign next_id        = spatz_req_i.id;
  assign next_idx_vreg  = spatz_req_i.op_vtl.idx_vreg;

  // Spare slot status
  logic spare_buf;
  assign spare_buf = ~active_buf_q;

  logic spare_holds_next;
  assign spare_holds_next = next_op_visible &&
                            ((index_buf_valid_q[spare_buf] &&
                              index_buf_id_q[spare_buf] == next_id) ||
                             (fetch_lock_q && fetch_id_q == next_id));

  // ── Fetch decision: only on-demand for the current op.
  //
  // The speculative prefetch into the spare slot was disabled because the
  // controller's scoreboard does NOT track `op_vtl.idx_vreg` as a read
  // dependency (it's not in vs1/vs2/vd). When the prefetch fires for the
  // next op, it can read the index vreg BEFORE the kernel's vlx32.v has
  // updated it, latching stale data into the spare slot. The next op then
  // gathers/scatters at the wrong lanes. The race was masked previously by
  // the init-zero FSM's ~4-cycle latency; with vventclr-based init that
  // gap closed and the race surfaced.
  //
  // Keeping the two-slot buffer infrastructure for now (the spare slot is
  // simply never written), so the existing slot/lock plumbing still works.
  logic       fetch_for_current;
  logic       fetch_for_next;
  logic       target_slot_sel;
  spatz_id_t  fetch_id_sel;
  vreg_t      fetch_vidx_sel;

  // vventclr has no operands and no index; suppress the fetch so we don't
  // pollute a buffer slot with a junk read of v0 (whose id can later be
  // reused by a vfxmul/vfxmacc, causing stale-index aliasing). Use a guard
  // form so we don't read spatz_req.op_vtl when spatz_req_valid is 0
  // (the spill data is X then, and X-AND short-circuit semantics aren't
  // reliable across all simulators).
  logic spatz_req_is_clear;
  assign spatz_req_is_clear = spatz_req_valid && spatz_req.op_vtl.clear_buffer;

  assign fetch_for_current = spatz_req_valid && !index_valid_q
                             && !spatz_req_is_clear
                             && !(fetch_lock_q && fetch_id_q == spatz_req.id);
  assign fetch_for_next    = 1'b0;  // prefetch disabled — see comment above

  always_comb begin : proc_fetch_sel
    target_slot_sel = active_buf_q;
    fetch_id_sel    = spatz_req.id;
    fetch_vidx_sel  = spatz_req.op_vtl.idx_vreg;
  end

  // Effective fetch state — locked overrides fresh once a fetch is in flight.
  logic       eff_fetch;
  logic       eff_target;
  spatz_id_t  eff_id;
  vreg_t      eff_vidx;
  always_comb begin : proc_eff_fetch
    if (fetch_lock_q) begin
      eff_fetch  = 1'b1;
      eff_target = fetch_target_q;
      eff_id     = fetch_id_q;
      eff_vidx   = fetch_vidx_q;
    end else begin
      eff_fetch  = fetch_for_current || fetch_for_next;
      eff_target = target_slot_sel;
      eff_id     = fetch_id_sel;
      eff_vidx   = fetch_vidx_sel;
    end
  end

  // Vector register file counter signals (gather multi-beat).
  logic      vreg_idx_counter_en;
  vrf_addr_t vreg_idx_counter_d;
  vrf_addr_t vreg_idx_counter_q;
  `FF(vreg_idx_counter_q, vreg_idx_counter_d, '0)

  always_comb begin : proc_idx_counter
    vreg_idx_counter_d = vreg_idx_counter_q;
    if (vreg_idx_counter_en)
      vreg_idx_counter_d = vreg_idx_counter_q + 1'b1;
    if (running_q[vfu_rsp_i.id] && vfu_rsp_valid_i)
      vreg_idx_counter_d = '0;
  end

  always_comb begin : proc_idx_addr_gen
    // Multi-beat counter only applies to the active op (gather flows);
    // prefetch always starts at counter=0.
    if (eff_id != spatz_req.id) begin
      vrf_raddr_o = {eff_vidx, $clog2(NrWordsPerVector)'(1'b0)};
    end else begin
      vrf_raddr_o = {eff_vidx, $clog2(NrWordsPerVector)'(1'b0)} + vreg_idx_counter_q;
      if (vreg_idx_counter_en)
        vrf_raddr_o = {eff_vidx, $clog2(NrWordsPerVector)'(1'b0)} + vreg_idx_counter_q + 1'b1;
    end
  end

  ////////////////////////////////
  // Counter for each operation //
  ////////////////////////////////

  // signals for gather or scatter
  logic is_gather, is_scatter;

  logic req_proceed;
  logic last_addr_bit_d, last_addr_bit_q;
  `FF(last_addr_bit_q, last_addr_bit_d, '0)

  assign last_addr_bit_d = raddr_i[0];
  assign req_proceed = last_addr_bit_d ^ last_addr_bit_q;

  logic [4:0] num_beats_per_op;
  // TODO: only considered N_FU=4
  assign num_beats_per_op = (spatz_req.vtype.vsew == EW_32) ? (spatz_req.vl >> 3) : spatz_req.vl[7:0];

  logic [4:0] op_beat_cnt_d, op_beat_cnt_q;
  `FF(op_beat_cnt_q, op_beat_cnt_d, '0)

  always_comb begin 
    op_beat_cnt_d = op_beat_cnt_q;
    // counter proceed 
    if (is_gather && rvalid_o && req_proceed ) op_beat_cnt_d = op_beat_cnt_q + 1;
    if (op_beat_cnt_q == num_beats_per_op - 1) op_beat_cnt_d = 0;
  end 

  // VRF master interface — driven by the effective (locked-or-fresh) fetch.
  // The controller's standard scoreboard still gates the read via the RAW
  // dep on op_vtl.idx_vreg of whichever op we're fetching for.
  assign vrf_re_o    = eff_fetch;
  assign vrf_id_o[0] = eff_id;

  // ── Buffer write-back, lock state, retire, pointer flip ──
  always_comb begin : proc_idx_buf
    for (int s = 0; s < 2; s++) begin
      index_data_d[s]      = index_data_q[s];
      index_buf_valid_d[s] = index_buf_valid_q[s];
      index_buf_id_d[s]    = index_buf_id_q[s];
    end
    active_buf_d   = active_buf_q;
    fetch_lock_d   = fetch_lock_q;
    fetch_target_d = fetch_target_q;
    fetch_id_d     = fetch_id_q;
    fetch_vidx_d   = fetch_vidx_q;

    // Issue cycle (no lock yet): latch the in-flight target/id/vreg so a
    // multi-cycle vrf response can't be misrouted.
    if (eff_fetch && !fetch_lock_q) begin
      fetch_lock_d   = 1'b1;
      fetch_target_d = target_slot_sel;
      fetch_id_d     = fetch_id_sel;
      fetch_vidx_d   = fetch_vidx_sel;
    end

    // Response: write into the locked target. Works for both same-cycle
    // (combinational rvalid) and multi-cycle responses, since eff_target/id
    // comes from the lock once it's set.
    if (vrf_rvalid_i && eff_fetch) begin
      index_data_d[eff_target]      = vrf_rdata_i;
      index_buf_valid_d[eff_target] = 1'b1;
      index_buf_id_d[eff_target]    = eff_id;
      fetch_lock_d                  = 1'b0;
    end

    // Retire: invalidate the slot that held the retired op. Only VFU
    // retirements matter — vventclr (VTL retire) doesn't write a buffer
    // slot in the first place because fetch_for_current is gated on
    // !clear_buffer above.
    if (vfu_rsp_valid_i) begin
      for (int s = 0; s < 2; s++) begin
        if (index_buf_valid_q[s] && (index_buf_id_q[s] == vfu_rsp_i.id))
          index_buf_valid_d[s] = 1'b0;
      end
    end

    // Op transition: VFU latches next op → flip pointer so the spare
    // (preloaded with the new op's index) becomes active.
    if (spatz_vfu_req_ready_i) active_buf_d = ~active_buf_q;
  end

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

  // (vtg_row_addr_t typedef defined earlier alongside the clear sequencer)

  /******************************/
  /*          Signals           */
  /******************************/

  // signals for gather or scatter
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

    if (clear_active_q) begin // priority 0: vventclr — zero every cell
      for (int unsigned channel = 0; channel < VTGNrChannels; channel++) begin
        waddr[channel] = clear_row_q;
        wdata[channel] = '0;
        we[channel]    = 1'b1;
        wbe[channel]   = '1;
      end
      // wvalid_o stays 0: no slave-port write is being acknowledged.
    end else if (!is_scatter) begin // priority 1: normal requests
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
      for (int unsigned b_channel = 0; b_channel < VTGNrChannels; b_channel++) begin
        for (int unsigned s_channel = 0; s_channel < VTGNrChannels; s_channel++) begin
          if (scatter_write_request[b_channel][s_channel][VRF_WD]) begin
            waddr[b_channel]                  = f_row(scatter_waddr[s_channel][VRF_WD]);
            wdata[b_channel]                  = scatter_wdata[s_channel][VRF_WD];
            we[b_channel]                     = 1'b1;
            wbe[b_channel]                    = scatter_wbe[s_channel][VRF_WD];
            scatter_wvalid[s_channel][VRF_WD] = 1'b1;
          end
        end
      end
      wvalid_o[VRF_WD] = post_scatter_wvalid[0];
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
    .num_beats_per_op_i  (num_beats_per_op),
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

  /******************************/
  /*       Debug Logger         */
  /******************************/
  // Focused trace of vfxmul/vfxmacc lifecycle for v16/v18 family. Only logs
  // events that help diagnose init-zero / index-buffer correctness; should
  // produce a few hundred lines for a single SpMM run. Synthesis ignores
  // all of `synthesis translate_off` / `_on`.
  // synthesis translate_off
  int dbg_fd;
  initial dbg_fd = $fopen("logs/ventaglio_debug.log", "w");

  // Cycle counter for human-readable timestamps
  longint dbg_cyc;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) dbg_cyc <= '0;
    else         dbg_cyc <= dbg_cyc + 1;
  end

  // Convenience: only log scatter writes targeted at v16..v19 (the SpMM
  // accumulators). vd_row_start covers spatz_req.vd, but for filtering
  // scatter writes we use the vd field directly.
  function automatic logic dbg_vd_of_interest(vreg_t v);
    return (v >= 16) && (v <= 19);
  endfunction

  always_ff @(posedge clk_i) begin
    if (rst_ni && dbg_fd != 0) begin
      // 1) New op latched into ventaglio's spill
      if (new_vtl_request_d) begin
        $fdisplay(dbg_fd,
          "[%0d] OP_LATCH id=%0d vd=v%0d idx_vreg=v%0d vlmul=%0d sp_ratio=%0d scatter=%0b clear=%0b",
          dbg_cyc, spatz_req.id, spatz_req.vd, spatz_req.op_vtl.idx_vreg,
          spatz_req.vtype.vlmul,
          spatz_req.op_vtl.sp_cfg.sp_cfg_ratio, spatz_req.op_vtl.scatter_vd,
          spatz_req.op_vtl.clear_buffer);
      end

      // 1b) Clear sequencer activity
      if (clear_active_d && !clear_active_q) begin
        $fdisplay(dbg_fd, "[%0d] CLEAR_START id=%0d (rows=%0d)",
                  dbg_cyc, clear_id_d, VTGNrWordsPerChannel);
      end
      if (clear_done) begin
        $fdisplay(dbg_fd, "[%0d] CLEAR_DONE id=%0d (vtl_rsp asserted)",
                  dbg_cyc, clear_id_q);
      end

      // 2) VFU response
      if (vfu_rsp_valid_i) begin
        $fdisplay(dbg_fd,
          "[%0d] VFU_RSP rsp.id=%0d spatz_req.id=%0d (match=%0b)",
          dbg_cyc, vfu_rsp_i.id, spatz_req.id,
          (vfu_rsp_i.id == spatz_req.id));
      end

      // 3) VFU input handshake (op transition)
      if (spatz_vfu_req_ready_i && spatz_req_valid) begin
        $fdisplay(dbg_fd,
          "[%0d] VFU_INPUT_READY (current spatz_req.id=%0d vd=v%0d)",
          dbg_cyc, spatz_req.id, spatz_req.vd);
      end

      // 4) Active-buffer pointer flip
      if (active_buf_d != active_buf_q) begin
        $fdisplay(dbg_fd,
          "[%0d] BUF_FLIP %0d -> %0d", dbg_cyc, active_buf_q, active_buf_d);
      end

      // 5) Index buffer slot write
      if (vrf_rvalid_i && eff_fetch) begin
        $fdisplay(dbg_fd,
          "[%0d] BUF_WRITE slot=%0d id=%0d data_lo=%h",
          dbg_cyc, eff_target, eff_id, vrf_rdata_i[31:0]);
      end

      // 6) Scatter writes targeted at v16..v19
      if (is_scatter && dbg_vd_of_interest(spatz_req.vd)) begin
        for (int b_ch = 0; b_ch < VTGNrChannels; b_ch++) begin
          if (we[b_ch]) begin
            $fdisplay(dbg_fd,
              "[%0d] SCATTER vd=v%0d id=%0d b_ch=%0d row=%0d wbe=%h data_lo=%h accepted=%0b",
              dbg_cyc, spatz_req.vd, spatz_req.id, b_ch,
              waddr[b_ch], wbe[b_ch], wdata[b_ch][31:0], wvalid_o[VRF_WD]);
          end
        end
      end
    end
  end
  // synthesis translate_on

endmodule : ventaglio