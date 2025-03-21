// - - - - - TOP MODULE - - - - - //
// 
// The top module integrates all lower level modules in the FPGA based audio processor, placing each
// module, which behaves similarly to a guitar pedal, in series.
// 
// 

`timescale 1ns / 1ps
`default_nettype none

module top (
    input  wire        clk_100mhz,   // 100MHz input clock
    input  wire        reset,        // Active low reset from button
    input  wire [7:0]  sw,          // 8 switches for various controls
    
    // I2S2 Interface signals
    output wire        tx_mclk,
    output wire        tx_lrck,
    output wire        tx_sclk,
    output wire        tx_sdout,
    output wire        rx_mclk,
    output wire        rx_lrck,
    output wire        rx_sclk,
    input  wire        rx_sdin
);
    // Wire declarations
    wire        resetn;
    wire        axis_clk;        // 22.591MHz clock
    wire        locked;          // PLL lock signal
    wire        axis_resetn;     // Reset for I2S module
    
    // AXIS interface signals
    wire [31:0] rx_axis_m_data;    // From I2S2 to noise gate
    wire        rx_axis_m_valid;
    wire        rx_axis_m_ready;
    wire        rx_axis_m_last;
    
    wire [31:0] gate_axis_m_data;   // From noise gate to filter
    wire        gate_axis_m_valid;
    wire        gate_axis_m_ready;
    wire        gate_axis_m_last;
    
    wire [31:0] filter_axis_m_data;  // From filter to clip distortion
    wire        filter_axis_m_valid;
    wire        filter_axis_m_ready;
    wire        filter_axis_m_last;
    
    wire [31:0] clip_axis_m_data;    // From clip distortion to ring modulator
    wire        clip_axis_m_valid;
    wire        clip_axis_m_ready;
    wire        clip_axis_m_last;
    
    wire [31:0] ringmod_axis_m_data;   // From ring modulator to volume controller
    wire        ringmod_axis_m_valid;
    wire        ringmod_axis_m_ready;
    wire        ringmod_axis_m_last;
    
    wire [31:0] vol_axis_m_data;    // From volume controller to I2S2
    wire        vol_axis_m_valid;
    wire        vol_axis_m_ready;
    wire        vol_axis_m_last;

    // Reset logic
    assign resetn = ~reset;
    assign axis_resetn = resetn & locked;

    // Instantiate Clock Generator
    clk_wiz_0 clk_gen (
        .clk_in1(clk_100mhz),
        .reset(~resetn),
        .locked(locked),
        .axis_clk(axis_clk)
    );

    // Instantiate I2S2 Controller
    axis_i2s2 i2s2_inst (
        .axis_clk(axis_clk),
        .axis_resetn(axis_resetn),
        
        .tx_axis_s_data(vol_axis_m_data),
        .tx_axis_s_valid(vol_axis_m_valid),
        .tx_axis_s_ready(vol_axis_m_ready),
        .tx_axis_s_last(vol_axis_m_last),
        
        .rx_axis_m_data(rx_axis_m_data),
        .rx_axis_m_valid(rx_axis_m_valid),
        .rx_axis_m_ready(rx_axis_m_ready),
        .rx_axis_m_last(rx_axis_m_last),
        
        .tx_mclk(tx_mclk),
        .tx_lrck(tx_lrck),
        .tx_sclk(tx_sclk),
        .tx_sdout(tx_sdout),
        .rx_mclk(rx_mclk),
        .rx_lrck(rx_lrck),
        .rx_sclk(rx_sclk),
        .rx_sdin(rx_sdin)
    );

    // Instantiate Noise Gate (first in chain)
    axis_noise_gate noise_gate_inst (
        .clk(axis_clk),
        .resetn(axis_resetn),
        .gate_enable(sw[7]),       // Switch 7 for noise gate
        .threshold_level(sw[6:5]), // Switches 5-6 for threshold level
        
        .s_axis_data(rx_axis_m_data),
        .s_axis_valid(rx_axis_m_valid),
        .s_axis_ready(rx_axis_m_ready),
        .s_axis_last(rx_axis_m_last),
        
        .m_axis_data(gate_axis_m_data),
        .m_axis_valid(gate_axis_m_valid),
        .m_axis_ready(gate_axis_m_ready),
        .m_axis_last(gate_axis_m_last)
    );

    // Instantiate Low Pass Filter (second in chain)
    axis_lowpass_filter filter_inst (
        .clk(axis_clk),
        .resetn(axis_resetn),
        .filter_enable(sw[2]),      // Switch 2 for filter control
        
        .s_axis_data(gate_axis_m_data),
        .s_axis_valid(gate_axis_m_valid),
        .s_axis_ready(gate_axis_m_ready),
        .s_axis_last(gate_axis_m_last),
        
        .m_axis_data(filter_axis_m_data),
        .m_axis_valid(filter_axis_m_valid),
        .m_axis_ready(filter_axis_m_ready),
        .m_axis_last(filter_axis_m_last)
    );

    // Instantiate Clip Distortion (third in chain)
    axis_clip_distortion clip_inst (
        .clk(axis_clk),
        .resetn(axis_resetn),
        .distortion_enable(sw[3]),     // Switch 3 for distortion
        
        .s_axis_data(filter_axis_m_data),
        .s_axis_valid(filter_axis_m_valid),
        .s_axis_ready(filter_axis_m_ready),
        .s_axis_last(filter_axis_m_last),
        
        .m_axis_data(clip_axis_m_data),
        .m_axis_valid(clip_axis_m_valid),
        .m_axis_ready(clip_axis_m_ready),
        .m_axis_last(clip_axis_m_last)
    );
    
    // Instantiate Ring Modulator (fourth in chain)
    axis_ring_modulator ringmod_inst (
        .clk(axis_clk),
        .resetn(axis_resetn),
        .ringmod_enable(sw[4]),         // Switch 4 for ring modulator effect
        .ringmod_freq(sw[6:5]),         // Switches 5-6 for frequency (shared with noise gate)
        
        .s_axis_data(clip_axis_m_data),
        .s_axis_valid(clip_axis_m_valid),
        .s_axis_ready(clip_axis_m_ready),
        .s_axis_last(clip_axis_m_last),
        
        .m_axis_data(ringmod_axis_m_data),
        .m_axis_valid(ringmod_axis_m_valid),
        .m_axis_ready(ringmod_axis_m_ready),
        .m_axis_last(ringmod_axis_m_last)
    );

    // Instantiate Volume Controller (final stage)
    axis_volume_controller #(
        .SWITCH_WIDTH(2),        // Use 2 switches for volume
        .DATA_WIDTH(32)          // Full stereo width
    ) vol_ctrl_inst (
        .clk(axis_clk),
        .sw(sw[1:0]),           // Switches 0 and 1 for volume
        
        .s_axis_data(ringmod_axis_m_data),  // Input from ring modulator
        .s_axis_valid(ringmod_axis_m_valid),
        .s_axis_ready(ringmod_axis_m_ready),
        .s_axis_last(ringmod_axis_m_last),
        
        .m_axis_data(vol_axis_m_data),
        .m_axis_valid(vol_axis_m_valid),
        .m_axis_ready(vol_axis_m_ready),
        .m_axis_last(vol_axis_m_last)
    );

endmodule