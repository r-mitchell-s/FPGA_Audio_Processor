// - - - - - TOP MODULE - - - - - //
// 
// The top module integrates all lower level modules in the FPGA based audio processor, placing each
// module, which behaves similarly to a guitar pedal, in series.
// 
// 

`timescale 1ns / 1ps
`default_nettype none

module top (
    input  wire        clk_100mhz,          // 100MHz input clock (from FPGA)
    input  wire        reset,               // active low reset from button
    input  wire [3:0]  sw,                  // 4 switches (2 for volume, 1 for filter, 1 for distortion)
    
    // I2S2 Interface signals
    output wire        tx_mclk,             // master clock (Tx)
    output wire        tx_lrck,             // left-right toggle clock (Tx)
    output wire        tx_sclk,             // slave clock (Tx)
    output wire        tx_sdout,            // serial data out (bits going out to DAC)
    output wire        rx_mclk,             // master clock (Rx)
    output wire        rx_lrck,             // left-right toggle clock (Rx)
    output wire        rx_sclk,             // slave clock (Rx)
    input  wire        rx_sdin              // serial data input (bits coming in from ADC)
);
    // wire declarations
    wire        resetn;
    wire        axis_clk;                   // 22.591MHz clock
    wire        locked;                     // PLL lock signal
    wire        axis_resetn;                // reset for I2S module
    
    // - - - - - AXIS MODULE INTERFACES - - - - - //

    // from I2S2 to filter
    wire [31:0] rx_axis_m_data;             
    wire        rx_axis_m_valid;
    wire        rx_axis_m_ready;
    wire        rx_axis_m_last;
    
    // from filter to clip distortion
    wire [31:0] filter_axis_m_data;        
    wire        filter_axis_m_valid;
    wire        filter_axis_m_ready;
    wire        filter_axis_m_last;
    
    // from clip distortion to volume controller
    wire [31:0] clip_axis_m_data;           
    wire        clip_axis_m_valid;
    wire        clip_axis_m_ready;
    wire        clip_axis_m_last;
    
    // from volume controller to I2S2
    wire [31:0] vol_axis_m_data;          
    wire        vol_axis_m_valid;
    wire        vol_axis_m_ready;
    wire        vol_axis_m_last;

    // reset logic
    assign resetn = ~reset;
    assign axis_resetn = resetn & locked;

    // instantiate Clocking Wizard
    clk_wiz_0 clk_gen (
        .clk_in1(clk_100mhz),
        .reset(~resetn),
        .locked(locked),
        .axis_clk(axis_clk)
    );

    // instantiate I2S2 Controller
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

    // instantiate Low Pass Filter (activated with switch 2)
    axis_lowpass_filter filter_inst (
        
        // control signals
        .clk(axis_clk),
        .resetn(axis_resetn),
        .filter_enable(sw[2]),
        
        // sink interface
        .s_axis_data(rx_axis_m_data),
        .s_axis_valid(rx_axis_m_valid),
        .s_axis_ready(rx_axis_m_ready),
        .s_axis_last(rx_axis_m_last),
        
        // source interface
        .m_axis_data(filter_axis_m_data),
        .m_axis_valid(filter_axis_m_valid),
        .m_axis_ready(filter_axis_m_ready),
        .m_axis_last(filter_axis_m_last)
    );

    // instantiate Clip Distortion
    axis_clip_distortion clip_inst (
        
        // control interface
        .clk(axis_clk),
        .resetn(axis_resetn),
        .distortion_enable(sw[3]),    
        
        // sink interface
        .s_axis_data(filter_axis_m_data),
        .s_axis_valid(filter_axis_m_valid),
        .s_axis_ready(filter_axis_m_ready),
        .s_axis_last(filter_axis_m_last),
        
        // source interface
        .m_axis_data(clip_axis_m_data),
        .m_axis_valid(clip_axis_m_valid),
        .m_axis_ready(clip_axis_m_ready),
        .m_axis_last(clip_axis_m_last)
    );

    // instantiate Volume Controller
    axis_volume_controller #(
        .SWITCH_WIDTH(2),
        .DATA_WIDTH(32)          
    ) vol_ctrl_inst (
        
        // control interface
        .clk(axis_clk),
        .sw(sw[1:0]),           
        
        // sink interface
        .s_axis_data(clip_axis_m_data),
        .s_axis_valid(clip_axis_m_valid),
        .s_axis_ready(clip_axis_m_ready),
        .s_axis_last(clip_axis_m_last),
        
        // source interface
        .m_axis_data(vol_axis_m_data),
        .m_axis_valid(vol_axis_m_valid),
        .m_axis_ready(vol_axis_m_ready),
        .m_axis_last(vol_axis_m_last)
    );
endmodule