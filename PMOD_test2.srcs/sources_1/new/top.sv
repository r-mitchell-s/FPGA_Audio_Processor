//`timescale 1ns / 1ps
//`default_nettype none

//module top (
//    input  wire        clk_100mhz,   // 100MHz input clock
//    input  wire        reset,        // Active low reset from button
//    input  wire [3:0]  sw,          // 4 switches (2 for volume, 1 for filter, 1 spare)
    
//    // I2S2 Interface signals
//    output wire        tx_mclk,
//    output wire        tx_lrck,
//    output wire        tx_sclk,
//    output wire        tx_sdout,
//    output wire        rx_mclk,
//    output wire        rx_lrck,
//    output wire        rx_sclk,
//    input  wire        rx_sdin
//);
//    // Wire declarations
//    wire        resetn;
//    wire        axis_clk;        // 22.591MHz clock
//    wire        locked;          // PLL lock signal
//    wire        axis_resetn;     // Reset for I2S module
    
//    // AXIS interface signals
//    wire [31:0] rx_axis_m_data;    // From I2S2 to filter
//    wire        rx_axis_m_valid;
//    wire        rx_axis_m_ready;
//    wire        rx_axis_m_last;
    
//    wire [31:0] filter_axis_m_data;  // From filter to volume controller
//    wire        filter_axis_m_valid;
//    wire        filter_axis_m_ready;
//    wire        filter_axis_m_last;
    
//    wire [31:0] vol_axis_m_data;    // From volume controller to I2S2
//    wire        vol_axis_m_valid;
//    wire        vol_axis_m_ready;
//    wire        vol_axis_m_last;

//    // Reset logic
//    assign resetn = ~reset;
//    assign axis_resetn = resetn & locked;

//    // Instantiate Clock Generator
//    clk_wiz_0 clk_gen (
//        .clk_in1(clk_100mhz),
//        .reset(~resetn),
//        .locked(locked),
//        .axis_clk(axis_clk)
//    );

//    // Instantiate I2S2 Controller
//    axis_i2s2 i2s2_inst (
//        .axis_clk(axis_clk),
//        .axis_resetn(axis_resetn),
        
//        .tx_axis_s_data(vol_axis_m_data),
//        .tx_axis_s_valid(vol_axis_m_valid),
//        .tx_axis_s_ready(vol_axis_m_ready),
//        .tx_axis_s_last(vol_axis_m_last),
        
//        .rx_axis_m_data(rx_axis_m_data),
//        .rx_axis_m_valid(rx_axis_m_valid),
//        .rx_axis_m_ready(rx_axis_m_ready),
//        .rx_axis_m_last(rx_axis_m_last),
        
//        .tx_mclk(tx_mclk),
//        .tx_lrck(tx_lrck),
//        .tx_sclk(tx_sclk),
//        .tx_sdout(tx_sdout),
//        .rx_mclk(rx_mclk),
//        .rx_lrck(rx_lrck),
//        .rx_sclk(rx_sclk),
//        .rx_sdin(rx_sdin)
//    );

//    // Instantiate Low Pass Filter with switch control
//    axis_lowpass_filter filter_inst (
//        .clk(axis_clk),
//        .resetn(axis_resetn),
//        .filter_enable(sw[2]),      // Use switch 2 for filter control
        
//        .s_axis_data(rx_axis_m_data),
//        .s_axis_valid(rx_axis_m_valid),
//        .s_axis_ready(rx_axis_m_ready),
//        .s_axis_last(rx_axis_m_last),
        
//        .m_axis_data(filter_axis_m_data),
//        .m_axis_valid(filter_axis_m_valid),
//        .m_axis_ready(filter_axis_m_ready),
//        .m_axis_last(filter_axis_m_last)
//    );

//    // Instantiate Volume Controller
//    axis_volume_controller #(
//        .SWITCH_WIDTH(2),        // Use 2 switches for volume
//        .DATA_WIDTH(32)          // Full stereo width
//    ) vol_ctrl_inst (
//        .clk(axis_clk),
//        .sw(sw[1:0]),           // Use lower two switches for volume
        
//        .s_axis_data(filter_axis_m_data),    // Input from filter
//        .s_axis_valid(filter_axis_m_valid),
//        .s_axis_ready(filter_axis_m_ready),
//        .s_axis_last(filter_axis_m_last),
        
//        .m_axis_data(vol_axis_m_data),
//        .m_axis_valid(vol_axis_m_valid),
//        .m_axis_ready(vol_axis_m_ready),
//        .m_axis_last(vol_axis_m_last)
//    );

//endmodule

`timescale 1ns / 1ps
`default_nettype none

module top (
    input  wire        clk_100mhz,   // 100MHz input clock
    input  wire        reset,        // Active low reset from button
    input  wire [3:0]  sw,          // 4 switches (2 for volume, 1 for filter, 1 for distortion)
    
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
    wire [31:0] rx_axis_m_data;    // From I2S2 to filter
    wire        rx_axis_m_valid;
    wire        rx_axis_m_ready;
    wire        rx_axis_m_last;
    
    wire [31:0] filter_axis_m_data;  // From filter to clip distortion
    wire        filter_axis_m_valid;
    wire        filter_axis_m_ready;
    wire        filter_axis_m_last;
    
    wire [31:0] clip_axis_m_data;    // From clip distortion to volume controller
    wire        clip_axis_m_valid;
    wire        clip_axis_m_ready;
    wire        clip_axis_m_last;
    
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

    // Instantiate Low Pass Filter
    axis_lowpass_filter filter_inst (
        .clk(axis_clk),
        .resetn(axis_resetn),
        .filter_enable(sw[2]),      // Switch 2 for filter control
        
        .s_axis_data(rx_axis_m_data),
        .s_axis_valid(rx_axis_m_valid),
        .s_axis_ready(rx_axis_m_ready),
        .s_axis_last(rx_axis_m_last),
        
        .m_axis_data(filter_axis_m_data),
        .m_axis_valid(filter_axis_m_valid),
        .m_axis_ready(filter_axis_m_ready),
        .m_axis_last(filter_axis_m_last)
    );

    // Instantiate Clip Distortion
    axis_clip_distortion clip_inst (
        .clk(axis_clk),
        .resetn(axis_resetn),
        .distortion_enable(sw[3]),      // Switch 3 for distortion
        
        .s_axis_data(filter_axis_m_data),
        .s_axis_valid(filter_axis_m_valid),
        .s_axis_ready(filter_axis_m_ready),
        .s_axis_last(filter_axis_m_last),
        
        .m_axis_data(clip_axis_m_data),
        .m_axis_valid(clip_axis_m_valid),
        .m_axis_ready(clip_axis_m_ready),
        .m_axis_last(clip_axis_m_last)
    );

    // Instantiate Volume Controller
    axis_volume_controller #(
        .SWITCH_WIDTH(2),        // Use 2 switches for volume
        .DATA_WIDTH(32)          // Full stereo width
    ) vol_ctrl_inst (
        .clk(axis_clk),
        .sw(sw[1:0]),           // Switches 0 and 1 for volume
        
        .s_axis_data(clip_axis_m_data),    // Input from clip distortion
        .s_axis_valid(clip_axis_m_valid),
        .s_axis_ready(clip_axis_m_ready),
        .s_axis_last(clip_axis_m_last),
        
        .m_axis_data(vol_axis_m_data),
        .m_axis_valid(vol_axis_m_valid),
        .m_axis_ready(vol_axis_m_ready),
        .m_axis_last(vol_axis_m_last)
    );

endmodule