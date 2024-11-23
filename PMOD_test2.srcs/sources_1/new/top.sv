`timescale 1ns / 1ps
`default_nettype none
module top (
    input  wire        clk_100mhz,   // 100MHz input clock
    input  wire        reset,        // Active low reset from button
    input  wire [3:0]  sw,          // 4 switches (2 for volume, 2 for delay mix)
    
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
    // temp assignment test 
    assign resetn = ~reset;
    
    // Internal signals
    wire        resetn;
    wire        axis_clk;        // 22.591MHz clock
    wire        locked;          // PLL lock signal
    wire        axis_resetn;     // Reset for I2S module
    
    // AXIS interface signals
    wire [31:0] rx_axis_m_data;    // From I2S2 to delay effect
    wire        rx_axis_m_valid;
    wire        rx_axis_m_ready;
    wire        rx_axis_m_last;
    
    wire [31:0] delay_axis_m_data;  // From delay effect to volume controller
    wire        delay_axis_m_valid;
    wire        delay_axis_m_ready;
    wire        delay_axis_m_last;
    
    wire [31:0] vol_axis_m_data;    // From volume controller to I2S2
    wire        vol_axis_m_valid;
    wire        vol_axis_m_ready;
    wire        vol_axis_m_last;

    // Create stable reset once clock is locked
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

    // Instantiate Delay Effect
    axis_delay_effect #(
        .DATA_WIDTH(24),
        .DELAY_LENGTH(22050)  // 0.5 second delay at 44.1kHz
    ) delay_inst (
        .clk(axis_clk),
        .resetn(axis_resetn),
        .sw(sw[3:2]),  // Use upper two switches for delay mix
        
        .s_axis_data(rx_axis_m_data[23:0]),
        .s_axis_valid(rx_axis_m_valid),
        .s_axis_ready(rx_axis_m_ready),
        .s_axis_last(rx_axis_m_last),
        
        .m_axis_data(delay_axis_m_data[23:0]),
        .m_axis_valid(delay_axis_m_valid),
        .m_axis_ready(delay_axis_m_ready),
        .m_axis_last(delay_axis_m_last)
    );

    // Handle upper bits of the delay data path
    assign delay_axis_m_data[31:24] = 8'b0;

    // Instantiate Volume Controller
    axis_volume_controller #(
        .SWITCH_WIDTH(2),        // Modified to use only 2 switches
        .DATA_WIDTH(24)
    ) vol_ctrl_inst (
        .clk(axis_clk),
        .sw(sw[1:0]),           // Use lower two switches for volume
        
        .s_axis_data(delay_axis_m_data[23:0]),
        .s_axis_valid(delay_axis_m_valid),
        .s_axis_ready(delay_axis_m_ready),
        .s_axis_last(delay_axis_m_last),
        
        .m_axis_data(vol_axis_m_data[23:0]),
        .m_axis_valid(vol_axis_m_valid),
        .m_axis_ready(vol_axis_m_ready),
        .m_axis_last(vol_axis_m_last)
    );

    // Handle upper bits of the volume data path
    assign vol_axis_m_data[31:24] = 8'b0;

endmodule