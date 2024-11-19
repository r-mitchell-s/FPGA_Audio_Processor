`timescale 1ns / 1ps
`default_nettype none

module axis_volume_controller #(
    parameter SWITCH_WIDTH = 4,
    parameter DATA_WIDTH = 24
) (
    input wire clk,
    input wire [SWITCH_WIDTH-1:0] sw,
    
    //AXIS SLAVE INTERFACE
    input  wire [DATA_WIDTH-1:0] s_axis_data,
    input  wire s_axis_valid,
    output reg  s_axis_ready = 1'b1,
    input  wire s_axis_last,
    
    // AXIS MASTER INTERFACE
    output reg [DATA_WIDTH-1:0] m_axis_data = 1'b0,
    output reg m_axis_valid = 1'b0,
    input  wire m_axis_ready,
    output reg m_axis_last = 1'b0
);
    localparam MULTIPLIER_WIDTH = 24;
    reg [MULTIPLIER_WIDTH+DATA_WIDTH-1:0] data [1:0];
        
    reg [SWITCH_WIDTH-1:0] sw_sync_r [2:0];
    wire [SWITCH_WIDTH-1:0] sw_sync = sw_sync_r[2];
    reg [MULTIPLIER_WIDTH:0] multiplier = 'b0;
    
    wire m_select = m_axis_last;
    wire m_new_word = (m_axis_valid == 1'b1 && m_axis_ready == 1'b1) ? 1'b1 : 1'b0;
    wire m_new_packet = (m_new_word == 1'b1 && m_axis_last == 1'b1) ? 1'b1 : 1'b0;
    
    wire s_select = s_axis_last;
    wire s_new_word = (s_axis_valid == 1'b1 && s_axis_ready == 1'b1) ? 1'b1 : 1'b0;
    wire s_new_packet = (s_new_word == 1'b1 && s_axis_last == 1'b1) ? 1'b1 : 1'b0;
    reg s_new_packet_r = 1'b0;
    
    always@(posedge clk) begin
        sw_sync_r[2] <= sw_sync_r[1];
        sw_sync_r[1] <= sw_sync_r[0];
        sw_sync_r[0] <= sw;
        
        multiplier <= {sw_sync,{MULTIPLIER_WIDTH{1'b0}}} / {SWITCH_WIDTH{1'b1}};
        s_new_packet_r <= s_new_packet;
    end
    
    always@(posedge clk)
        if (s_new_word == 1'b1)
            data[s_select] <= {{MULTIPLIER_WIDTH{s_axis_data[DATA_WIDTH-1]}}, s_axis_data};
        else if (s_new_packet_r == 1'b1) begin
            data[0] <= $signed(data[0]) * multiplier;
            data[1] <= $signed(data[1]) * multiplier;
        end
        
    always@(posedge clk)
        if (s_new_packet_r == 1'b1)
            m_axis_valid <= 1'b1;
        else if (m_new_packet == 1'b1)
            m_axis_valid <= 1'b0;
            
    always@(posedge clk)
        if (m_new_packet == 1'b1)
            m_axis_last <= 1'b0;
        else if (m_new_word == 1'b1)
            m_axis_last <= 1'b1;
            
    always@(m_axis_valid, data[0], data[1], m_select)
        if (m_axis_valid == 1'b1)
            m_axis_data = data[m_select][MULTIPLIER_WIDTH+DATA_WIDTH-1:MULTIPLIER_WIDTH];
        else
            m_axis_data = 'b0;
            
    always@(posedge clk)
        if (s_new_packet == 1'b1)
            s_axis_ready <= 1'b0;
        else if (m_new_packet == 1'b1)
            s_axis_ready <= 1'b1;
endmodule
