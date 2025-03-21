// - - - - - VOLUME CONTROLLER - - - - - //

`timescale 1ns / 1ps
`default_nettype none
module axis_volume_controller #(
    parameter SWITCH_WIDTH = 2,
    parameter DATA_WIDTH = 32
) (
    input wire clk,
    input wire [SWITCH_WIDTH-1:0] sw,
    
    // AXIS Slave Interface
    input  wire [DATA_WIDTH-1:0] s_axis_data,
    input  wire s_axis_valid,
    output wire s_axis_ready,
    input  wire s_axis_last,
    
    // AXIS Master Interface
    output reg [DATA_WIDTH-1:0] m_axis_data,
    output reg m_axis_valid,
    input  wire m_axis_ready,
    output reg m_axis_last
);
    // Simple state machine
    reg busy;
    assign s_axis_ready = !busy && m_axis_ready;
    
    // Register for synchronized switches
    reg [1:0] vol_level;
    
    // Intermediate signal for signed arithmetic (24-bit audio in lower bits)
    wire signed [23:0] audio_in = s_axis_data[23:0];
    reg signed [23:0] audio_out;
    
    // Synchronize switches
    always @(posedge clk) begin
        vol_level <= sw[1:0];
    end
    
    // Process single audio sample with proper signed arithmetic
    always @(posedge clk) begin
        if (s_axis_valid && s_axis_ready) begin
            case(vol_level)
                2'b00: begin // -18dB
                    audio_out <= {{3{audio_in[23]}}, audio_in[23:3]};
                end
                2'b01: begin // -12dB
                    audio_out <= {{2{audio_in[23]}}, audio_in[23:2]};
                end
                2'b10: begin // -6dB
                    audio_out <= {audio_in[23], audio_in[23:1]};
                end
                2'b11: begin // Full volume
                    audio_out <= audio_in;
                end
            endcase
            
            // Preserve upper 8 bits from input
            m_axis_data <= {s_axis_data[31:24], audio_out[23:0]};
            m_axis_valid <= 1'b1;
            m_axis_last <= s_axis_last;
            busy <= 1'b1;
        end
        else if (m_axis_valid && m_axis_ready) begin
            m_axis_valid <= 1'b0;
            busy <= 1'b0;
        end
    end
endmodules