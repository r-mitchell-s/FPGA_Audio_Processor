`timescale 1ns / 1ps
`default_nettype none

module axis_noise_gate (
    input wire clk,
    input wire resetn,
    input wire gate_enable,
    input wire [1:0] threshold_level, // 00: very low, 01: low, 10: medium, 11: high
    
    // AXIS Slave Interface
    input  wire [31:0] s_axis_data,
    input  wire s_axis_valid,
    output wire s_axis_ready,
    input  wire s_axis_last,
    
    // AXIS Master Interface
    output reg [31:0] m_axis_data,
    output reg m_axis_valid,
    input  wire m_axis_ready,
    output reg m_axis_last
);
    // Threshold levels
    localparam [23:0] THRESHOLD_VERY_LOW = 24'h020000; // ~7.8% of full scale
    localparam [23:0] THRESHOLD_LOW      = 24'h080000; // ~31.2% of full scale
    localparam [23:0] THRESHOLD_MEDIUM   = 24'h180000; // ~46.9% of full scale
    localparam [23:0] THRESHOLD_HIGH     = 24'h300000; // ~59.4% of full scale
    
    // State machine
    reg processing;
    assign s_axis_ready = !processing && resetn;
    
    // Selected threshold
    reg [23:0] current_threshold;
    
    // Simple gate variables
    reg gate_open;
    reg [7:0] gate_gain; // 0-255 (0 = muted, 255 = full volume)
    
    // Set threshold based on input level
    always @(*) begin
        case(threshold_level)
            2'b00: current_threshold = THRESHOLD_VERY_LOW;
            2'b01: current_threshold = THRESHOLD_LOW;
            2'b10: current_threshold = THRESHOLD_MEDIUM;
            2'b11: current_threshold = THRESHOLD_HIGH;
            default: current_threshold = THRESHOLD_LOW;
        endcase
    end
    
    // Calculate absolute value (24-bit)
    function [23:0] get_abs;
        input [23:0] value;
        begin
            get_abs = (value[23]) ? (~value + 1'b1) : value;
        end
    endfunction
    
    reg [23:0] signal_abs;
    reg signed [23:0] audio_out;
    
    // Main processing
    always @(posedge clk) begin
        if (!resetn) begin
            m_axis_valid <= 0;
            m_axis_data <= 0;
            processing <= 0;
            m_axis_last <= 0;
            gate_open <= 0;
            gate_gain <= 0;
            signal_abs <= 0;
            audio_out <= 0;
        end else begin
            // Audio processing
            if (s_axis_valid && s_axis_ready) begin
                // Get absolute value of the audio sample
                signal_abs <= get_abs(s_axis_data[23:0]);
                
                if (gate_enable) begin
                    // Simple gating - open if above threshold, close if below
                    if (signal_abs >= current_threshold) begin
                        gate_open <= 1;
                    end else begin
                        gate_open <= 0;
                    end
                    
                    // Simple gain application - either full on or full off
                    gate_gain <= gate_open ? 8'd255 : 8'd0;
                    
                    // Apply gate - simple method
                    if (gate_gain > 0) begin
                        // Gate open - pass signal through directly
                        audio_out <= s_axis_data[23:0];
                    end else begin
                        // Gate closed - output zero
                        audio_out <= 24'd0;
                    end
                    
                    // Preserve upper 8 bits from input
                    m_axis_data <= {s_axis_data[31:24], audio_out[23:0]};
                end else begin
                    // Gate disabled, pass through original signal
                    m_axis_data <= s_axis_data;
                    
                    // Reset gate state
                    gate_open <= 0;
                    gate_gain <= 0;
                end
                
                m_axis_valid <= 1;
                m_axis_last <= s_axis_last;
                processing <= 1;
            end else if (m_axis_valid && m_axis_ready) begin
                m_axis_valid <= 0;
                processing <= 0;
            end
        end
    end
endmodule