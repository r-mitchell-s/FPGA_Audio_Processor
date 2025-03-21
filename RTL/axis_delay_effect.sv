`timescale 1ns / 1ps
`default_nettype none
module axis_delay_effect #(
    parameter MAX_DELAY_DEPTH = 20480,  // Maximum delay in samples (reduced to avoid synthesis issues)
    parameter DELAY_WIDTH = 16          // Width of the delay setting input
) (
    input wire clk,
    input wire resetn,
    input wire delay_enable,
    input wire [DELAY_WIDTH-1:0] delay_length, // Number of samples to delay
    input wire [7:0] feedback_level,          // 0-255 for feedback amount (255 = 100%)
    input wire [7:0] wet_level,               // 0-255 for wet/dry mix (255 = 100% wet)
    
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
    // State machine for processing
    reg processing;
    assign s_axis_ready = !processing && resetn;
    
    // Simplified approach - use a single smaller buffer
    // This helps avoid synthesis issues
    localparam ACTUAL_MAX_DEPTH = 16384; // 16k samples (~341ms at 48kHz)
    
    // Delay buffer with BRAM inference attribute
    (* ram_style = "block" *) reg signed [23:0] delay_buffer [ACTUAL_MAX_DEPTH-1:0];
    
    // Current read/write positions
    reg [13:0] write_ptr; // 14 bits for 16384 depth
    reg [13:0] read_ptr;
    
    // Calculate actual delay length (bounded by ACTUAL_MAX_DEPTH)
    wire [13:0] actual_delay = (delay_length > ACTUAL_MAX_DEPTH) ? 
                               ACTUAL_MAX_DEPTH[13:0] : 
                               delay_length[13:0];
    
    // Temporary variables for processing
    reg signed [23:0] audio_in;
    reg signed [23:0] delayed_sample;
    reg signed [23:0] output_sample;
    
    // Main processing logic
    always @(posedge clk) begin
        if (!resetn) begin
            write_ptr <= 0;
            read_ptr <= 0;
            m_axis_valid <= 0;
            m_axis_data <= 0;
            processing <= 0;
            m_axis_last <= 0;
            audio_in <= 0;
            delayed_sample <= 0;
            output_sample <= 0;
        end 
        else begin
            // Process audio data
            if (s_axis_valid && s_axis_ready) begin
                // Extract input audio
                audio_in <= s_axis_data[23:0];
                
                // Calculate read pointer - simplified to avoid timing issues
                if (write_ptr >= actual_delay)
                    read_ptr <= write_ptr - actual_delay;
                else
                    read_ptr <= (ACTUAL_MAX_DEPTH - actual_delay) + write_ptr;
                
                // Read the delayed sample from buffer
                delayed_sample <= delay_buffer[read_ptr];
                
                if (delay_enable) begin
                    // Keep the feedback calculation simple to start
                    // Store the input audio plus scaled feedback
                    delay_buffer[write_ptr] <= audio_in + 
                                              ((delayed_sample * $signed({1'b0, feedback_level})) >>> 8);
                    
                    // Simple wet/dry mix
                    output_sample <= ((audio_in * $signed({1'b0, 8'd255 - wet_level})) >>> 8) + 
                                     ((delayed_sample * $signed({1'b0, wet_level})) >>> 8);
                    
                    // Update output with mixed signal
                    m_axis_data <= {s_axis_data[31:24], output_sample};
                end 
                else begin
                    // Pass through when effect is disabled, but still update buffer
                    delay_buffer[write_ptr] <= audio_in;
                    m_axis_data <= s_axis_data;
                end
                
                // Increment write pointer with wraparound
                if (write_ptr == ACTUAL_MAX_DEPTH - 1)
                    write_ptr <= 0;
                else
                    write_ptr <= write_ptr + 1;
                
                m_axis_valid <= 1;
                m_axis_last <= s_axis_last;
                processing <= 1;
            end 
            else if (m_axis_valid && m_axis_ready) begin
                m_axis_valid <= 0;
                processing <= 0;
            end
        end
    end
endmodule