`timescale 1ns / 1ps
`default_nettype none
module axis_lowpass_filter (
    input wire clk,
    input wire resetn,
    input wire filter_enable,
    
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
    // Window size options (can try these one at a time):
    // localparam WINDOW_SIZE = 256;  // ~172 Hz cutoff
    // localparam WINDOW_SIZE = 128;  // ~344 Hz cutoff
    localparam WINDOW_SIZE = 64;   // ~689 Hz cutoff
    // localparam WINDOW_SIZE = 32;   // ~1378 Hz cutoff
    // localparam WINDOW_SIZE = 16;   // ~2756 Hz cutoff
    
    // Delay line for audio samples - assuming 24-bit audio in the lower 24 bits
    reg signed [23:0] buffer [WINDOW_SIZE-1:0];
    
    // Accumulator sized appropriately for the window size
    reg signed [31:0] sum;
    
    // Additional stage to smooth the output
    reg signed [31:0] smooth;
    
    // Temporary variable for audio input (declared outside procedural block)
    reg signed [23:0] audio_in;
    
    reg processing;
    assign s_axis_ready = !processing && resetn;
    
    // Calculate the right shift amount based on window size
    localparam SHIFT_AMT = $clog2(WINDOW_SIZE);
    
    integer i;
    
    always @(posedge clk) begin
        if (!resetn) begin
            m_axis_valid <= 0;
            m_axis_data <= 0;
            processing <= 0;
            m_axis_last <= 0;
            sum <= 0;
            smooth <= 0;
            audio_in <= 0;
            for (i = 0; i < WINDOW_SIZE; i = i + 1) begin
                buffer[i] <= 0;
            end
        end else begin
            if (s_axis_valid && s_axis_ready) begin
                // Extract audio sample from lower 24 bits (outside if statement)
                audio_in <= $signed(s_axis_data[23:0]);
                
                if (filter_enable) begin
                    // Update running sum (remove oldest, add newest)
                    sum <= sum - buffer[WINDOW_SIZE-1] + $signed(s_axis_data[23:0]);
                    
                    // Shift values through buffer
                    for (i = WINDOW_SIZE-1; i > 0; i = i - 1) begin
                        buffer[i] <= buffer[i-1];
                    end
                    buffer[0] <= $signed(s_axis_data[23:0]);
                    
                    // Additional smoothing stage with dynamic shift amount
                    smooth <= (smooth + (sum >>> SHIFT_AMT)) >>> 1;
                    
                    // Output the filtered result (preserve upper 8 bits)
                    m_axis_data <= {s_axis_data[31:24], smooth[23:0]};
                end else begin
                    // Pass through when filter is disabled
                    m_axis_data <= s_axis_data;
                    
                    // Keep buffer and smoothing values updated even when disabled
                    for (i = WINDOW_SIZE-1; i > 0; i = i - 1) begin
                        buffer[i] <= buffer[i-1];
                    end
                    buffer[0] <= $signed(s_axis_data[23:0]);
                    sum <= sum - buffer[WINDOW_SIZE-1] + $signed(s_axis_data[23:0]);
                    smooth <= $signed(s_axis_data[23:0]) << SHIFT_AMT;
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