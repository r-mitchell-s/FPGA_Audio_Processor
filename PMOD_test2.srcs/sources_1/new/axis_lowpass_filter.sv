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
    // localparam WINDOW_SIZE = 256;  // ~172 Hz cutoff (current)
    // localparam WINDOW_SIZE = 128;  // ~344 Hz cutoff
    localparam WINDOW_SIZE = 64;   // ~689 Hz cutoff
    // localparam WINDOW_SIZE = 32;   // ~1378 Hz cutoff
    // localparam WINDOW_SIZE = 16;   // ~2756 Hz cutoff
    
    // Delay lines for left and right channels
    reg signed [23:0] buffer_left [WINDOW_SIZE-1:0];
    reg signed [23:0] buffer_right [WINDOW_SIZE-1:0];
    
    // Accumulator size based on window size
    reg signed [31:0] sum_left;
    reg signed [31:0] sum_right;
    
    // Additional stage to smooth the output
    reg signed [31:0] smooth_left;
    reg signed [31:0] smooth_right;
    
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
            sum_left <= 0;
            sum_right <= 0;
            smooth_left <= 0;
            smooth_right <= 0;
            for (i = 0; i < WINDOW_SIZE; i = i + 1) begin
                buffer_left[i] <= 0;
                buffer_right[i] <= 0;
            end
        end else begin
            if (s_axis_valid && s_axis_ready) begin
                if (filter_enable) begin
                    // Update running sum
                    sum_left <= sum_left - buffer_left[WINDOW_SIZE-1] + $signed(s_axis_data[23:0]);
                    sum_right <= sum_right - buffer_right[WINDOW_SIZE-1] + $signed(s_axis_data[31:8]);
                    
                    // Shift values through buffer
                    for (i = WINDOW_SIZE-1; i > 0; i = i - 1) begin
                        buffer_left[i] <= buffer_left[i-1];
                        buffer_right[i] <= buffer_right[i-1];
                    end
                    buffer_left[0] <= s_axis_data[23:0];
                    buffer_right[0] <= s_axis_data[31:8];
                    
                    // Additional smoothing stage with dynamic shift amount
                    smooth_left <= (smooth_left + (sum_left >>> SHIFT_AMT)) >>> 1;
                    smooth_right <= (smooth_right + (sum_right >>> SHIFT_AMT)) >>> 1;
                    
                    // Output the filtered result
                    m_axis_data <= {smooth_right[23:0], smooth_left[23:0]};
                end else begin
                    m_axis_data <= s_axis_data;
                    
                    // Keep buffers and smoothing values updated
                    for (i = WINDOW_SIZE-1; i > 0; i = i - 1) begin
                        buffer_left[i] <= buffer_left[i-1];
                        buffer_right[i] <= buffer_right[i-1];
                    end
                    buffer_left[0] <= s_axis_data[23:0];
                    buffer_right[0] <= s_axis_data[31:8];
                    sum_left <= sum_left - buffer_left[WINDOW_SIZE-1] + $signed(s_axis_data[23:0]);
                    sum_right <= sum_right - buffer_right[WINDOW_SIZE-1] + $signed(s_axis_data[31:8]);
                    smooth_left <= $signed(s_axis_data[23:0]) << SHIFT_AMT;
                    smooth_right <= $signed(s_axis_data[31:8]) << SHIFT_AMT;
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