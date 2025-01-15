`timescale 1ns / 1ps
`default_nettype none
module axis_clip_distortion (
    input wire clk,
    input wire resetn,
    input wire distortion_enable,
    
    // AXIS Slave Interface (Input)
    input  wire [31:0] s_axis_data,
    input  wire s_axis_valid,
    output reg  s_axis_ready = 1'b1,
    input  wire s_axis_last,
    
    // AXIS Master Interface (Output)
    output reg [31:0] m_axis_data = 0,
    output reg m_axis_valid = 1'b0,
    input  wire m_axis_ready,
    output reg m_axis_last = 1'b0
);

    // Much lower threshold and pre-gain
    localparam signed [23:0] CLIP_THRESHOLD = 24'h080000;  // About 12.5% of full scale
    localparam signed [23:0] NEG_THRESHOLD = -24'h080000;
    localparam signed [23:0] PRE_GAIN = 24'h020000;  // 2x gain

    // Control signals
    wire m_new_word = (m_axis_valid && m_axis_ready);
    wire m_new_packet = (m_new_word && m_axis_last);
    wire s_new_word = (s_axis_valid && s_axis_ready);
    wire s_new_packet = (s_new_word && s_axis_last);
    reg s_new_packet_r = 1'b0;

    // Internal signals for gain stage
    reg signed [47:0] left_gained, right_gained;
    reg signed [23:0] left_clipped, right_clipped;

    always @(posedge clk) begin
        s_new_packet_r <= s_new_packet;
    end

    // Control logic
    always @(posedge clk) begin
        if (!resetn) begin
            m_axis_valid <= 1'b0;
            m_axis_last <= 1'b0;
            s_axis_ready <= 1'b1;
        end else begin
            // Ready logic
            if (s_new_packet)
                s_axis_ready <= 1'b0;
            else if (m_new_packet)
                s_axis_ready <= 1'b1;

            // Valid logic
            if (s_new_packet_r)
                m_axis_valid <= 1'b1;
            else if (m_new_packet)
                m_axis_valid <= 1'b0;

            // Last logic
            if (m_new_packet)
                m_axis_last <= 1'b0;
            else if (m_new_word)
                m_axis_last <= 1'b1;
        end
    end

    // Data processing with gain and aggressive clipping
    always @(posedge clk) begin
        if (!resetn) begin
            m_axis_data <= 32'b0;
            left_gained <= 0;
            right_gained <= 0;
            left_clipped <= 0;
            right_clipped <= 0;
        end else if (s_axis_valid && s_axis_ready) begin
            if (distortion_enable) begin
                // Apply gain first
                left_gained <= $signed(s_axis_data[23:0]) * $signed(PRE_GAIN);
                right_gained <= $signed(s_axis_data[31:8]) * $signed(PRE_GAIN);
                
                // Clip the gained signal
                // Left channel
                if (left_gained[47:24] > CLIP_THRESHOLD)
                    left_clipped <= CLIP_THRESHOLD;
                else if (left_gained[47:24] < NEG_THRESHOLD)
                    left_clipped <= NEG_THRESHOLD;
                else
                    left_clipped <= left_gained[47:24];

                // Right channel
                if (right_gained[47:24] > CLIP_THRESHOLD)
                    right_clipped <= CLIP_THRESHOLD;
                else if (right_gained[47:24] < NEG_THRESHOLD)
                    right_clipped <= NEG_THRESHOLD;
                else
                    right_clipped <= right_gained[47:24];

                // Combine channels
                m_axis_data <= {right_clipped, left_clipped};
            end else begin
                m_axis_data <= s_axis_data;
            end
        end
    end

endmodule