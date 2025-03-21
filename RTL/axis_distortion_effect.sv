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
    // Simple hard clipping thresholds
    // Adjust these values to control where clipping occurs
    localparam signed [23:0] POS_THRESHOLD = 24'h400000;  // 25% of full scale
    localparam signed [23:0] NEG_THRESHOLD = -24'h400000; // -25% of full scale

    // Control signals
    wire m_new_word = (m_axis_valid && m_axis_ready);
    wire m_new_packet = (m_new_word && m_axis_last);
    wire s_new_word = (s_axis_valid && s_axis_ready);
    wire s_new_packet = (s_new_word && s_axis_last);
    reg s_new_packet_r = 1'b0;
    
    // Register delay of packet signal
    always @(posedge clk) begin
        s_new_packet_r <= s_new_packet;
    end
    
    // Control logic for AXIS protocol
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
    
    // Pure hard clipping with single-cycle latency
    always @(posedge clk) begin
        if (!resetn) begin
            m_axis_data <= 32'b0;
        end else if (s_axis_valid && s_axis_ready) begin
            if (distortion_enable) begin
                // Extract audio sample
                reg signed [23:0] audio_in;
                reg signed [23:0] audio_out;
                
                // Get input sample
                audio_in = $signed(s_axis_data[23:0]);
                
                // Apply simple threshold-based hard clipping
                // This preserves the exact input below threshold
                if (audio_in > POS_THRESHOLD)
                    audio_out = POS_THRESHOLD;
                else if (audio_in < NEG_THRESHOLD)
                    audio_out = NEG_THRESHOLD;
                else
                    audio_out = audio_in; // Pass through unmodified when within threshold
                
                // Preserve upper 8 bits
                m_axis_data <= {s_axis_data[31:24], audio_out};
            end else begin
                // Pass through when distortion is disabled
                m_axis_data <= s_axis_data;
            end
        end
    end
    
endmodule