`timescale 1ns / 1ps
`default_nettype none
module axis_ring_modulator (
    input wire clk,
    input wire resetn,
    input wire ringmod_enable,
    input wire [1:0] ringmod_freq, // Controls the frequency of the modulation
    
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
    
    // Carrier oscillator
    reg [15:0] carrier_counter;
    wire [15:0] carrier_increment;
    reg signed [15:0] carrier_value;
    
    // Set carrier increment based on ringmod_freq (determines modulation frequency)
    assign carrier_increment = (ringmod_freq == 2'b00) ? 16'd100 :    // ~150Hz
                              (ringmod_freq == 2'b01) ? 16'd200 :    // ~300Hz
                              (ringmod_freq == 2'b10) ? 16'd400 :    // ~600Hz
                                                        16'd800;     // ~1200Hz
    
    // Temp signals for calculation
    reg signed [23:0] audio_in;
    reg signed [23:0] audio_out;
    reg signed [39:0] temp_audio;
    reg signed [23:0] scaled_input;
    reg signed [15:0] scaled_carrier;
    
    // Generate carrier waveform and handle processing
    always @(posedge clk) begin
        if (!resetn) begin
            carrier_counter <= 0;
            carrier_value <= 0;
            m_axis_valid <= 0;
            m_axis_data <= 0;
            processing <= 0;
            m_axis_last <= 0;
            audio_in <= 0;
            audio_out <= 0;
            temp_audio <= 0;
            scaled_input <= 0;
            scaled_carrier <= 0;
        end else begin
            // Update carrier oscillator (triangle wave)
            carrier_counter <= carrier_counter + (ringmod_enable ? carrier_increment : 16'd0);
            
            // Create a triangle wave oscillator
            if (carrier_counter[15]) begin
                // Falling edge of triangle
                carrier_value <= 16'h7FFF - {1'b0, carrier_counter[14:0]};
            end else begin
                // Rising edge of triangle
                carrier_value <= carrier_counter[14:0];
            end
            
            // Process audio data
            if (s_axis_valid && s_axis_ready) begin
                // Get input signal
                audio_in <= s_axis_data[23:0];
                
                if (ringmod_enable) begin
                    // Scale input to avoid overflow
                    scaled_input = audio_in >>> 1;  // Scale by 50%
                    
                    // Normalize carrier to [-0.5, 0.5] range (not [-1, 1])
                    // to further prevent overflow
                    scaled_carrier = ($signed({1'b0, carrier_value}) - 16'sd32768) >>> 1;
                    
                    // Multiply with both inputs scaled to prevent overflow
                    temp_audio = $signed(scaled_input) * $signed(scaled_carrier);
                    
                    // Result is now effectively scaled by 0.25, preventing overflow
                    audio_out = temp_audio >>> 14;  // Adjusted shift to account for carrier scaling
                    
                    // Create output with preserved upper 8 bits
                    m_axis_data <= {s_axis_data[31:24], audio_out[23:0]};
                end else begin
                    // Pass through unmodified
                    m_axis_data <= s_axis_data;
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