`timescale 1ns / 1ps
`default_nettype none

module axis_delay_effect #(
    parameter DATA_WIDTH = 24,
    parameter DELAY_LENGTH = 22050  // 0.5 second delay at 44.1kHz
) (
    input wire clk,
    input wire resetn,
    input wire [1:0] sw,  // Switch control for delay mix
    
    // AXIS Slave Interface (Input)
    input  wire [31:0] s_axis_data,      // Changed to 32-bit for full stereo sample
    input  wire s_axis_valid,
    output reg  s_axis_ready = 1'b1,
    input  wire s_axis_last,
    
    // AXIS Master Interface (Output)
    output reg [31:0] m_axis_data = 0,   // Changed to 32-bit for full stereo sample
    output reg m_axis_valid = 1'b0,
    input  wire m_axis_ready,
    output reg m_axis_last = 1'b0
);

    // Separate delay buffers for left and right channels
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] delay_buffer_left [DELAY_LENGTH-1:0];
    (* ram_style = "block" *) reg [DATA_WIDTH-1:0] delay_buffer_right [DELAY_LENGTH-1:0];
    
    // Buffer pointers
    reg [$clog2(DELAY_LENGTH)-1:0] write_ptr = 0;
    reg [$clog2(DELAY_LENGTH)-1:0] read_ptr = 0;
    
    // FSM states
    localparam IDLE = 0;
    localparam APPLY_GAIN = 1;
    localparam GENERATE_OUTPUT = 2;
    reg [1:0] state = IDLE;
    
    // Internal registers for processing
    reg [DATA_WIDTH-1:0] current_sample_left;
    reg [DATA_WIDTH-1:0] current_sample_right;
    reg [DATA_WIDTH-1:0] delayed_sample_left;
    reg [DATA_WIDTH-1:0] delayed_sample_right;
    reg [DATA_WIDTH-1:0] feedback_sample_left;
    reg [DATA_WIDTH-1:0] feedback_sample_right;
    
    // Control signals
    wire s_new_word = s_axis_valid && s_axis_ready;
    wire m_new_word = m_axis_valid && m_axis_ready;
    
    // Feedback gain based on switches (0 to 0.75)
    wire [7:0] feedback_gain = {sw, 6'b000000};  // Scaled for fixed-point math
    
    always @(posedge clk) begin
        if (!resetn) begin
            state <= IDLE;
            write_ptr <= 0;
            read_ptr <= DELAY_LENGTH/2;
            m_axis_valid <= 0;
            s_axis_ready <= 1;
            m_axis_data <= 0;
        end else begin
            case (state)
                IDLE: begin
                    m_axis_valid <= 0;
                    if (s_new_word) begin
                        // Split incoming stereo sample
                        current_sample_left <= s_axis_data[23:0];
                        current_sample_right <= s_axis_data[31:8];  // Upper 24 bits
                        
                        // Read delayed samples
                        delayed_sample_left <= delay_buffer_left[read_ptr];
                        delayed_sample_right <= delay_buffer_right[read_ptr];
                        
                        s_axis_ready <= 0;
                        state <= APPLY_GAIN;
                    end
                end
                
                APPLY_GAIN: begin
                    // Apply feedback gain to both channels
                    feedback_sample_left <= (delayed_sample_left * feedback_gain) >>> 8;
                    feedback_sample_right <= (delayed_sample_right * feedback_gain) >>> 8;
                    state <= GENERATE_OUTPUT;
                end
                
                GENERATE_OUTPUT: begin
                    // Mix current and feedback samples for both channels
                    m_axis_data[23:0] <= current_sample_left + feedback_sample_left;
                    m_axis_data[31:8] <= current_sample_right + feedback_sample_right;
                    m_axis_valid <= 1;
                    m_axis_last <= s_axis_last;
                    
                    // Store the outputs in delay buffers (feedback path)
                    delay_buffer_left[write_ptr] <= current_sample_left + feedback_sample_left;
                    delay_buffer_right[write_ptr] <= current_sample_right + feedback_sample_right;
                    
                    // Update pointers
                    if (write_ptr == DELAY_LENGTH-1) begin
                        write_ptr <= 0;
                    end else begin
                        write_ptr <= write_ptr + 1;
                    end
                    
                    if (read_ptr == DELAY_LENGTH-1) begin
                        read_ptr <= 0;
                    end else begin
                        read_ptr <= read_ptr + 1;
                    end
                    
                    s_axis_ready <= 1;
                    state <= IDLE;
                end
                
                default: state <= IDLE;
            endcase
        end
    end

endmodule