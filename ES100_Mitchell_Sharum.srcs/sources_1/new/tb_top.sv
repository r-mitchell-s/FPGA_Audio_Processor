`timescale 1ns / 1ps

module tb_top;

// Inputs
reg i_codec_bit_clock;
reg i_codec_lr_clock;
reg i_codec_adc_data;

// Outputs
wire o_codec_mclock;
wire o_codec_dac_data;

// Instantiate the Unit Under Test (UUT)
top uut (
    .i_codec_bit_clock(i_codec_bit_clock),
    .i_codec_lr_clock(i_codec_lr_clock),
    .i_codec_adc_data(i_codec_adc_data),
    .o_codec_mclock(o_codec_mclock),
    .o_codec_dac_data(o_codec_dac_data)
);

// Clock generation for bit clock and LR clock
initial begin
    i_codec_bit_clock = 0;
    forever #22.675 i_codec_bit_clock = ~i_codec_bit_clock;  // MCLK ~22.579 MHz -> Period ~44.1 ns
end

initial begin
    i_codec_lr_clock = 0;
    forever #2907 i_codec_lr_clock = ~i_codec_lr_clock;  // LR clock toggles every 64 SCLK periods (~2.907 µs)
end

//// Test sequence
//initial begin
//    // Initialize Inputs
//    i_codec_adc_data = 0;

//    // Reset all signals
//    #10;

//    // Simulate a stream of ADC data
//    repeat (10) begin
//        @(posedge i_codec_bit_clock);
//        i_codec_adc_data = 1'b1;  // Send some test data
//        @(posedge i_codec_bit_clock);
//        i_codec_adc_data = 1'b0;  // Toggle data bits to simulate audio samples
//    end

//    // Observe outputs in waveform for verification
//    #500000 $finish;
//end

endmodule
