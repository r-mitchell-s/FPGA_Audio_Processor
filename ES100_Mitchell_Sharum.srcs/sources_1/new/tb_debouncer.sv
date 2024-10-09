//`timescale 1ns / 1ps

//module tb_debouncer;

//  // Testbench signals
//  reg i_clk;
//  reg i_data;
//  wire o_data;

//  // Clock period
//  parameter CLK_PERIOD = 20;

//  // Instantiate the debouncer module
//  debouncer #(.COUNTER_WIDTH(16)) uut (
//    .i_clk(i_clk),
//    .i_data(i_data),
//    .o_data(o_data)
//  );

//  // Clock generation
//  initial begin
//    i_clk = 0;
//    forever #(CLK_PERIOD/2) i_clk = ~i_clk;
//  end

//  // Test sequence
//  initial begin
//    // Initialize input
//    i_data = 0;

//    // Wait for some time and apply a "bouncing" signal
//    #100;

//    // Simulate bouncing (rapid high-low transitions)
//    i_data = 1;
//    #1 i_data = 0;
//    #1 i_data = 1;
//    #1 i_data = 0;
//    #1 i_data = 1;
//    #1 i_data = 0;
//    #1 i_data = 1;

//    // Allow the debouncer to stabilize
//    #200;
    
//    // Apply a clean high signal (simulate pressing the button)
//    i_data = 1;
//    #200;
    
//    // simulate switching off
//    #1 i_data = 0;
//    #1 i_data = 1;
//    #1 i_data = 0;
//    #1 i_data = 1;
//    #1 i_data = 0;
//    #1 i_data = 1;
//    #1 i_data = 0;
    
//    // Release the button (signal goes low)
//    i_data = 0;
//    #1000;

//    // Finish the simulation
//    $finish;
//  end

//  // Monitor the inputs and outputs
//  initial begin
//    $monitor("Time = %0t | i_data = %b | o_data = %b", $time, i_data, o_data);
//  end

//endmodule

