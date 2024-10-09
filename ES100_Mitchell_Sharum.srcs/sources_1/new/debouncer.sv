//`timescale 1ns / 1ps

///*
//    This module is used to debounce the signals asserted by mechanical inputs (buttons, switches) such that 
//    the FPGA perceives them as not several very short inputs in a burst.but as one clearly defined high-low transition
//*/

//module debouncer(
//    input i_clk, i_data,
////    input [COUNTER_WIDTH - 1:0] i_debounce_counter,
//    output o_data
//    );
    
//    /* ---------- CONTROL LOGIC ---------- */
    
//    // adjustable based on how much mitigation is necessary
//    parameter COUNTER_WIDTH = 16;
    
//    //state enumeration
//    enum logic [1:0] {IDLE, COUNTER_START, HIGH, LOW} next_state, state = IDLE;
    
//    // internal signal declarations for state transitions, metastability resolution, and counter
//    logic fsm_out;
//    logic data_metastable, data_stable;
//    logic [COUNTER_WIDTH - 1:0] debounce_counter;
    
    
//    // state machine output assignment
//    assign o_data = fsm_out;
    
//    // state machine logic
//    always_ff @(posedge i_clk) begin
        
//        // transition to the next state
//        state <= next_state;
        
//        // double register the data input so that metastability does not propoagate and create unwanted error
//        data_metastable <= i_data;
//        data_stable <= data_metastable;
        
//        // counter logic
//        case (state)
            
//            // in IDLE state, ensure internal values are 0 and await input edge
//            IDLE : begin       
//                debounce_counter <= 0;
//                fsm_out <= 1'b0;
                
//                // transition logic
//                if (i_data) begin
//                    next_state <= COUNTER_START;
//                end else begin
//                    next_state <= IDLE;
//                end                
//            end
            
//            // COUNTER state delays the "decision" on whether the signal is high or low until counter reaches its max
//            COUNTER_START : begin
            
//                // transition logic
//                if (debounce_counter == COUNTER_WIDTH - 1) begin
//                    if (data_stable) begin
//                        next_state <= HIGH;
//                    end else begin
//                        next_state <= LOW;
//                    end 
                    
//                // else increment the counter
//                end else begin
//                    debounce_counter <= debounce_counter + 1;
//                    next_state <= COUNTER_START;
//                end                 
//            end
            
//            // HIGH state simply asserts a high output
//            HIGH : begin
//                debounce_counter <= 0;
//                fsm_out <= 1'b1;
                
//                // state transition logic
//                if (!data_stable) begin
//                    next_state <= COUNTER_START;
//                end else begin
//                    next_state <= HIGH;
//                end
//            end 
            
//            // LOW state simply asserts a low ouput
//            LOW : begin
//                debounce_counter <= 0;
//                fsm_out <= 1'b0;
                
//                // state transition logic
//                if (data_stable) begin
//                    next_state <= COUNTER_START;
//                end else begin
//                    next_state <= LOW;
//                end
//            end
//        endcase
//    end
//endmodule