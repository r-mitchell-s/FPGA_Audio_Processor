# create and configure the clocking wizard IP with parameters for this project
create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz_0

set_property -dict [list \
    CONFIG.PRIMITIVE {MMCM} \
    CONFIG.PRIM_IN_FREQ {100.000} \
    CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {22.591} \
    CONFIG.CLK_OUT1_PORT {axis_clk} \
    CONFIG.CLKOUT1_DRIVES {BUFGCE} \
    CONFIG.USE_RESET {true} \
    CONFIG.RESET_TYPE {ACTIVE_HIGH} \
    CONFIG.USE_LOCKED {true} \
    CONFIG.USE_PHASE_ALIGNMENT {false} \
    CONFIG.FEEDBACK_SOURCE {FDBK_AUTO} \
    CONFIG.MMCM_DIVCLK_DIVIDE {6} \
    CONFIG.MMCM_CLKFBOUT_MULT_F {48.625} \
    CONFIG.MMCM_CLKOUT0_DIVIDE_F {35.875} \
    CONFIG.MMCM_COMPENSATION {ZHOLD} \
    CONFIG.USE_SAFE_CLOCK_STARTUP {true} \
] [get_ips clk_wiz_0]

# generate the IP
generate_target all [get_ips clk_wiz_0]