## Clock signal from ZedBoard's 100MHz oscillator
set_property -dict { PACKAGE_PIN Y9 IOSTANDARD LVCMOS33 } [get_ports clk_100mhz]
create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} [get_ports clk_100mhz]

## Reset signal - connect to BTNU (Up button)
set_property -dict { PACKAGE_PIN T18 IOSTANDARD LVCMOS33 } [get_ports reset]

## Volume control switches
set_property -dict { PACKAGE_PIN F22 IOSTANDARD LVCMOS33 } [get_ports {sw[0]}]
set_property -dict { PACKAGE_PIN G22 IOSTANDARD LVCMOS33 } [get_ports {sw[1]}]
set_property -dict { PACKAGE_PIN H22 IOSTANDARD LVCMOS33 } [get_ports {sw[2]}]
set_property -dict { PACKAGE_PIN F21 IOSTANDARD LVCMOS33 } [get_ports {sw[3]}]

## Pmod JA Connections (12-pin) for I2S2:
# Top row - Line Out (DAC)
set_property -dict { PACKAGE_PIN Y11  IOSTANDARD LVCMOS33 } [get_ports tx_mclk]
set_property -dict { PACKAGE_PIN AA11 IOSTANDARD LVCMOS33 } [get_ports tx_lrck]
set_property -dict { PACKAGE_PIN Y10  IOSTANDARD LVCMOS33 } [get_ports tx_sclk]
set_property -dict { PACKAGE_PIN AA9  IOSTANDARD LVCMOS33 } [get_ports tx_sdout]

# Bottom row - Line In (ADC)
set_property -dict { PACKAGE_PIN AB11 IOSTANDARD LVCMOS33 } [get_ports rx_mclk]
set_property -dict { PACKAGE_PIN AB10 IOSTANDARD LVCMOS33 } [get_ports rx_lrck]
set_property -dict { PACKAGE_PIN AB9  IOSTANDARD LVCMOS33 } [get_ports rx_sclk]
set_property -dict { PACKAGE_PIN AA8  IOSTANDARD LVCMOS33 } [get_ports rx_sdin]

## Configuration and Voltage
set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]