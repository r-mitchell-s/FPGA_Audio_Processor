# # get the directory where this script is being run (newly created by the bash script)
# set project_dir [pwd]

# # get root directory (one level up from pwd)
# set root_dir [file normalize [file join $project_dir ..]]

# # get datetime from the project directory name
# set datetime [file tail $project_dir]
# regexp {Project_(.+)$} $datetime -> datetime

# # create the project 
# create_project "Project_$datetime" $project_dir -part xc7z020clg484-1

# # add source files (RTL, constraints, IPs)
# set rtl_dir [file join $root_dir "RTL"]
# set constraints_dir [file join $root_dir "Constraints"]
# set ips_dir [file join $root_dir "IPs"]

# # get a list of all .sv files in the RTL directory and add them to sources
# set rtl_files [glob -nocomplain $rtl_dir/*.sv]
# foreach file $rtl_files {
#     add_files $file
# }

# # add constraints files
# set constraints_files [glob -nocomplain $constraints_dir/*.xdc]
# foreach file $constraints_files {
#     add_files $file
# }

# # - - - - - IP CREATION AND GENERATION - - - - - #
# create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz_0

# # configure the clk_wiz_0
# set_property -dict [list \
#     CONFIG.PRIMITIVE {MMCM} \
#     CONFIG.PRIM_IN_FREQ {100.000} \
#     CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {22.591} \
#     CONFIG.CLK_OUT1_PORT {axis_clk} \
#     CONFIG.CLKOUT1_DRIVES {BUFGCE} \
#     CONFIG.USE_RESET {true} \
#     CONFIG.RESET_TYPE {ACTIVE_HIGH} \
#     CONFIG.USE_LOCKED {true} \
#     CONFIG.USE_PHASE_ALIGNMENT {false} \
#     CONFIG.FEEDBACK_SOURCE {FDBK_AUTO} \
#     CONFIG.MMCM_DIVCLK_DIVIDE {6} \
#     CONFIG.MMCM_CLKFBOUT_MULT_F {48.625} \
#     CONFIG.MMCM_CLKOUT0_DIVIDE_F {35.875} \
#     CONFIG.MMCM_COMPENSATION {ZHOLD} \
#     CONFIG.USE_SAFE_CLOCK_STARTUP {true} \
# ] [get_ips clk_wiz_0]

# # generate the IP
# generate_target all [get_ips clk_wiz_0]

# # directory for storing synthesis reports
# file mkdir ../../Reports

# # run synthesis
# launch_runs synth_1
# wait_on_run synth_1

# # save checkpoint before writing reports
# write_checkpoint -force ../../Reports/post_synth.dcp

# # generate synthess reports and write them to "Reports" directory
# report_timing_summary -file ../../Reports/timing_report.rpt
# report_utilization -file ../../Reports/synthesis_utilization_report.rpt
# report_area -file ../../Reports/synthesis_area_report.rpt
# report_clk -file ../../Reports/synthesis_clock_report.rpt
# report_fanout -file ../../Reports/synthesis_fanout_report.rpt
# report_registers -file ../../Reports/synthesis_registers_report.rpt
# report_nets -file ../../Reports/synthesis_netlist_report.rpt
# report_power -file ../../Reports/synthesis_power_report.rpt

# # run implemenation ad open elaborated design
# launch_runs impl_1 -to_step write_bitstream
# wait_on_run impl_1
# open_run impl_1

# # now write the bitstream
# write_bitstream -force $project_dir/Project_$datetime.bit

# # open the hardware manager
# open_hw_manager

# # connect to the hardware server
# connect_hw_server

# # open the hardware target
# open_hw_target

# # set programming properties (for Zynq)
# set_property PROGRAM.FILE {$project_dir/Project_$datetime.bit} [current_hw_device]
# set_property PROBES.FILE {} [current_hw_device]
# set_property FULL_PROBES.FILE {} [current_hw_device]

# # program the device
# program_hw_devices [current_hw_device]

# # close hw manager ad exit vivado
# close_hw_manager
# # exit

# get the directory where this script is being run (newly created by the bash script)
set project_dir [pwd]
puts "Current working directory: $project_dir"

# get root directory (one level up from pwd)
set root_dir [file normalize [file join $project_dir ..]]
puts "Root directory: $root_dir"

# get datetime from the project directory name
set datetime [file tail $project_dir]
regexp {Project_(.+)$} $datetime -> datetime

# create the project 
create_project "Project_$datetime" $project_dir -part xc7z020clg484-1

# add source files (RTL, constraints, IPs)
set rtl_dir [file join $root_dir "RTL"]
set constraints_dir [file join $root_dir "Constraints"]
set ips_dir [file join $root_dir "IPs"]

# get a list of all .sv files in the RTL directory and add them to sources
set rtl_files [glob -nocomplain $rtl_dir/*.sv]
foreach file $rtl_files {
    add_files $file
}

# add constraints files
set constraints_files [glob -nocomplain $constraints_dir/*.xdc]
foreach file $constraints_files {
    add_files $file
}

# - - - - - IP CREATION AND GENERATION - - - - - #
create_ip -name clk_wiz -vendor xilinx.com -library ip -version 6.0 -module_name clk_wiz_0

# configure the clk_wiz_0
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

# Create the Reports directory before synthesis
set reports_dir [file normalize [file join $project_dir ../../Reports]]
puts "Reports directory: $reports_dir"

# Try creating the directory and catch errors
try {
    file mkdir $reports_dir
} catch {errorMsg} {
    puts "Error creating directory: $errorMsg"
}

# run synthesis
launch_runs synth_1
wait_on_run synth_1

# save checkpoint before writing reports
write_checkpoint -force $reports_dir/post_synth.dcp

# generate synthesis reports and write them to "Reports" directory
report_timing_summary -file $reports_dir/timing_report.rpt
report_utilization -file $reports_dir/synthesis_utilization_report.rpt
report_area -file $reports_dir/synthesis_area_report.rpt
report_clocks -file $reports_dir/synthesis_clock_report.rpt
report_fanout -file $reports_dir/synthesis_fanout_report.rpt
report_registers -file $reports_dir/synthesis_registers_report.rpt
report_nets -file $reports_dir/synthesis_netlist_report.rpt
report_power -file $reports_dir/synthesis_power_report.rpt

# run implementation and open elaborated design
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1
open_run impl_1

# now write the bitstream
write_bitstream -force $project_dir/Project_$datetime.bit

# open the hardware manager
open_hw_manager

# connect to the hardware server
connect_hw_server

# open the hardware target
open_hw_target

# set programming properties (for Zynq)
set_property PROGRAM.FILE {$project_dir/Project_$datetime.bit} [current_hw_device]
set_property PROBES.FILE {} [current_hw_device]
set_property FULL_PROBES.FILE {} [current_hw_device]

# program the device
program_hw_devices [current_hw_device]

# close hw manager and exit vivado
close_hw_manager
# exit
