# set project name
set project_name "FPGA_Audio_Processor"
set project_dir [file normalize ".vivado"]

# create the project and set the part number for Zynq7000
current_project ${project_name} ${project_dir} 
#-part xc7z020clg484-1

# set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# add sources, sim files, and constraints to the new project
add_files -fileset sources_1 [glob ../RTL/*.sv]
add_files -fileset constr_1 [glob ../Constraints/*.xdc]
add_files -fileset sim_1 [glob ../Simulation/*.sv]

# create IPs
source create_ip.tcl