# set project name
set project_name "FPGA_Audio_Processor"
set project_dir [file normalize ".vivado"]

# create the project and set the part number for Zynq7000
create_project ${project_name} ${project_dir} -part xc7z020clg484-1 -force

# set project properties
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

# add sources if they exist in the RTL directory
if {[file isdirectory ../RTL]} {
    set source_files [glob ../RTL/*.sv]
    if {[llength $source_files] > 0} {
        add_files -fileset sources_1 $source_files 
    }
}

# add constraints if they exist in the RTL directory
if {[file isdirectory ../Constraints]} {
    set constr_files [glob ../Constraints/*.xdc]
    if {[llength $constr_files] > 0} {
        add_files -fileset constrs_1 $constr_files 
    }
}


# add sims if they exist in the RTL directory
if {[file isdirectory ../Simulation]} {
    set sim_files [glob ../Simulation/*.sv]
    if {[llength $sim_files] > 0} {
        add_files -fileset sim_1 $sim_files 
    }
}

# create IPs
source create_ip.tcl