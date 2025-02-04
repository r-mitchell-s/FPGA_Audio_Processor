#!/bin/bash

# make sure vivado is in PATH
if ! command -v vivado &> /dev/null; then
    echo "ERROR: Vivado is not in PATH"
    exit 1 
fi

# create project name with current datetime
datetime=$(date +"%Y_%m_%d_%H_%M_%S")
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
project_dir="${script_dir}/../Project_${datetime}"

# create the project directory
mkdir -p "${project_dir}"

# change to the project directory before running Vivado
cd "${project_dir}"

# run the tcl script
vivado -mode batch -source "${script_dir}/create_project.tcl"

# verify status to user
if [ $? -eq 0 ]; then
    echo "SUCCESS: FPGA programmed successfully"
else    
    echo "ERROR: FPGA was not able to be programmed"
    exit 1
fi