#!/bin/bash

# cd to the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# check if we're in the correct directory
if [[ ! -f "${SCRIPT_DIR}/run_all.tcl" ]]; then
    echo "Error: run_all.tcl not found in ${SCRIPT_DIR}"
    exit 1
fi

# check if Vivado is in PATH
if ! command -v vivado &> /dev/null; then
    echo "Vivado not found in PATH. Please source Vivado settings before running this script."
    exit 1
fi

# run Vivado in batch mode with the main TCL script
cd "${SCRIPT_DIR}"
vivado -mode batch -source ./run_all.tcl

# check if the build was successful
if [ $? -eq 0 ]; then
    echo "Project build completed successfully!"
else
    echo "Project build failed. Check the Vivado logs for details."
    exit 1
fi