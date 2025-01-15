# FPGA_Audio_Processor
Senior thesis project for Harvard University's ES100. Focused on the implementation of DSP algorithms on the Zedboard Zynq7000, which takes a stereo audio input via 3.5mm jack sampling and processing the signal before routing it through a DAC to an output audio device.

# Background
Senior design project completed for partial fulfillment of the Bachelor of Sciences Degree in the department of Electrical Engineering at Harvard University. The project consists of the deisgn and synthesis of an audio processor on an FPGA for live, low-latency musical effects applications.

# High Level Description
The Zedboard Zynq7000 interfaces with a PMOD I2S2 audio codec for analog-to-digital conversion from a 3.5mm sound source. The digitized sound signal (interprereted via I2S protocol) is sent through a signal path consisting of several custom effects modules before being routed out to the DAC on the PMOD i2S2 peripheral board and then to an output device.

# Directory Structure
At the top level there are seven directories that make up this project. RTL, IP, and Constraints contain the modules and constraints that constitute the build itself. Scripts contains any Bash or Tcl script used to automate the build process, simulations, or other yet to be determined functions of the project. Documentation contains schematics, the project report, and other informative files, and Simaulation contains testbench files, along with waveforms and other simulation materials.

# Usage and Build
To run this build on a Zedboard Zynq7000 interfacing with a PMOD I2S2 (in the JA PMOD jumper), clone the repository and run build.sh from inside the "Scripts" directory. This will start Vivado; build the sources and constraints; generate the necessary IPs, and run simulation and synthesis. From here the bitstrean should be easy to generate and upload to the FPGA. If trying to run this project on a different FPGA, the approporiate changes to constraints w need to be made, in addition to changing the part number in create_project.tcl.