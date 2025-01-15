# FPGA_Audio_Processor
Senior thesis project for Harvard University's ES100. Focused on the implementation of DSP algorithms on the Zedboard Zynq7000, which takes a stereo audio input via 3.5mm jack sampling and processing the signal before routing it through a DAC to an output audio device.

# Background
Senior design project completed for partial fulfillment of the Bachelor of Sciences Degree in the department of Electrical Engineering at Harvard University. The project consists of the deisgn and synthesis of an audio processor on an FPGA for live, low-latency musical effects applications.

# High Level Description
The Zedboard Zynq7000 interfaces with a PMOD I2S2 audio codec for analog-to-digital conversion from a 3.5mm sound source. The digitized sound signal (interprereted via I2S protocol) is sent through a signal path consisting of several custom effects modules before being routed out to the DAC on the PMOD i2S2 peripheral board and then to an output device.

# Implementation Details

# Usage (Simulation and Build)

