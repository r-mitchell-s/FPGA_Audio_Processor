
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
create_project: 2

00:00:052

00:00:062	
513.7662	
197.789Z17-268h px� 
]
Command: %s
53*	vivadotcl2,
*link_design -top top -part xc7z020clg484-1Z4-113h px� 
U
#Design is defaulting to srcset: %s
437*	planAhead2
	sources_1Z12-437h px� 
X
&Design is defaulting to constrset: %s
434*	planAhead2
	constrs_1Z12-434h px� 
D
Loading part %s157*device2
xc7z020clg484-1Z21-403h px� 
Z
$Part: %s does not have CEAM library.966*device2
xc7z020clg484-1Z21-9227h px� 
�
-Reading design checkpoint '%s' for cell '%s'
275*project2i
gc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0.dcp2	
clk_genZ1-454h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002
00:00:00.0072	
974.6412
0.000Z17-268h px� 
T
-Analyzing %s Unisim elements for replacement
17*netlist2
17Z29-17h px� 
X
2Unisim Transformation completed in %s CPU seconds
28*netlist2
1Z29-28h px� 
V
Netlist was created with %s %s291*project2
Vivado2

2024.1.1Z1-479h px� 
K
)Preparing netlist for logic optimization
349*projectZ1-570h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2q
mc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0_board.xdc2
clk_gen/inst	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2q
mc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0_board.xdc2
clk_gen/inst	8Z20-847h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2k
gc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0.xdc2
clk_gen/inst	8Z20-848h px� 
�
%Done setting XDC timing constraints.
35*timing2k
gc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0.xdc2
548@Z38-35h px� 
�
Deriving generated clocks
2*timing2k
gc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0.xdc2
548@Z38-2h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
get_clocks: 2

00:00:052

00:00:052

1673.6292	
578.492Z17-268h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2k
gc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0.xdc2
clk_gen/inst	8Z20-847h px� 
�
Parsing XDC File [%s]
179*designutils2e
aC:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.srcs/constrs_1/new/constraints.xdc8Z20-179h px� 
�
�Clock '%s' completely overrides clock '%s', which is referenced by one or more other constraints. Any constraints that refer to the overridden clock will be ignored.
New: %s
Previous: %s
737*constraints2
sys_clk_pin2

clk_100mhz2�
dummy1"^
\create_clock -period 10.000 -name sys_clk_pin -waveform {0.000 5.000} [get_ports clk_100mhz]"l
aC:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.srcs/constrs_1/new/constraints.xdc2 [::"

32 :]2�
dummy2"4
2create_clock -period 10.000 [get_ports clk_100mhz]"r
gc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0.xdc2 [::"
532 :]8Z18-1055h px� 
�
Finished Parsing XDC File [%s]
178*designutils2e
aC:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.srcs/constrs_1/new/constraints.xdc8Z20-178h px� 
�
$Parsing XDC File [%s] for cell '%s'
848*designutils2p
lc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0_late.xdc2
clk_gen/inst	8Z20-848h px� 
�
-Finished Parsing XDC File [%s] for cell '%s'
847*designutils2p
lc:/Users/ros3142/Desktop/ES100/FPGA_Audio_Processor/PMOD_test2.gen/sources_1/ip/clk_wiz_0/clk_wiz_0_late.xdc2
clk_gen/inst	8Z20-847h px� 
Q
)Pushed %s inverter(s) to %s load pin(s).
98*opt2
02
0Z31-138h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
Netlist sorting complete. 2

00:00:002

00:00:002

1674.8442
0.000Z17-268h px� 
l
!Unisim Transformation Summary:
%s111*project2'
%No Unisim elements were transformed.
Z1-111h px� 
~
G%s Infos, %s Warnings, %s Critical Warnings and %s Errors encountered.
28*	vivadotcl2
112
02
12
0Z4-41h px� 
K
%s completed successfully
29*	vivadotcl2
link_designZ4-42h px� 
�
I%sTime (s): cpu = %s ; elapsed = %s . Memory (MB): peak = %s ; gain = %s
268*common2
link_design: 2

00:00:092

00:00:102

1674.8442

1145.371Z17-268h px� 


End Record