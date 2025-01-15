# create the project
source create_project.tcl

# run synthesis
launch_runs synth_1
wait_on_run synth_1

# run implentation
launch_runs impl_1 -to_step write_bitstream
wait_on_run impl_1