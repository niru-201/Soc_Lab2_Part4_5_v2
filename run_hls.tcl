# vitis_hls -f run_hls.tcl
open_project -reset hls_dot
set_top dot_q8_0
add_files dot_q8_0.cpp
add_files -tb tb_dot_q8_0.cpp
add_files -tb vec_dot_q8_0.c

open_solution -reset solution1 -flow_target vivado
set_part {xczu3eg-sbva484-1-e}
create_clock -period 10 -name default

csim_design
csynth_design
cosim_design -rtl verilog -trace_level all
exit
