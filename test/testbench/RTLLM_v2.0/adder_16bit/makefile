.PHONY: vcs sim clean

TEST_DESIGN = adder_16bit

vcs:
		vcs -sverilog +v2k -timescale=1ns/1ns       \
		-debug_all         							\
		-l compile.log 		-full64						\
		${TEST_DESIGN}.v testbench.v

sim:
		./simv -l run.log

clean:
		rm -rf *.log  csrc  simv*  *.key *.vpd  DVEfiles coverage *.vdb output.txt
