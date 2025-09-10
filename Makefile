cc=gcc

flags= -O3 -Wall
dpuflags= `dpu-pkg-config --cflags --libs dpu`
tasklets = 16
# dpus = 2546
dpus = 64
stack_size = 256

all: main dpucode

dpucode: quicksort_dpu.c
	dpu-upmem-dpurte-clang -O2 -DNR_TASKLETS=$(tasklets) -DNR_DPUS=${dpus} -DSTACK_SIZE_DEFAULT=$(stack_size) quicksort_dpu.c -o quicksort_dpu

main: quicksort_host.c dpucode
	$(cc) quicksort_host.c $(flags) $(dpuflags) -o main

clean:
	rm -rf main quicksort_dpu *.o *.out