cc=gcc

flags= -O3 -Wall
dpuflags= `dpu-pkg-config --cflags --libs dpu`
tasklets = 16
# dpus = 2546
dpus = 64
stack_size = 256

all: main quicksort_dpu mergesort_dpu

quicksort_dpu: quicksort_dpu.c
	dpu-upmem-dpurte-clang -O2 \
		-DNR_TASKLETS=$(tasklets) \
		-DNR_DPUS=$(dpus) \
		-DSTACK_SIZE_DEFAULT=$(stack_size) \
		quicksort_dpu.c -o quicksort_dpu

mergesort_dpu: mergesort_dpu.c
	dpu-upmem-dpurte-clang -O2 \
		-DNR_TASKLETS=$(tasklets) \
		-DNR_DPUS=$(dpus) \
		-DSTACK_SIZE_DEFAULT=$(stack_size) \
		mergesort_dpu.c -o mergesort_dpu

main: sort_host.c
	$(cc) sort_host.c $(flags) $(dpuflags) \
		-DDPU_QUICK_BINARY="\"./quicksort_dpu\"" \
		-DDPU_MERGE_BINARY="\"./mergesort_dpu\"" \
		-o main

clean:
	rm -rf main quicksort_dpu mergesort_dpu *.o *.out