
all: Net.cu Net.cuh 
	nvcc -g -O2 -o timing Net.cu -gencode arch=compute_75,code=sm_75 -std=c++11 -Xcompiler -fopenmp
