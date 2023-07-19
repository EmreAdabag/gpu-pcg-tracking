# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall -arch=sm_89  -O3 -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -I. -Irbdfiles -I./qdldl/include -lcublas -Lqdldl/build/out -lqdldl 

# Name of the output executable
EXECUTABLE = runme.exe

# Source file
SOURCE = runme.cu

test:
	$(NVCC) $(CFLAGS) test.cu -o $(EXECUTABLE)

all:
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE)

lcm:
	$(NVCC) $(CFLAGS) -o runMPC_LCM.exe MPCGPU_LCM_wrapper/runMPC_LCM.cu -llcm -lpthread -gencode arch=compute_89,code=sm_89 -O3

printer:
	$(NVCC) $(CFLAGS) -o runprinter.exe MPCGPU_LCM_wrapper/print.cu -llcm -lpthread -gencode arch=compute_89,code=sm_89 -O3


clean:
	rm -f $(EXECUTABLE)
