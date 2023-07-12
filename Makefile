# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = --compiler-options -Wall -g -G -arch=sm_86  -O3 -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -I. -Irbdfiles -I./qdldl/include -lcublas -Lqdldl/build/out -lqdldl 

# Name of the output executable
EXECUTABLE = runme.exe

# Source file
SOURCE = runme.cu

all:
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
