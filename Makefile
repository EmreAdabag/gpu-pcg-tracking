# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = -O3 -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -I. -Irbdfiles -I./qdldl/include -lcublas -lqdldl 

# Name of the output executable
EXECUTABLE = runme.exe

# Source file
SOURCE = runme.cu

all:
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
