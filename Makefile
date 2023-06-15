# Makefile

# Compiler and compiler flags
NVCC = nvcc
CFLAGS = -g -G -I. -IGPU-PCG/include -IGLASS -IGPU-PCG -Irbdfiles -lcublas

# Name of the output executable
EXECUTABLE = runme

# Source file
SOURCE = runme.cu

all:
	$(NVCC) $(CFLAGS) $(SOURCE) -o $(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE)
