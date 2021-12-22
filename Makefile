INC="helperFiles/inc"
INCFLAGS=-I$(INC)
GLUTFLAGS=-lglut -lGL
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc

all: imageConvolution
imageConvolution: imageConvolution.cu
	$(NVCC) $(INCFLAGS) imageConvolution.cu -o imageConvolution
clean:
	rm imageConvolution
