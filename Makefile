NVCC     = nvcc
NVCCFLAGS = -O3 -std=c++17
LIBS      = -lcurand

gpuMDS: gpuMDS.cu
	$(NVCC) $(NVCCFLAGS) -o gpuMDS.out gpuMDS.cu $(LIBS)

all: v1 v2 v3 v3.1 v4 v4.1

v1: ./All\ versions/gpuMDS-v1.cu
	$(NVCC) $(NVCCFLAGS) -o v1.out ./All\ versions/gpuMDS-v1.cu $(LIBS)

v2: ./All\ versions/gpuMDS-v2.cu
	$(NVCC) $(NVCCFLAGS) -o v2.out ./All\ versions/gpuMDS-v2.cu $(LIBS)

v3: ./All\ versions/gpuMDS-v3.cu
	$(NVCC) $(NVCCFLAGS) -o v3.out ./All\ versions/gpuMDS-v3.cu $(LIBS)

v3.1: ./All\ versions/gpuMDS-v3.1.cu
	$(NVCC) $(NVCCFLAGS) -o v3.1.out ./All\ versions/gpuMDS-v3.1.cu $(LIBS)

v4: ./All\ versions/gpuMDS-v4.cu
	$(NVCC) $(NVCCFLAGS) -o v4.out ./All\ versions/gpuMDS-v4.cu $(LIBS)

v4.1: ./All\ versions/gpuMDS-v4.1.cu
	$(NVCC) $(NVCCFLAGS) -o v4.1.out ./All\ versions/gpuMDS-v4.1.cu $(LIBS)

clean:
	rm -f *.out

cleanFolders:
	rm -rf outinputs*
