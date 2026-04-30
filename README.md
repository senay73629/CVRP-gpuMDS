# CVRP-gpuMDS

GPU-accelerated implementation of the parMDS heuristic for the **Capacitated Vehicle Routing Problem (CVRP)** using CUDA.

> B.Tech Project — Senay Patel, Department of CSE, IIT Madras  
> Guide: Prof. Rupesh Nasre

## Overview

The CVRP asks for minimum-cost vehicle routes serving all customers exactly once while respecting vehicle capacity. This project accelerates the [parMDS](https://github.com/souzamarcelo/parMDS) algorithm on the GPU through:

1. **GPU-resident Borůvka MST** — Union-Find with atomic operations, path-splitting compression, and packed `atomicMin` for per-component best-edge selection.
2. **Spatial hash grid with spiral-search KNN** — Reduces per-point nearest-neighbor search from O(N) to O(k) on average (k = log₂N) with early termination.
3. **Parallel route search kernel** — 1,000 independent randomized DFS traversals run concurrently, using a memory-coalesced buffer layout for efficient global memory access.
4. **CPU 2-opt post-processing** — Local search improvement on the best solution found.

## Repository Structure

```
CVRP-gpuMDS/
├── gpuMDS.cu          # Final GPU implementation (v4.1)
├── All versions/      # Earlier versions (v1 – v4)
├── Makefile           # Build rules
├── runAll.sh          # Batch execution script
└── inputs/            # CVRP benchmark instances (.vrp)
```

## Requirements

- **CUDA Toolkit** ≥ 11.0 (`nvcc`, Thrust)
- **cuRAND** (`-lcurand`)
- **C++17** compatible host compiler
- **Python 3** with `matplotlib`, `numpy` (for plotting scripts)

## Build & Run

```bash
# Build (default compiles gpuMDS.cu)
make

# Run on a single instance
./gpuMDS.out inputs/X-n101-k25.vrp -round 1 -v

# Run all instances
bash runAll.sh
```

**Options:**
- `-round 0|1` — disable/enable Euclidean distance rounding (default: 1)
- `-v` — verbose output (MST phase details, KNN timing)

## Output Format

Results are appended to a text file in tab-separated format:

```
<input_file>  MinCost: <cost>  TimeMST: <s>  TimeLoop: <s>  TimePostProcess: <s>  TimeTotal: <s>  VALID
```

## Results

Tested on 130 CVRP benchmark instances (N = 100 to 1,000) and custom instances up to N = 10,000.

- **Solution quality:** 15–25% error over best-known solutions (equivalent to the CPU baseline)
- **Speedup:** Significant runtime improvement over sequential CPU, particularly for larger instances
- All GPU versions produce equivalent solution quality — optimizations only affect runtime

## License

This project was developed as part of a B.Tech thesis at IIT Madras.