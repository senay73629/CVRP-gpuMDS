# CVRP-gpuMDS

GPU-accelerated implementation of the parMDS heuristic for the **Capacitated Vehicle Routing Problem (CVRP)** using CUDA.

> B.Tech Project — Senay Patel, Department of CSE, IIT Madras  
> Guide: Prof. Rupesh Nasre

## Overview

The CVRP asks for minimum-cost vehicle routes serving all customers exactly once while respecting vehicle capacity. This project accelerates the [parMDS](https://github.com/mrprajesh/parMDS/tree/main) algorithm on the GPU through:

1. **GPU-resident Borůvka MST** — Union-Find with atomic operations, path-splitting compression, and packed `atomicMin` for per-component best-edge selection.
2. **Spatial hash grid with spiral-search KNN** — Reduces per-point nearest-neighbor search from O(N) to O(k) on average (k = log₂N) with early termination.
3. **Parallel route search kernel** — 1,000 independent randomized DFS traversals run concurrently, using a memory-coalesced buffer layout for efficient global memory access.
4. **CPU 2-opt post-processing** — Local search improvement on the best solution found.

## Repository Structure

```
CVRP-gpuMDS/
├── gpuMDS.cu              # Final GPU implementation (v4.1)
├── Makefile               # Build rules (nvcc, C++17, -lcurand)
├── runAll.sh              # Batch compile-and-run script
├── compare.py             # Results comparison & plotting tool
├── bks_costs.txt          # Best-known solution costs for 130 benchmark instances
├── inputs/                # 130 standard CVRP benchmark instances (.vrp)
├── generated_inputs/      # 100 synthetically generated CVRP instances (.vrp)
└── All versions/          # Earlier implementation versions
    ├── gpuMDS-v1.cu
    ├── gpuMDS-v2.cu
    ├── gpuMDS-v3.cu
    ├── gpuMDS-v3.1.cu
    ├── gpuMDS-v4.cu
    └── gpuMDS-v4.1.cu
```

## Requirements

- **CUDA Toolkit** ≥ 11.0 (`nvcc`, Thrust)
- **cuRAND** (`-lcurand`)
- **C++17** compatible host compiler
- **Python 3** with `matplotlib`, `numpy` (for `compare.py`)

## Build & Run

```bash
# Build (default compiles gpuMDS.cu)
make

# Build a specific version
make v3.1

# Run on a single instance
./gpuMDS.out inputs/X-n101-k25.vrp -v

# Run all instances (see runAll.sh section below)
bash runAll.sh
```

**Options:**
- `-v` — verbose output (MST phase details, KNN timing)

---

## `runAll.sh` — Batch Execution

Compiles and runs a chosen variant on every `.vrp` file in the selected input directory (sorted by size, smallest first).

```bash
./runAll.sh [INPUT_LOC] [VARIANT] [OUTFILE]
```

| Argument     | Default       | Description |
|--------------|---------------|-------------|
| `INPUT_LOC`  | `generated`   | Set to `normal` to use `inputs/`; any other value (or omitted) uses `generated_inputs/`. |
| `VARIANT`    | *(empty)*     | Version to compile and run: `v1`, `v2`, `v3`, `v3.1`, `v4`, `v4.1`. If omitted, defaults to the main `gpuMDS` target. |
| `OUTFILE`    | *(none)*      | If specified, stdout is appended to this file instead of printed to the terminal. |

**Examples:**

```bash
# Run latest gpuMDS on generated inputs, print to terminal
./runAll.sh

# Run gpuMDS on the standard benchmark inputs
./runAll.sh normal

# Run v3 on standard inputs, save results to results.txt
./runAll.sh normal v3 results.txt

# Run v4.1 on generated inputs, save to out.txt
./runAll.sh generated v4.1 out.txt
```

---

## `compare.py` — Results Comparison

Compares two result files (e.g., from different variants or machines) across cost and timing metrics. Produces text reports and dark-themed comparison graphs.

```bash
python compare.py <file1> <file2> [-n N [N ...]] [-l LOCATION]
```

| Argument | Description |
|----------|-------------|
| `file1`  | Path to the first results file. |
| `file2`  | Path to the second results file. |
| `-n`     | Feature(s) to run (default: all). **1** = % error report (.txt), **2** = % error graph, **3** = total time graph, **4** = loop time graph, **5** = MST time graph. |
| `-l`     | Location label for the output directory name (default: `local`). |

**Examples:**

```bash
# Run all 5 features
python compare.py gpu_results.txt parmds_results.txt

# Only generate the error report
python compare.py gpu_results.txt parmds_results.txt -n 1

# Error graph + loop time graph, labelled as "remote"
python compare.py gpu_results.txt parmds_results.txt -n 2 4 -l remote
```

All outputs (report, graphs, and input files) are saved to a directory named `<label1>_vs_<label2>_on_<location>/`.

> **Note — Percentage error & `bks_costs.txt`:**
> The percentage error shown in the report and error graph is computed as `(cost − BKS) / BKS × 100`, where BKS values come from `bks_costs.txt`. This file contains the best-known solution costs for the **130 standard benchmark instances** in the `inputs/` directory (sourced from [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/en/instances)). Since no BKS values exist for the synthetically generated instances in `generated_inputs/`, the percentage error column will show **N/A** for those instances. To get meaningful error analysis, run with results produced from the `inputs/` directory (i.e., `./runAll.sh normal ...`).

## Output Format

Results are appended to a text file in tab-separated format:

```
<input_file>  MinCost: <cost>  TimeMST: <s>  TimeLoop: <s>  TimePostProcess: <s>  TimeTotal: <s>  VALID/INVALID
```

## Results

Tested on 130 CVRP benchmark instances (N = 100 to 1,000) and custom instances up to N = 30,000.

- **Solution quality:** 10–15% error over best-known solutions (equivalent to the CPU baseline)
- **Speedup:** Significant runtime improvement over sequential CPU, particularly for larger instances
- All GPU versions produce equivalent solution quality — optimizations only affect runtime

## License

This project was developed as part of a B.Tech thesis at IIT Madras.