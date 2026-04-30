// gpuMDS-v4.1.cu
// Modified from gpuMDS-v4.cu:
// - Spatial hash grid built from (x,y) coordinates before Borůvka loop
// - First Borůvka iteration uses KNN candidates from hash grid (k = log2(n))
// - Cell size auto-computed from average point spacing
// - v4.1: Spiral search KNN with early exit replaces fixed-radius square scan
// - Hash grid freed post-loop only if GPU memory headroom is insufficient
//
// Part 1 = Borůvka's MST (CUDA GPU, on-the-fly Euclidean distances)
//          v3.1: merge + CSR construction fully on GPU
//          v4:   spatial hash grid KNN acceleration for first Borůvka iterations
//          v4.1: spiral search with early termination
// Part 2 = CUDA parallel 1k route search (one thread per iteration) with memory coalescing
// Part 3 = seqMDS 2-opt post-processing (CPU)
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <climits>
#include <chrono>
#include <iomanip>
#include <random>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
// MODIFIED: v4 — additional Thrust headers for hash grid
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/count.h>

// MODIFIED: v4.1 — Compile-time max k for register-resident arrays in spiral search
#define MAX_K 64

// -----------------------------------------------------------------------
// Error checking macro
// -----------------------------------------------------------------------
#define CUDA_CHECK(X, call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                << " — " << cudaGetErrorString(err) << "\nLine Number: " << X << "\n";               \
      exit(1);                                                              \
    }                                                                       \
  } while (0)

// -----------------------------------------------------------------------
// Types (same as seqMDS)
// -----------------------------------------------------------------------
using point_t  = double;
using weight_t = double;
using demand_t = double;
using node_t   = int;

// Global verbose flag — controlled by -v argument
bool g_verbose = false;

// -----------------------------------------------------------------------
// Edge class (same as seqMDS)
// -----------------------------------------------------------------------
class Edge {
 public:
  node_t   to;
  weight_t length;
  Edge() {}
  Edge(node_t t, weight_t l) : to(t), length(l) {}
  bool operator<(const Edge& e) const { return length < e.length; }
};

// -----------------------------------------------------------------------
// Point
// -----------------------------------------------------------------------
class Point {
 public:
  point_t  x, y;
  demand_t demand;
};

// -----------------------------------------------------------------------
// VRP class — identical to seqMDS (with precomputed dist table)
// -----------------------------------------------------------------------
class VRP {
  size_t   size;
  demand_t capacity;
  std::string type;

 public:
  VRP()  {}
  ~VRP() {}

  unsigned read(const std::string& filename);

  weight_t get_dist(node_t i, node_t j) const {
    weight_t w = std::sqrt(
          (node[i].x - node[j].x) * (node[i].x - node[j].x) +
          (node[i].y - node[j].y) * (node[i].y - node[j].y));
    return toRound ? std::round(w) : w;
  }

  std::vector<std::vector<Edge>> cal_graph_dist();

  std::vector<Point>   node;
  std::vector<weight_t> dist;
  bool toRound = true;

  size_t   getSize()     const { return size; }
  demand_t getCapacity() const { return capacity; }
};

// -----------------------------------------------------------------------
// VRP::read  (identical to seqMDS)
// -----------------------------------------------------------------------
unsigned VRP::read(const std::string& filename) {
  std::ifstream in(filename);
  if (!in.is_open()) {
    std::cerr << "Cannot open \"" << filename << "\"\n";
    exit(1);
  }
  std::string line;
  for (int i = 0; i < 3; ++i) std::getline(in, line);

  std::getline(in, line);
  size = (size_t)std::stof(line.substr(line.find(':') + 2));

  std::getline(in, line);  // distance type
  std::getline(in, line);
  capacity = std::stof(line.substr(line.find(':') + 2));

  std::getline(in, line);  // NODE_COORD_SECTION
  node.resize(size);
  for (size_t i = 0; i < size; ++i) {
    std::getline(in, line);
    std::stringstream ss(line);
    size_t id;
    ss >> id >> node[i].x >> node[i].y;
  }

  std::getline(in, line);  // DEMAND_SECTION
  for (size_t i = 0; i < size; ++i) {
    std::getline(in, line);
    std::stringstream ss(line);
    size_t id;
    ss >> id >> node[i].demand;
  }
  in.close();
  return (unsigned)capacity;
}

// -----------------------------------------------------------------------
// VRP::cal_graph_dist  (identical to seqMDS — precomputes dist[] table)
// Not using it anywhere in gpuMDS
// -----------------------------------------------------------------------
std::vector<std::vector<Edge>> VRP::cal_graph_dist() {
  dist.resize((size * (size - 1)) / 2);
  std::vector<std::vector<Edge>> nG(size);
  size_t k = 0;
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = i + 1; j < size; ++j) {
      weight_t w = std::sqrt(
          (node[i].x - node[j].x) * (node[i].x - node[j].x) +
          (node[i].y - node[j].y) * (node[i].y - node[j].y));
      dist[k] = toRound ? std::round(w) : w;
      k++;
    }
  }
  return nG;
}

// =========================================================================
//  PART 1: Borůvka's MST — CUDA GPU (on-the-fly Euclidean distances)
// =========================================================================

// -----------------------------------------------------------------------
// GPU-side Euclidean distance (on-the-fly, no matrix stored)
// -----------------------------------------------------------------------
__device__ inline double gpu_eucl(double x1, double y1, double x2, double y2) {
  double dx = x1 - x2, dy = y1 - y2;
  return sqrt(dx * dx + dy * dy);
}

// -----------------------------------------------------------------------
// Borůvka Phase Kernel (original full O(n²) scan)
// -----------------------------------------------------------------------
__global__ void boruvkaFindCheapest(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const int*    __restrict__ d_comp,
    int*   d_cheapest_to,
    double* d_cheapest_w,
    int N)
{
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= N) return;

  double best_w = DBL_MAX;
  int    best_v = -1;
  int    cu     = d_comp[u];
  double ux     = d_x[u], uy = d_y[u];

  for (int v = 0; v < N; ++v) {
    if (d_comp[v] == cu) continue;
    double w = gpu_eucl(ux, uy, d_x[v], d_y[v]);
    if (w < best_w) {
      best_w = w;
      best_v = v;
    }
  }

  d_cheapest_to[u] = best_v;
  d_cheapest_w[u]  = best_w;
}

// MODIFIED: v3.1 — GPU device Union-Find with path splitting + atomicCAS merge
__device__ int gpu_find(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

// MODIFIED: v3.1 — deterministic index ordering prevents cycle bug
__device__ bool gpu_unite(int* parent, int* rank_uf, int a, int b) {
    a = gpu_find(parent, a);
    b = gpu_find(parent, b);
    if (a == b) return false;
    if (a > b) { int tmp = a; a = b; b = tmp; }
    if (atomicCAS(&parent[b], b, a) == b) {
        if (rank_uf[a] == rank_uf[b]) atomicAdd(&rank_uf[a], 1);
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------
// MODIFIED: v3.1 — GPU kernels for Borůvka merge and CSR construction
// -----------------------------------------------------------------------
__global__ void initUFKernel(int* parent, int* rank_uf, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    parent[i] = i;
    rank_uf[i] = 0;
}

// MODIFIED: v3.1 — Per-component best-edge reduction using packed atomicMin.
__global__ void findCompBestKernel(
    const int* __restrict__ d_comp,
    const int* __restrict__ d_cheapest_to,
    const double* __restrict__ d_cheapest_w,
    unsigned long long* d_comp_best,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int v = d_cheapest_to[i];
    if (v < 0) return;

    int c = d_comp[i];
    double w = d_cheapest_w[i];
    unsigned int w_bits = __float_as_uint((float)w);
    unsigned long long packed = ((unsigned long long)w_bits << 32) | ((unsigned long long)(unsigned int)i);
    atomicMin(&d_comp_best[c], packed);
}

// MODIFIED: v3.1 — merge kernel only merges the PER-COMPONENT BEST edge.
__global__ void mergeComponentsKernel(
    int* parent, int* rank_uf,
    const int* __restrict__ d_comp,
    const int* __restrict__ d_cheapest_to,
    const double* __restrict__ d_cheapest_w,
    const unsigned long long* __restrict__ d_comp_best,
    int* d_mst_u, int* d_mst_v, double* d_mst_w,
    int* d_mst_count,
    int N,
    bool toRound)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int v = d_cheapest_to[i];
    if (v < 0) return;

    int c = d_comp[i];
    unsigned long long best = d_comp_best[c];
    int winner = (int)(best & 0xFFFFFFFF);
    if (winner != i) return;

    if (gpu_unite(parent, rank_uf, i, v)) {
        int idx = atomicAdd(d_mst_count, 1);
        d_mst_u[idx] = i;
        d_mst_v[idx] = v;
        double w = d_cheapest_w[i];
        d_mst_w[idx] = toRound ? round(w) : w;
    }
}

__global__ void updateComponentsKernel(int* comp, int* parent, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    comp[i] = gpu_find(parent, i);
}

__global__ void countDegreesKernel(
    const int* __restrict__ d_mst_u,
    const int* __restrict__ d_mst_v,
    int* d_degree,
    int num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    int u = d_mst_u[idx];
    int v = d_mst_v[idx];
    atomicAdd(&d_degree[u], 1);
    atomicAdd(&d_degree[v], 1);
}

__global__ void scatterEdgesKernel(
    const int* __restrict__ d_mst_u,
    const int* __restrict__ d_mst_v,
    const double* __restrict__ d_mst_w,
    int* d_cursor,
    int* d_csr_col_idx,
    double* d_csr_weights,
    int num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    int u = d_mst_u[idx];
    int v = d_mst_v[idx];
    double w = d_mst_w[idx];

    int pos_u = atomicAdd(&d_cursor[u], 1);
    d_csr_col_idx[pos_u] = v;
    d_csr_weights[pos_u] = w;

    int pos_v = atomicAdd(&d_cursor[v], 1);
    d_csr_col_idx[pos_v] = u;
    d_csr_weights[pos_v] = w;
}

// -----------------------------------------------------------------------
// MODIFIED: v4 — Spatial hash grid kernels
// -----------------------------------------------------------------------

// MODIFIED: v4 — Assign each point to a grid cell
__global__ void assignCellsKernel(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    int* d_cell_ids,
    int* d_point_ids,
    float x_min, float y_min, float cell_size,
    int grid_w, int grid_h, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cell_x = (int)(((float)(d_x[i]) - x_min) / cell_size);
    int cell_y = (int)(((float)(d_y[i]) - y_min) / cell_size);
    // Clamp to valid range
    if (cell_x < 0) cell_x = 0;
    if (cell_y < 0) cell_y = 0;
    if (cell_x >= grid_w) cell_x = grid_w - 1;
    if (cell_y >= grid_h) cell_y = grid_h - 1;
    d_cell_ids[i] = cell_y * grid_w + cell_x;
    d_point_ids[i] = i;
}

// MODIFIED: v4 — Find start and end index of each cell in the sorted point array
__global__ void findCellBoundsKernel(
    const int* __restrict__ d_cell_ids,
    int* d_cell_start,
    int* d_cell_end,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int cur = d_cell_ids[i];
    if (i == 0 || d_cell_ids[i - 1] != cur) {
        d_cell_start[cur] = i;
    }
    if (i == n - 1 || d_cell_ids[i + 1] != cur) {
        d_cell_end[cur] = i;
    }
}

// -----------------------------------------------------------------------
// OLD findCheapestEdgeKNN — commented out for reference (v4 original)
// -----------------------------------------------------------------------
#if 0
__global__ void findCheapestEdgeKNN(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const int* __restrict__ d_component,
    const int* __restrict__ d_point_ids,
    const int* __restrict__ d_cell_start,
    const int* __restrict__ d_cell_end,
    int*   d_cheapest_to,
    double* d_cheapest_w,
    float x_min, float y_min, float cell_size,
    int grid_w, int grid_h,
    float search_radius,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double best_w = DBL_MAX;
    int    best_v = -1;
    int    ci     = d_component[i];
    double ix     = d_x[i], iy = d_y[i];
    int cx = (int)(((float)ix - x_min) / cell_size);
    int cy = (int)(((float)iy - y_min) / cell_size);
    if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
    if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;
    int r = (int)ceilf(search_radius / cell_size);
    int cx_lo = cx - r; if (cx_lo < 0) cx_lo = 0;
    int cx_hi = cx + r; if (cx_hi >= grid_w) cx_hi = grid_w - 1;
    int cy_lo = cy - r; if (cy_lo < 0) cy_lo = 0;
    int cy_hi = cy + r; if (cy_hi >= grid_h) cy_hi = grid_h - 1;
    for (int sy = cy_lo; sy <= cy_hi; ++sy) {
        for (int sx = cx_lo; sx <= cx_hi; ++sx) {
            int cell = sy * grid_w + sx;
            int start = d_cell_start[cell];
            if (start < 0) continue;
            int end = d_cell_end[cell];
            for (int p = start; p <= end; ++p) {
                int j = d_point_ids[p];
                if (j == i) continue;
                if (d_component[j] == ci) continue;
                double w = gpu_eucl(ix, iy, d_x[j], d_y[j]);
                if (w < best_w) { best_w = w; best_v = j; }
            }
        }
    }
    d_cheapest_to[i] = best_v;
    d_cheapest_w[i]  = (best_v < 0) ? (double)FLT_MAX : best_w;
}
#endif

// -----------------------------------------------------------------------
// MODIFIED: v4.1 — Spiral Search KNN Kernel
// -----------------------------------------------------------------------
// Search strategy: Visits hash grid cells ring-by-ring in order of increasing
//   Chebyshev distance from the query cell (spiral outward).
// Early exit: Stops expanding when the closest possible point in the next ring
//   (at Euclidean distance >= (r-1)*cell_size) is farther than the current
//   k-th nearest neighbor distance AND k neighbors have been found.
// Worst-case rings: r_max = max(grid_w, grid_h). For well-distributed 2D data,
//   average rings visited is ~1-3 for k = log2(n).
// Expected speedup: O(k) average vs O(n) worst-case of the old square scan.
//   Early exit eliminates redundant cell visits; ring-only iteration avoids
//   re-scanning interior cells.
// -----------------------------------------------------------------------
__launch_bounds__(256, 2)
__global__ void knnSpiralSearchKernel(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const int* __restrict__ d_point_ids,
    const int* __restrict__ d_cell_start,
    const int* __restrict__ d_cell_end,
    int*   d_knn_indices,   // output: [n * k] nearest neighbor indices
    double* d_knn_dists,    // output: [n * k] nearest neighbor distances
    float x_min, float y_min, float cell_size,
    int grid_w, int grid_h,
    int k, int r_max, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // --- Step 1: Initialize k-NN sorted array in registers ---
    float heap_dist[MAX_K];
    int   heap_idx[MAX_K];
    int   found = 0;
    float kth_dist = FLT_MAX;

    // Clamp k to MAX_K
    int kk = (k < MAX_K) ? k : MAX_K;

    #pragma unroll 4
    for (int t = 0; t < MAX_K; t++) {
        heap_dist[t] = FLT_MAX;
        heap_idx[t]  = -1;
    }

    double ix = d_x[i], iy = d_y[i];

    // Compute which cell point i belongs to
    int cx = (int)(((float)ix - x_min) / cell_size);
    int cy = (int)(((float)iy - y_min) / cell_size);
    if (cx < 0) cx = 0; if (cx >= grid_w) cx = grid_w - 1;
    if (cy < 0) cy = 0; if (cy >= grid_h) cy = grid_h - 1;

    // --- Step 2: Spiral ring expansion ---
    for (int r = 0; r <= r_max; r++) {
        // Early exit: closest possible point in ring r is at distance (r-1)*cell_size
        float min_possible_dist = (r <= 1) ? 0.0f : (float)(r - 1) * cell_size;
        if (found >= kk && min_possible_dist > kth_dist) break;

        if (r == 0) {
            // Ring 0: just the query cell itself
            int cell = cy * grid_w + cx;
            int start = d_cell_start[cell];
            if (start >= 0) {
                int end = d_cell_end[cell];
                for (int p = start; p <= end; ++p) {
                    int j = d_point_ids[p];
                    if (j == i) continue;
                    float w = (float)gpu_eucl(ix, iy, d_x[j], d_y[j]);
                    if (w < kth_dist || found < kk) {
                        // Insertion sort: find position
                        int pos = found < kk ? found : kk - 1;
                        for (int s = 0; s < found && s < kk; s++) {
                            if (w < heap_dist[s]) { pos = s; break; }
                        }
                        // Shift elements right
                        int limit = (found < kk) ? found : kk - 1;
                        for (int s = limit; s > pos; s--) {
                            heap_dist[s] = heap_dist[s - 1];
                            heap_idx[s]  = heap_idx[s - 1];
                        }
                        heap_dist[pos] = w;
                        heap_idx[pos]  = j;
                        if (found < kk) found++;
                        kth_dist = heap_dist[found - 1];
                    }
                }
            }
        } else {
            // Ring r > 0: visit only border cells at Chebyshev distance exactly r
            // Top row: (cx-r..cx+r, cy-r)
            {
                int ny = cy - r;
                if (ny >= 0 && ny < grid_h) {
                    int sx_lo = cx - r; if (sx_lo < 0) sx_lo = 0;
                    int sx_hi = cx + r; if (sx_hi >= grid_w) sx_hi = grid_w - 1;
                    for (int nx = sx_lo; nx <= sx_hi; ++nx) {
                        int cell = ny * grid_w + nx;
                        int start = d_cell_start[cell];
                        if (start < 0) continue;
                        int end = d_cell_end[cell];
                        for (int p = start; p <= end; ++p) {
                            int j = d_point_ids[p];
                            if (j == i) continue;
                            float w = (float)gpu_eucl(ix, iy, d_x[j], d_y[j]);
                            if (w < kth_dist || found < kk) {
                                int pos = found < kk ? found : kk - 1;
                                for (int s = 0; s < found && s < kk; s++) {
                                    if (w < heap_dist[s]) { pos = s; break; }
                                }
                                int limit = (found < kk) ? found : kk - 1;
                                for (int s = limit; s > pos; s--) {
                                    heap_dist[s] = heap_dist[s - 1];
                                    heap_idx[s]  = heap_idx[s - 1];
                                }
                                heap_dist[pos] = w;
                                heap_idx[pos]  = j;
                                if (found < kk) found++;
                                kth_dist = heap_dist[found - 1];
                            }
                        }
                    }
                }
            }
            // Bottom row: (cx-r..cx+r, cy+r)
            {
                int ny = cy + r;
                if (ny >= 0 && ny < grid_h) {
                    int sx_lo = cx - r; if (sx_lo < 0) sx_lo = 0;
                    int sx_hi = cx + r; if (sx_hi >= grid_w) sx_hi = grid_w - 1;
                    for (int nx = sx_lo; nx <= sx_hi; ++nx) {
                        int cell = ny * grid_w + nx;
                        int start = d_cell_start[cell];
                        if (start < 0) continue;
                        int end = d_cell_end[cell];
                        for (int p = start; p <= end; ++p) {
                            int j = d_point_ids[p];
                            if (j == i) continue;
                            float w = (float)gpu_eucl(ix, iy, d_x[j], d_y[j]);
                            if (w < kth_dist || found < kk) {
                                int pos = found < kk ? found : kk - 1;
                                for (int s = 0; s < found && s < kk; s++) {
                                    if (w < heap_dist[s]) { pos = s; break; }
                                }
                                int limit = (found < kk) ? found : kk - 1;
                                for (int s = limit; s > pos; s--) {
                                    heap_dist[s] = heap_dist[s - 1];
                                    heap_idx[s]  = heap_idx[s - 1];
                                }
                                heap_dist[pos] = w;
                                heap_idx[pos]  = j;
                                if (found < kk) found++;
                                kth_dist = heap_dist[found - 1];
                            }
                        }
                    }
                }
            }
            // Left column (excluding corners): (cx-r, cy-r+1..cy+r-1)
            {
                int nx = cx - r;
                if (nx >= 0 && nx < grid_w) {
                    int sy_lo = cy - r + 1; if (sy_lo < 0) sy_lo = 0;
                    int sy_hi = cy + r - 1; if (sy_hi >= grid_h) sy_hi = grid_h - 1;
                    for (int ny = sy_lo; ny <= sy_hi; ++ny) {
                        int cell = ny * grid_w + nx;
                        int start = d_cell_start[cell];
                        if (start < 0) continue;
                        int end = d_cell_end[cell];
                        for (int p = start; p <= end; ++p) {
                            int j = d_point_ids[p];
                            if (j == i) continue;
                            float w = (float)gpu_eucl(ix, iy, d_x[j], d_y[j]);
                            if (w < kth_dist || found < kk) {
                                int pos = found < kk ? found : kk - 1;
                                for (int s = 0; s < found && s < kk; s++) {
                                    if (w < heap_dist[s]) { pos = s; break; }
                                }
                                int limit = (found < kk) ? found : kk - 1;
                                for (int s = limit; s > pos; s--) {
                                    heap_dist[s] = heap_dist[s - 1];
                                    heap_idx[s]  = heap_idx[s - 1];
                                }
                                heap_dist[pos] = w;
                                heap_idx[pos]  = j;
                                if (found < kk) found++;
                                kth_dist = heap_dist[found - 1];
                            }
                        }
                    }
                }
            }
            // Right column (excluding corners): (cx+r, cy-r+1..cy+r-1)
            {
                int nx = cx + r;
                if (nx >= 0 && nx < grid_w) {
                    int sy_lo = cy - r + 1; if (sy_lo < 0) sy_lo = 0;
                    int sy_hi = cy + r - 1; if (sy_hi >= grid_h) sy_hi = grid_h - 1;
                    for (int ny = sy_lo; ny <= sy_hi; ++ny) {
                        int cell = ny * grid_w + nx;
                        int start = d_cell_start[cell];
                        if (start < 0) continue;
                        int end = d_cell_end[cell];
                        for (int p = start; p <= end; ++p) {
                            int j = d_point_ids[p];
                            if (j == i) continue;
                            float w = (float)gpu_eucl(ix, iy, d_x[j], d_y[j]);
                            if (w < kth_dist || found < kk) {
                                int pos = found < kk ? found : kk - 1;
                                for (int s = 0; s < found && s < kk; s++) {
                                    if (w < heap_dist[s]) { pos = s; break; }
                                }
                                int limit = (found < kk) ? found : kk - 1;
                                for (int s = limit; s > pos; s--) {
                                    heap_dist[s] = heap_dist[s - 1];
                                    heap_idx[s]  = heap_idx[s - 1];
                                }
                                heap_dist[pos] = w;
                                heap_idx[pos]  = j;
                                if (found < kk) found++;
                                kth_dist = heap_dist[found - 1];
                            }
                        }
                    }
                }
            }
        } // end else r > 0

        // Warp-level early exit: if ALL threads in warp are done, break together
        unsigned int warp_mask = __activemask();
        int my_done = (found >= kk && min_possible_dist > kth_dist) ? 1 : 0;
        unsigned int done_ballot = __ballot_sync(warp_mask, my_done);
        if (done_ballot == warp_mask) break;

    } // end ring loop

    // --- Step 3: Write results ---
    for (int t = 0; t < kk; t++) {
        d_knn_indices[i * kk + t] = (t < found) ? heap_idx[t] : -1;
        d_knn_dists[i * kk + t]   = (t < found) ? (double)heap_dist[t] : (double)FLT_MAX;
    }
}

// -----------------------------------------------------------------------
// MODIFIED: v4.1 — Extract cheapest cross-component neighbor from k-NN list
// -----------------------------------------------------------------------
__global__ void extractCheapestFromKNN(
    const int* __restrict__ d_knn_indices,
    const double* __restrict__ d_knn_dists,
    const int* __restrict__ d_component,
    int*   d_cheapest_to,
    double* d_cheapest_w,
    int k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double best_w = DBL_MAX;
    int    best_v = -1;
    int    ci = d_component[i];

    for (int t = 0; t < k; t++) {
        int j = d_knn_indices[i * k + t];
        if (j < 0) continue;
        if (d_component[j] == ci) continue;
        double w = d_knn_dists[i * k + t];
        if (w < best_w) {
            best_w = w;
            best_v = j;
        }
    }

    d_cheapest_to[i] = best_v;
    d_cheapest_w[i]  = (best_v < 0) ? (double)FLT_MAX : best_w;
}

// -----------------------------------------------------------------------
// MODIFIED: v4.1 — Fallback kernel: full O(n) linear scan for points that
// failed to find enough neighbors via spiral search (degenerate distributions)
// -----------------------------------------------------------------------
__global__ void knnFallbackKernel(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const int* __restrict__ d_component,
    int*   d_cheapest_to,
    double* d_cheapest_w,
    const int* __restrict__ d_knn_indices,  // to check which points need fallback
    int k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Only run for points that found no valid neighbor (knn_indices[i*k] == -1)
    if (d_knn_indices[i * k] != -1) return;

    double best_w = DBL_MAX;
    int    best_v = -1;
    int    ci = d_component[i];
    double ix = d_x[i], iy = d_y[i];

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        if (d_component[j] == ci) continue;
        double w = gpu_eucl(ix, iy, d_x[j], d_y[j]);
        if (w < best_w) {
            best_w = w;
            best_v = j;
        }
    }

    d_cheapest_to[i] = best_v;
    d_cheapest_w[i]  = (best_v < 0) ? (double)FLT_MAX : best_w;
}

// MODIFIED: v4 — Thrust functor to extract weight from cheapest edge results
struct ExtractWeight {
    const double* weights;
    __host__ __device__ ExtractWeight(const double* w) : weights(w) {}
    __host__ __device__ double operator()(int idx) const { return weights[idx]; }
};

// MODIFIED: v4 — Check if any point failed to find a cross-component neighbor
bool checkMissedEdges(double* d_cheapest_w, int n) {
    thrust::device_ptr<double> w_ptr(d_cheapest_w);
    double max_w = thrust::reduce(w_ptr, w_ptr + n, (double)(-FLT_MAX), thrust::maximum<double>());
    return (max_w >= (double)FLT_MAX);
}

// -----------------------------------------------------------------------
// MODIFIED: v3.1 + v4 — BoruvkaMST with GPU merge + GPU CSR construction
//                        + spatial hash grid KNN acceleration
// -----------------------------------------------------------------------
void BoruvkaMST(const VRP& vrp, int N,
                int*& d_csr_row, int*& d_csr_col, int& mst_nnz) {
  // ---- INITIALIZATION ----
  // Host coordinate arrays
  std::vector<double> h_x(N), h_y(N);
  for (int i = 0; i < N; ++i) {
    h_x[i] = vrp.node[i].x;
    h_y[i] = vrp.node[i].y;
  }

  // Allocate GPU buffers for coordinates and component labels
  double *d_x, *d_y;
  int    *d_comp;
  int    *d_cheapest_to;
  double *d_cheapest_w;

  CUDA_CHECK(100, cudaMalloc(&d_x,            N * sizeof(double)));
  CUDA_CHECK(101, cudaMalloc(&d_y,            N * sizeof(double)));
  CUDA_CHECK(102, cudaMalloc(&d_comp,         N * sizeof(int)));
  CUDA_CHECK(103, cudaMalloc(&d_cheapest_to,  N * sizeof(int)));
  CUDA_CHECK(104, cudaMalloc(&d_cheapest_w,   N * sizeof(double)));

  CUDA_CHECK(105, cudaMemcpy(d_x, h_x.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(106, cudaMemcpy(d_y, h_y.data(), N*sizeof(double), cudaMemcpyHostToDevice));

  // MODIFIED: v3.1 — GPU Union-Find arrays
  int *d_parent, *d_rank_uf;
  CUDA_CHECK(107, cudaMalloc(&d_parent,  N * sizeof(int)));
  CUDA_CHECK(108, cudaMalloc(&d_rank_uf, N * sizeof(int)));

  // MST edge accumulator arrays (max N-1 edges across all phases)
  int *d_mst_u, *d_mst_v;
  double *d_mst_w;
  int *d_mst_count;  // atomic counter on device
  CUDA_CHECK(109, cudaMalloc(&d_mst_u,     (N - 1) * sizeof(int)));
  CUDA_CHECK(110, cudaMalloc(&d_mst_v,     (N - 1) * sizeof(int)));
  CUDA_CHECK(111, cudaMalloc(&d_mst_w,     (N - 1) * sizeof(double)));
  CUDA_CHECK(112, cudaMalloc(&d_mst_count, sizeof(int)));
  CUDA_CHECK(113, cudaMemset(d_mst_count, 0, sizeof(int)));

  // Per-component best-edge array (packed weight+index)
  unsigned long long *d_comp_best;
  CUDA_CHECK(114, cudaMalloc(&d_comp_best, N * sizeof(unsigned long long)));

  const int BLK_BORUVKA = 128;
  const int GRD_BORUVKA = (N + BLK_BORUVKA - 1) / BLK_BORUVKA;

  // Initialize UF: parent[i]=i, rank_uf[i]=0
  initUFKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(d_parent, d_rank_uf, N);
  CUDA_CHECK(115, cudaGetLastError());
  CUDA_CHECK(116, cudaDeviceSynchronize());

  // Initialize d_comp[i] = i
  CUDA_CHECK(117, cudaMemcpy(d_comp, d_parent, N * sizeof(int), cudaMemcpyDeviceToDevice));

  // ======================================================================
  // MODIFIED: v4 — Step 1: Compute k and cell size on host before Borůvka loop
  // ======================================================================
  // int k = max(8, (int)sqrtf((float)N));
  int k = max(8, (int)log2f((float)N));
  // k is the number of nearest neighbors per point used in iteration 1
  // minimum clamped to 8 to avoid degenerate cases for very small n

  // MODIFIED: v4 — Compute bounding box using Thrust reductions on device
  float x_min, x_max, y_min, y_max;
  {
    thrust::device_ptr<double> dx_ptr(d_x);
    thrust::device_ptr<double> dy_ptr(d_y);
    x_min = (float)*thrust::min_element(dx_ptr, dx_ptr + N);
    x_max = (float)*thrust::max_element(dx_ptr, dx_ptr + N);
    y_min = (float)*thrust::min_element(dy_ptr, dy_ptr + N);
    y_max = (float)*thrust::max_element(dy_ptr, dy_ptr + N);
  }

  // MODIFIED: v4 — Compute cell size from average point spacing
  float bbox_area = (x_max - x_min) * (y_max - y_min);
  float avg_spacing = sqrtf(bbox_area / (float)N);
  float cell_size = avg_spacing * sqrtf((float)k);
  // cell_size is chosen so that a single cell contains approximately k points on average

  // Guard against degenerate cell_size (all points collinear or coincident)
  if (cell_size < 1e-6f) cell_size = 1.0f;

  // MODIFIED: v4 — Compute grid dimensions
  int grid_w = (int)ceilf((x_max - x_min) / cell_size) + 1;
  int grid_h = (int)ceilf((y_max - y_min) / cell_size) + 1;
  int num_cells = grid_w * grid_h;

  if (g_verbose) std::cerr << "v4 hash grid: k=" << k
            << " cell_size=" << cell_size
            << " grid=" << grid_w << "x" << grid_h
            << " num_cells=" << num_cells << "\n";

  // ======================================================================
  // MODIFIED: v4 — Step 2: Build spatial hash grid on GPU
  // ======================================================================
  int* d_cell_ids;
  int* d_point_ids;
  int* d_cell_start;
  int* d_cell_end;

  CUDA_CHECK(200, cudaMalloc(&d_cell_ids,   N * sizeof(int)));
  CUDA_CHECK(201, cudaMalloc(&d_point_ids,  N * sizeof(int)));
  CUDA_CHECK(202, cudaMalloc(&d_cell_start, num_cells * sizeof(int)));
  CUDA_CHECK(203, cudaMalloc(&d_cell_end,   num_cells * sizeof(int)));

  // Kernel 1 — Assign each point to a cell
  assignCellsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
      d_x, d_y, d_cell_ids, d_point_ids,
      x_min, y_min, cell_size, grid_w, grid_h, N
  );
  CUDA_CHECK(204, cudaGetLastError());
  CUDA_CHECK(205, cudaDeviceSynchronize());

  // Sort points by cell using Thrust
  {
    thrust::device_ptr<int> cell_ptr(d_cell_ids);
    thrust::device_ptr<int> point_ptr(d_point_ids);
    thrust::sort_by_key(cell_ptr, cell_ptr + N, point_ptr);
  }

  // Kernel 2 — Find start and end of each cell in the sorted array
  CUDA_CHECK(206, cudaMemset(d_cell_start, -1, num_cells * sizeof(int)));
  CUDA_CHECK(207, cudaMemset(d_cell_end,   -1, num_cells * sizeof(int)));

  findCellBoundsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
      d_cell_ids, d_cell_start, d_cell_end, N
  );
  CUDA_CHECK(208, cudaGetLastError());
  CUDA_CHECK(209, cudaDeviceSynchronize());

  if (g_verbose) std::cerr << "v4: Spatial hash grid built.\n";

  // ======================================================================
  // MODIFIED: v4.1 — Step 4: Borůvka loop with spiral search KNN
  // ======================================================================
  int phase = 0;
  int h_mst_count = 0;

  bool use_knn = true;  // flag: use KNN kernel or full scan kernel
  int boruvka_iter = 0;
  int r_max = max(grid_w, grid_h);  // maximum possible ring radius

  // MODIFIED: v4.1 — Allocate KNN output buffers
  int* d_knn_indices = nullptr;
  double* d_knn_dists = nullptr;
  if (use_knn) {
    CUDA_CHECK(300, cudaMalloc(&d_knn_indices, (size_t)N * k * sizeof(int)));
    CUDA_CHECK(301, cudaMalloc(&d_knn_dists,   (size_t)N * k * sizeof(double)));
  }

  // MODIFIED: v4.1 — Timing events for spiral search kernel
  cudaEvent_t knn_start_evt, knn_stop_evt;
  cudaEventCreate(&knn_start_evt);
  cudaEventCreate(&knn_stop_evt);
  float knn_total_ms = 0.0f;

  // Use 256 threads for spiral search kernel (matches __launch_bounds__)
  const int BLK_SPIRAL = 256;
  const int GRD_SPIRAL = (N + BLK_SPIRAL - 1) / BLK_SPIRAL;

  // ---- BORŮVKA MAIN LOOP ----
  while (h_mst_count < N - 1) {
    phase++;

    // MODIFIED: v4.1 — Choose between spiral search KNN and full O(n²) scan
    if (use_knn) {
        // --- Launch spiral search KNN kernel with timing ---
        // cudaEventRecord(knn_start_evt);

        knnSpiralSearchKernel<<<GRD_SPIRAL, BLK_SPIRAL>>>(
            d_x, d_y,
            d_point_ids, d_cell_start, d_cell_end,
            d_knn_indices, d_knn_dists,
            x_min, y_min, cell_size,
            grid_w, grid_h,
            k, r_max, N
        );
        CUDA_CHECK(120, cudaGetLastError());

        // Extract cheapest cross-component neighbor from k-NN list
        extractCheapestFromKNN<<<GRD_BORUVKA, BLK_BORUVKA>>>(
            d_knn_indices, d_knn_dists, d_comp,
            d_cheapest_to, d_cheapest_w,
            k, N
        );
        CUDA_CHECK(121, cudaGetLastError());

        // cudaEventRecord(knn_stop_evt);
        // cudaEventSynchronize(knn_stop_evt);
        // float knn_ms = 0.0f;
        // cudaEventElapsedTime(&knn_ms, knn_start_evt, knn_stop_evt);
        // knn_total_ms += knn_ms;

        // if (g_verbose) std::cerr << "v4.1: Spiral KNN phase " << phase
        //           << " took " << knn_ms << " ms\n";

        // --- Validation: fallback for points with no valid neighbor ---
        bool any_missed = checkMissedEdges(d_cheapest_w, N);
        if (any_missed) {
            // Launch fallback kernel for degenerate points
            knnFallbackKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
                d_x, d_y, d_comp,
                d_cheapest_to, d_cheapest_w,
                d_knn_indices,
                k, N
            );
            CUDA_CHECK(122, cudaGetLastError());
            CUDA_CHECK(123, cudaDeviceSynchronize());
            if (g_verbose) std::cerr << "v4.1: Fallback kernel launched for missed edges\n";
        }

        // After first KNN iteration, switch to full scan for remaining iterations
        // (components are larger, KNN may not find cross-component edges efficiently)
        use_knn = false;

    } else {
        // Use original full O(n²) scan kernel from gpuMDS-v3.1.cu
        boruvkaFindCheapest<<<GRD_BORUVKA, BLK_BORUVKA>>>(
            d_x, d_y, d_comp,
            d_cheapest_to, d_cheapest_w,
            N);
        CUDA_CHECK(120, cudaGetLastError());
        CUDA_CHECK(121, cudaDeviceSynchronize());
    }
    boruvka_iter++;

    // MODIFIED: v3.1 — Per-component best-edge reduction
    CUDA_CHECK(122, cudaMemset(d_comp_best, 0xFF, N * sizeof(unsigned long long)));
    findCompBestKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
        d_comp, d_cheapest_to, d_cheapest_w, d_comp_best, N);
    CUDA_CHECK(123, cudaGetLastError());
    CUDA_CHECK(124, cudaDeviceSynchronize());

    // MODIFIED: v3.1 — Only merge per-component best edges
    int prev_count = h_mst_count;
    mergeComponentsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
        d_parent, d_rank_uf, d_comp, d_cheapest_to, d_cheapest_w,
        d_comp_best,
        d_mst_u, d_mst_v, d_mst_w, d_mst_count,
        N, vrp.toRound);
    CUDA_CHECK(125, cudaGetLastError());
    CUDA_CHECK(126, cudaDeviceSynchronize());

    // Update component labels from parent array
    updateComponentsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(d_comp, d_parent, N);
    CUDA_CHECK(125, cudaGetLastError());
    CUDA_CHECK(126, cudaDeviceSynchronize());

    // Read current MST edge count from device
    CUDA_CHECK(124, cudaMemcpy(&h_mst_count, d_mst_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (g_verbose) std::cerr << "Borůvka phase " << phase
              << ": MST edges so far = " << h_mst_count
              << (use_knn ? " (KNN)" : " (full)") << "\n";

    if (h_mst_count == prev_count) {
      if (g_verbose) std::cerr << "Borůvka: no merges in phase " << phase
                << " — graph may be disconnected. MST edges: " << h_mst_count << "\n";
      break;
    }
  }

  if (g_verbose) std::cerr << "Borůvka complete: " << h_mst_count
            << " edges in " << phase << " phases ("
            << boruvka_iter << " iterations).\n";

  // MODIFIED: v4.1 — Print KNN spiral search total time
  if (g_verbose) std::cerr << "v4.1: Spiral search KNN total time: " << knn_total_ms << " ms\n";

  // MODIFIED: v4.1 — Free KNN buffers and timing events
  if (d_knn_indices) { CUDA_CHECK(310, cudaFree(d_knn_indices)); d_knn_indices = nullptr; }
  if (d_knn_dists)   { CUDA_CHECK(311, cudaFree(d_knn_dists));   d_knn_dists = nullptr; }
  cudaEventDestroy(knn_start_evt);
  cudaEventDestroy(knn_stop_evt);

  // ======================================================================
  // MODIFIED: v4 — Step 5: Memory management for hash grid
  // ======================================================================
  {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t hash_grid_size = (size_t)N * sizeof(int)           // d_cell_ids
                          + (size_t)N * sizeof(int)           // d_point_ids
                          + (size_t)num_cells * sizeof(int)   // d_cell_start
                          + (size_t)num_cells * sizeof(int);  // d_cell_end

    if (free_mem < hash_grid_size * 2) {
        // Not enough headroom — free the hash grid
        if (g_verbose) std::cerr << "v4: Freeing hash grid (low memory headroom).\n";
        cudaFree(d_cell_ids);
        cudaFree(d_point_ids);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
        d_cell_ids = nullptr;
        d_point_ids = nullptr;
        d_cell_start = nullptr;
        d_cell_end = nullptr;
    } else {
        if (g_verbose) std::cerr << "v4: Keeping hash grid alive (sufficient memory).\n";
    }
  }

  // ======================================================================
  // MODIFIED: v3.1 — GPU-resident CSR construction from accumulated edges
  // ======================================================================
  int num_mst_edges = h_mst_count;
  mst_nnz = 2 * num_mst_edges;  // each MST edge appears twice (bidirectional)

  // Allocate CSR arrays on device
  int* d_degree;
  CUDA_CHECK(140, cudaMalloc(&d_degree,   N * sizeof(int)));
  CUDA_CHECK(141, cudaMemset(d_degree, 0, N * sizeof(int)));
  CUDA_CHECK(142, cudaMalloc(&d_csr_row, (N + 1) * sizeof(int)));
  CUDA_CHECK(143, cudaMalloc(&d_csr_col,  mst_nnz * sizeof(int)));

  // Also build weights on device (double to match weight_t)
  double* d_csr_weights;
  CUDA_CHECK(144, cudaMalloc(&d_csr_weights, mst_nnz * sizeof(double)));

  // Step 1 — degree counting kernel
  int grd_sel = (num_mst_edges + BLK_BORUVKA - 1) / BLK_BORUVKA;
  if (grd_sel == 0) grd_sel = 1;  // guard against zero-edge case
  countDegreesKernel<<<grd_sel, BLK_BORUVKA>>>(
      d_mst_u, d_mst_v, d_degree, num_mst_edges);
  CUDA_CHECK(145, cudaGetLastError());
  CUDA_CHECK(146, cudaDeviceSynchronize());

  // Step 2 — build row pointer via Thrust exclusive scan
  {
    thrust::device_ptr<int> deg_ptr(d_degree);
    thrust::device_ptr<int> row_ptr(d_csr_row);
    thrust::exclusive_scan(deg_ptr, deg_ptr + N, row_ptr, 0);
  }
  // Write final value: csr_row_ptr[N] = mst_nnz
  CUDA_CHECK(147, cudaMemcpy(d_csr_row + N, &mst_nnz, sizeof(int), cudaMemcpyHostToDevice));

  // Step 3 — scatter kernel to fill col_idx and weights
  int* d_cursor;
  CUDA_CHECK(148, cudaMalloc(&d_cursor, N * sizeof(int)));
  CUDA_CHECK(149, cudaMemcpy(d_cursor, d_csr_row, N * sizeof(int), cudaMemcpyDeviceToDevice));

  scatterEdgesKernel<<<grd_sel, BLK_BORUVKA>>>(
      d_mst_u, d_mst_v, d_mst_w,
      d_cursor, d_csr_col, d_csr_weights,
      num_mst_edges);
  CUDA_CHECK(150, cudaGetLastError());
  CUDA_CHECK(151, cudaDeviceSynchronize());

  // ---- CLEANUP temporary GPU buffers ----
  CUDA_CHECK(160, cudaFree(d_cursor));
  CUDA_CHECK(161, cudaFree(d_degree));
  CUDA_CHECK(162, cudaFree(d_mst_u));
  CUDA_CHECK(163, cudaFree(d_mst_v));
  CUDA_CHECK(164, cudaFree(d_mst_w));
  CUDA_CHECK(165, cudaFree(d_mst_count));
  CUDA_CHECK(166, cudaFree(d_parent));
  CUDA_CHECK(167, cudaFree(d_rank_uf));
  CUDA_CHECK(168, cudaFree(d_comp_best));
  CUDA_CHECK(169, cudaFree(d_x));
  CUDA_CHECK(170, cudaFree(d_y));
  CUDA_CHECK(171, cudaFree(d_comp));
  CUDA_CHECK(172, cudaFree(d_cheapest_to));
  CUDA_CHECK(173, cudaFree(d_cheapest_w));
  CUDA_CHECK(174, cudaFree(d_csr_weights));  // not used by Part 2
  // MODIFIED: v4 — Free hash grid arrays if they were not already freed
  if (d_cell_ids)   cudaFree(d_cell_ids);
  if (d_point_ids)  cudaFree(d_point_ids);
  if (d_cell_start) cudaFree(d_cell_start);
  if (d_cell_end)   cudaFree(d_cell_end);
  // d_csr_row and d_csr_col are OUTPUT — freed by caller
}

// =========================================================================
//  PART 2: 100k LOOP — One CUDA thread per iteration  (same as gpuMDS.cu)
// =========================================================================

// On-the-fly Euclidean distance
__device__ __host__ inline double eucl_dist(double x1, double y1,
                                             double x2, double y2) {
  double dx = x1 - x2, dy = y1 - y2;
  return sqrt(dx * dx + dy * dy);
}

// -----------------------------------------------------------------------
// Initialize curand states
// -----------------------------------------------------------------------
__global__ void initRNG(curandState* states, unsigned long long seed, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  curand_init(seed, tid, 0, &states[tid]);
}

// -----------------------------------------------------------------------
// 100k kernel: one thread per iteration
// Each thread: shuffle MST neighbors, iterative DFS, compute cost
// -----------------------------------------------------------------------
__global__ void routeSearchKernelV2(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ demand,
    double capacity,
    int N,
    int  mst_nnz,
    const int*   __restrict__ csr_row_ptr,
    const int*   __restrict__ csr_col_idx,
    double* d_costs,
    unsigned long long seed,
    int*    d_buf_adj,   // per-thread adj scratch, size n_threads * mst_nnz
    int*    d_buf_ptr,   // per-thread nextChild,   size n_threads * (N+1)
    int*    d_buf_stk,   // per-thread DFS stack,   size n_threads * N
    bool*   d_buf_vis,   // per-thread visited,     size n_threads * N
    int*    d_buf_tour,  // per-thread tour,        size n_threads * N
    int n_threads)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_threads) return;

  curandState rng;
  curand_init(seed, tid, 0, &rng);

  // Slice per-thread buffers (coalesced: layout is [index * n_threads + tid])
  #define local_adj(X)   d_buf_adj[(long long)(X) * n_threads + tid]
  #define nextChild(X)   d_buf_ptr[(X) * n_threads + tid]
  #define stk(X)         d_buf_stk[(X) * n_threads + tid]
  #define visited(X)     d_buf_vis[(X) * n_threads + tid]
  #define local_tour(X)  d_buf_tour[(X) * n_threads + tid]

  // Copy MST neighbors and Fisher-Yates shuffle
  for (int v = 0; v < N; ++v) {
    int start = csr_row_ptr[v];
    int deg   = csr_row_ptr[v + 1] - start;
    for (int k = 0; k < deg; ++k)
      local_adj(start + k) = csr_col_idx[start + k];
    for (int k = deg - 1; k > 0; --k) {
      unsigned rval = curand(&rng);
      int j = rval % (unsigned)(k + 1);
      int tmp = local_adj(start + k);
      local_adj(start + k) = local_adj(start + j);
      local_adj(start + j) = tmp;
    }
    nextChild(v) = start;
  }

  // Iterative DFS from depot (node 0)
  for (int i = 0; i < N; ++i) visited(i) = false;
  int tour_len = 0;

  int top = 0;
  stk(top++) = 0;
  visited(0)  = true;
  local_tour(tour_len++) = 0;

  while (top > 0) {
    int v = stk(top - 1);
    bool pushed = false;
    while (nextChild(v) < csr_row_ptr[v + 1]) {
      int nc = nextChild(v); nextChild(v) = nc + 1;
      int u = local_adj(nc);
      if (!visited(u)) {
        visited(u) = true;
        local_tour(tour_len++) = u;
        stk(top++) = u;
        pushed = true;
        break;
      }
    }
    if (!pushed) top--;
  }

  // Compute CVRP cost on-the-fly
  double cost    = 0.0;
  double residue = capacity;
  double prevX   = x[0], prevY = y[0];

  for (int i = 0; i < tour_len; ++i) {
    int v = local_tour(i);
    if (v == 0) continue;
    if (residue - demand[v] < 0.0) {
      cost  += eucl_dist(prevX, prevY, x[0], y[0]);
      prevX  = x[0]; prevY = y[0];
      residue = capacity;
    }
    cost   += eucl_dist(prevX, prevY, x[v], y[v]);
    prevX   = x[v]; prevY = y[v];
    residue -= demand[v];
  }
  cost += eucl_dist(prevX, prevY, x[0], y[0]);

  d_costs[tid]    = cost;
}

// =========================================================================
//  PART 3: Post-processing — CPU (identical to seqMDS)
// =========================================================================

std::vector<std::vector<node_t>>
convertToVrpRoutes(const VRP& vrp, const std::vector<node_t>& tour) {
  std::vector<std::vector<node_t>> routes;
  demand_t cap     = vrp.getCapacity();
  demand_t residue = cap;
  std::vector<node_t> aRoute;
  for (auto v : tour) {
    if (v == 0) continue;
    if (residue - vrp.node[v].demand >= 0) {
      aRoute.push_back(v);
      residue -= vrp.node[v].demand;
    } else {
      routes.push_back(aRoute);
      aRoute.clear();
      aRoute.push_back(v);
      residue = cap - vrp.node[v].demand;
    }
  }
  if (!aRoute.empty()) routes.push_back(aRoute);
  return routes;
}

weight_t routeCost(const VRP& vrp, const std::vector<node_t>& route) {
  weight_t c = 0;
  node_t prev = 0;
  for (auto v : route) { c += vrp.get_dist(prev, v); prev = v; }
  return c + vrp.get_dist(prev, 0);
}

weight_t totalCost(const VRP& vrp,
                   const std::vector<std::vector<node_t>>& routes) {
  weight_t c = 0;
  for (auto& r : routes) c += routeCost(vrp, r);
  return c;
}

// 2-opt (identical to seqMDS)
void tsp_2opt(const VRP& vrp, std::vector<node_t>& cities) {
  unsigned sz = (unsigned)cities.size();
  if (sz <= 2) return;
  std::vector<node_t> tour(sz);
  unsigned improve = 0;
  while (improve < 2) {
    double best = routeCost(vrp, cities);
    for (unsigned i = 0; i < sz - 1; ++i) {
      for (unsigned k = i + 1; k < sz; ++k) {
        unsigned dec = 0;
        for (unsigned c = 0; c < i; ++c)      tour[c] = cities[c];
        for (unsigned c = i; c < k+1; ++c)    { tour[c] = cities[k-dec]; dec++; }
        for (unsigned c = k+1; c < sz; ++c)   tour[c] = cities[c];
        double nd = routeCost(vrp, tour);
        if (nd < best) { improve = 0; cities = tour; best = nd; }
      }
    }
    improve++;
  }
}

// tsp_approx (identical to seqMDS)
void tsp_approx(const VRP& vrp, std::vector<node_t>& cities,
                std::vector<node_t>& tour, int ncities) {
  for (int i = 1; i < ncities; i++) tour[i] = cities[i-1];
  tour[0] = cities[ncities-1];
  for (int i = 1; i < ncities; i++) {
    double ThisX = vrp.node[tour[i-1]].x, ThisY = vrp.node[tour[i-1]].y;
    double CloseDist = DBL_MAX;
    int ClosePt = i;
    for (int j = ncities-1; ; j--) {
      double dx = vrp.node[tour[j]].x - ThisX;
      double d  = dx*dx;
      if (d <= CloseDist) {
        double dy = vrp.node[tour[j]].y - ThisY;
        d += dy*dy;
        if (d <= CloseDist) {
          if (j < i) break;
          CloseDist = d; ClosePt = j;
        }
      }
    }
    std::swap(tour[i], tour[ClosePt]);
  }
}

std::vector<std::vector<node_t>>
postprocess_tsp_approx(const VRP& vrp,
                        const std::vector<std::vector<node_t>>& solRoutes) {
  std::vector<std::vector<node_t>> out;
  for (const auto& r : solRoutes) {
    unsigned sz = (unsigned)r.size();
    std::vector<node_t> cities(sz+1), tour(sz+1);
    for (unsigned j = 0; j < sz; ++j) cities[j] = r[j];
    cities[sz] = 0;
    tsp_approx(vrp, cities, tour, sz+1);
    std::vector<node_t> cr;
    for (unsigned k = 1; k < sz+1; ++k) cr.push_back(tour[k]);
    out.push_back(cr);
  }
  return out;
}

std::vector<std::vector<node_t>>
postprocess_2OPT(const VRP& vrp,
                 const std::vector<std::vector<node_t>>& routes) {
  std::vector<std::vector<node_t>> out;
  for (auto route : routes) { tsp_2opt(vrp, route); out.push_back(route); }
  return out;
}

std::vector<std::vector<node_t>>
postProcessIt(const VRP& vrp,
              std::vector<std::vector<node_t>>& minRoute,
              weight_t& minCost) {
  auto r1 = postprocess_tsp_approx(vrp, minRoute);
  auto r2 = postprocess_2OPT(vrp, r1);
  auto r3 = postprocess_2OPT(vrp, minRoute);
  std::vector<std::vector<node_t>> result;
  weight_t total = 0;
  for (unsigned z = 0; z < (unsigned)minRoute.size(); ++z) {
    weight_t c2 = routeCost(vrp, r2[z]);
    weight_t c3 = routeCost(vrp, r3[z]);
    if (c2 <= c3) { result.push_back(r2[z]); total += c2; }
    else          { result.push_back(r3[z]); total += c3; }
  }
  minCost = total;
  return result;
}

bool verify_sol(const VRP& vrp,
                const std::vector<std::vector<node_t>>& routes) {
  std::vector<int> hist(vrp.getSize(), 0);
  for (const auto& r : routes) {
    double sum = 0;
    for (auto v : r) { sum += vrp.node[v].demand; hist[v]++; }
    if (sum > vrp.getCapacity()) return false;
  }
  for (size_t i = 1; i < vrp.getSize(); ++i)
    if (hist[i] != 1) return false;
  return true;
}

void printOutput(const VRP& vrp,
                 const std::vector<std::vector<node_t>>& routes) {
  for (unsigned ii = 0; ii < (unsigned)routes.size(); ++ii) {
    std::cout << "Route #" << ii+1 << ":";
    for (auto v : routes[ii]) std::cout << " " << v;
    std::cout << "\n";
  }
  std::cout << "Cost " << totalCost(vrp, routes) << "\n";
}

// =========================================================================
//  MAIN
// =========================================================================
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "gpuMDS-v4\nUsage: " << argv[0]
              << " input.vrp [-round 0|1] [-v]\n";
    return 1;
  }

  VRP vrp;
  for (int ii = 2; ii < argc; ++ii) {
    if (std::string(argv[ii]) == "-v") {
      g_verbose = true;
    } else if (std::string(argv[ii]) == "-round" && ii + 1 < argc) {
      vrp.toRound = (bool)atoi(argv[ii+1]);
      ii++;  // skip the value
    }
  }

  vrp.read(argv[1]);
  int N = (int)vrp.getSize();
  if (g_verbose) std::cerr << "N=" << N << " capacity=" << vrp.getCapacity() << "\n";

  auto t_start = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------------------------
  // PART 1: Borůvka's MST on GPU (replaces Prim's from gpuMDS-v2.cu)
  // ------------------------------------------------------------------

  // MODIFIED: v3.1 — BoruvkaMST now outputs device CSR pointers directly
  int *d_csr_row, *d_csr_col;
  int mst_nnz;
  BoruvkaMST(vrp, N, d_csr_row, d_csr_col, mst_nnz);
  if (g_verbose) std::cerr << "MST nnz (bidirectional): " << mst_nnz << "\n";

  auto t_mst  = std::chrono::high_resolution_clock::now();
  double time_mst = std::chrono::duration<double>(t_mst - t_start).count();
  if (g_verbose) std::cerr << "Part 1 (MST) time: " << time_mst << " s\n";

  // ------------------------------------------------------------------
  // PART 2: 100k loop on GPU
  // ------------------------------------------------------------------

  // Copy node data to GPU
  std::vector<double> hx(N), hy(N), hd(N);
  for (int i = 0; i < N; ++i) {
    hx[i] = vrp.node[i].x;
    hy[i] = vrp.node[i].y;
    hd[i] = vrp.node[i].demand;
  }

  double *d_x, *d_y, *d_demand;
  CUDA_CHECK(1, cudaMalloc(&d_x,      N * sizeof(double)));
  CUDA_CHECK(2, cudaMalloc(&d_y,      N * sizeof(double)));
  CUDA_CHECK(3, cudaMalloc(&d_demand, N * sizeof(double)));
  CUDA_CHECK(4, cudaMemcpy(d_x,      hx.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(5, cudaMemcpy(d_y,      hy.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(6, cudaMemcpy(d_demand, hd.data(), N*sizeof(double), cudaMemcpyHostToDevice));

  const int N_ITER = 1000;
  const int BLK    = 16;
  const int GRD    = (N_ITER + BLK - 1) / BLK;

  // Per-thread cost array
  double* d_costs;
  CUDA_CHECK(11, cudaMalloc(&d_costs, N_ITER * sizeof(double)));

  // Per-thread scratchpad buffers
  long long buf_adj_sz  = (long long)N_ITER * mst_nnz;
  long long buf_ptr_sz  = (long long)N_ITER * (N + 1);
  long long buf_stk_sz  = (long long)N_ITER * N;
  long long buf_vis_sz  = (long long)N_ITER * N;
  long long buf_tour_sz = (long long)N_ITER * N;

  if (g_verbose) {
    std::cerr << "Allocating per-thread GPU buffers...\n";
    std::cerr << "  adj:  " << buf_adj_sz  * sizeof(int)  / (1LL<<30) << " GB\n";
    std::cerr << "  ptr:  " << buf_ptr_sz  * sizeof(int)  / (1LL<<30) << " GB\n";
    std::cerr << "  stk:  " << buf_stk_sz  * sizeof(int)  / (1LL<<30) << " GB\n";
    std::cerr << "  vis:  " << buf_vis_sz  * sizeof(bool) / (1LL<<30) << " GB\n";
    std::cerr << "  tour: " << buf_tour_sz * sizeof(int)  / (1LL<<30) << " GB\n";
  }

  int*  d_buf_adj,  *d_buf_ptr, *d_buf_stk, *d_buf_tour;
  bool* d_buf_vis;
  CUDA_CHECK(12, cudaMalloc(&d_buf_adj,  buf_adj_sz  * sizeof(int)));
  CUDA_CHECK(13, cudaMalloc(&d_buf_ptr,  buf_ptr_sz  * sizeof(int)));
  CUDA_CHECK(14, cudaMalloc(&d_buf_stk,  buf_stk_sz  * sizeof(int)));
  CUDA_CHECK(15, cudaMalloc(&d_buf_vis,  buf_vis_sz  * sizeof(bool)));
  CUDA_CHECK(16, cudaMalloc(&d_buf_tour, buf_tour_sz * sizeof(int)));

  if (g_verbose) std::cerr << "Launching 1k route search kernel...\n";

  routeSearchKernelV2<<<GRD, BLK>>>(
      d_x, d_y, d_demand, (double)vrp.getCapacity(), N, mst_nnz,
      d_csr_row, d_csr_col, d_costs, (unsigned long long)time(nullptr),
      d_buf_adj, d_buf_ptr, d_buf_stk, d_buf_vis, d_buf_tour,
      N_ITER);
  CUDA_CHECK(17, cudaGetLastError());
  CUDA_CHECK(18, cudaDeviceSynchronize());

  // Find best thread
  thrust::device_ptr<double> dp_costs(d_costs);
  auto best_it  = thrust::min_element(thrust::device, dp_costs, dp_costs + N_ITER);
  int  best_tid = (int)(best_it - dp_costs);
  double best_cost_gpu;
  CUDA_CHECK(19, cudaMemcpy(&best_cost_gpu, d_costs + best_tid,
                         sizeof(double), cudaMemcpyDeviceToHost));

  if (g_verbose) std::cerr << "Best GPU cost: " << best_cost_gpu
            << " (thread " << best_tid << ")\n";

  // Copy best tour back
  std::vector<int> best_tour(N);

  for(int i=0; i<N; i++) {
    CUDA_CHECK(20, cudaMemcpy(&best_tour[i],
                           d_buf_tour + (long long)i * N_ITER + best_tid,
                           sizeof(int), cudaMemcpyDeviceToHost));
  }
  
  auto t_loop  = std::chrono::high_resolution_clock::now();
  double time_loop = std::chrono::duration<double>(t_loop - t_start).count();

  auto minRoute = convertToVrpRoutes(vrp, best_tour);
  weight_t minCost = totalCost(vrp, minRoute);

  // ------------------------------------------------------------------
  // PART 3: Post-processing on CPU (same as seqMDS)
  // ------------------------------------------------------------------
  auto postRoutes = postProcessIt(vrp, minRoute, minCost);

  auto t_end   = std::chrono::high_resolution_clock::now();
  double time_total = std::chrono::duration<double>(t_end - t_start).count();

  bool valid = verify_sol(vrp, postRoutes);
  // MODIFIED: v4 — output file name changed to v4.txt
  std::ofstream ofs("v4.1.txt", std::ios::app);
  if (ofs.is_open()) {
    ofs << argv[1]
        << "\tMinCost: " << minCost
        << "\tTimeMST: " << time_mst
        << "\tTimeLoop: " << time_loop - time_mst
        << "\tTimePostProcess: " << time_total - time_loop
        << "\tTimeTotal: " << time_total
        << "\t" << (valid ? "VALID" : "INVALID") << "\n";
    ofs.close();
  }
  
  // printOutput(vrp, postRoutes);

  // Cleanup
  CUDA_CHECK(21, cudaFree(d_x)); CUDA_CHECK(22, cudaFree(d_y));
  CUDA_CHECK(23, cudaFree(d_demand));
  CUDA_CHECK(24, cudaFree(d_csr_row)); CUDA_CHECK(25, cudaFree(d_csr_col));
  CUDA_CHECK(26, cudaFree(d_costs));
  CUDA_CHECK(27, cudaFree(d_buf_adj)); CUDA_CHECK(28, cudaFree(d_buf_ptr));
  CUDA_CHECK(29, cudaFree(d_buf_stk)); CUDA_CHECK(30, cudaFree(d_buf_vis));
  CUDA_CHECK(31, cudaFree(d_buf_tour));

  return 0;
}
