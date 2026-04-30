// gpuMDS.cu
// GPU-accelerated CVRP Solver using three-phase approach:
//   Part 1: Borůvka's MST on GPU (on-the-fly Euclidean distances)
//           - Spatial hash grid KNN acceleration for first Borůvka iteration
//           - Spiral search with early termination for KNN lookup
//           - Merge and CSR construction fully on GPU
//   Part 2: CUDA parallel 1k route search (one thread per iteration, memory coalesced)
//   Part 3: Sequential 2-opt post-processing on CPU

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
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/count.h>

// Compile-time max k for register-resident arrays in spiral search
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
// Types
// -----------------------------------------------------------------------
using point_t  = double;
using weight_t = double;
using demand_t = double;
using node_t   = int;

// Global verbose flag — controlled by -v argument
bool g_verbose = false;

// -----------------------------------------------------------------------
// Edge class
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
// VRP class — with precomputed dist table
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
// VRP::read
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
// VRP::cal_graph_dist — precomputes dist[] table (not used by GPU solver)
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
// Borůvka Phase Kernel — full O(n²) scan
// Each thread finds the cheapest cross-component edge for one node.
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

// -----------------------------------------------------------------------
// GPU device Union-Find with path splitting + atomicCAS merge
// -----------------------------------------------------------------------
__device__ int gpu_find(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

// Deterministic index ordering prevents cycle bug during concurrent merges
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
// GPU kernels for Borůvka merge and CSR construction
// -----------------------------------------------------------------------
__global__ void initUFKernel(int* parent, int* rank_uf, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    parent[i] = i;
    rank_uf[i] = 0;
}

// Per-component best-edge reduction using packed atomicMin.
// Packs (float weight, int index) into a single 64-bit value for atomic compare.
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

// Merge kernel — only merges the per-component best edge to avoid duplicates.
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
// Spatial hash grid kernels
// -----------------------------------------------------------------------

// Assign each point to a grid cell based on its (x, y) coordinate.
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

// Find the start and end index of each cell in the sorted point array.
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
// Spiral Search KNN Kernel
// -----------------------------------------------------------------------
// Visits hash grid cells ring-by-ring in order of increasing Chebyshev
// distance from the query cell (expanding outward).
//
// Early exit: Stops expanding when the closest possible point in the next
// ring (at Euclidean distance >= (r-1)*cell_size) is farther than the
// current k-th nearest neighbor distance AND k neighbors have been found.
//
// Expected behavior: ~1-3 rings visited on average for k = log2(n) with
// well-distributed 2D data. Falls back to full scan via knnFallbackKernel
// for degenerate distributions.
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

    // Initialize k-NN sorted array in registers
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

    // Spiral ring expansion
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

    // Write results
    for (int t = 0; t < kk; t++) {
        d_knn_indices[i * kk + t] = (t < found) ? heap_idx[t] : -1;
        d_knn_dists[i * kk + t]   = (t < found) ? (double)heap_dist[t] : (double)FLT_MAX;
    }
}

// -----------------------------------------------------------------------
// Extract cheapest cross-component neighbor from k-NN list.
// Called after knnSpiralSearchKernel to filter out same-component neighbors.
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
// Fallback kernel: full O(n) linear scan for points that failed to find
// any cross-component neighbor via spiral search (degenerate distributions).
// Only runs for points where knn_indices[i*k] == -1.
// -----------------------------------------------------------------------
__global__ void knnFallbackKernel(
    const double* __restrict__ d_x,
    const double* __restrict__ d_y,
    const int* __restrict__ d_component,
    int*   d_cheapest_to,
    double* d_cheapest_w,
    const int* __restrict__ d_knn_indices,
    int k, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Only run for points that found no valid neighbor
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

// Thrust functor to extract weight from cheapest edge results
struct ExtractWeight {
    const double* weights;
    __host__ __device__ ExtractWeight(const double* w) : weights(w) {}
    __host__ __device__ double operator()(int idx) const { return weights[idx]; }
};

// Check if any point failed to find a cross-component neighbor (weight == FLT_MAX)
bool checkMissedEdges(double* d_cheapest_w, int n) {
    thrust::device_ptr<double> w_ptr(d_cheapest_w);
    double max_w = thrust::reduce(w_ptr, w_ptr + n, (double)(-FLT_MAX), thrust::maximum<double>());
    return (max_w >= (double)FLT_MAX);
}

// -----------------------------------------------------------------------
// BoruvkaMST — builds MST on GPU and outputs device-resident CSR arrays.
//
// Algorithm:
//   1. Build spatial hash grid from (x, y) coordinates.
//   2. First Borůvka iteration: use KNN candidates from spiral search.
//   3. Remaining iterations: full O(n²) scan (components are large, KNN
//      may miss cross-component edges efficiently).
//   4. Merge using GPU Union-Find with per-component best-edge reduction.
//   5. Construct CSR adjacency on GPU via Thrust exclusive scan + scatter.
//
// Outputs d_csr_row and d_csr_col (device pointers) — freed by caller.
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

  CUDA_CHECK(755, cudaMalloc(&d_x,            N * sizeof(double)));
  CUDA_CHECK(756, cudaMalloc(&d_y,            N * sizeof(double)));
  CUDA_CHECK(757, cudaMalloc(&d_comp,         N * sizeof(int)));
  CUDA_CHECK(758, cudaMalloc(&d_cheapest_to,  N * sizeof(int)));
  CUDA_CHECK(759, cudaMalloc(&d_cheapest_w,   N * sizeof(double)));

  CUDA_CHECK(761, cudaMemcpy(d_x, h_x.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(762, cudaMemcpy(d_y, h_y.data(), N*sizeof(double), cudaMemcpyHostToDevice));

  // GPU Union-Find arrays
  int *d_parent, *d_rank_uf;
  CUDA_CHECK(766, cudaMalloc(&d_parent,  N * sizeof(int)));
  CUDA_CHECK(767, cudaMalloc(&d_rank_uf, N * sizeof(int)));

  // MST edge accumulator arrays (max N-1 edges across all phases)
  int *d_mst_u, *d_mst_v;
  double *d_mst_w;
  int *d_mst_count;  // atomic counter on device
  CUDA_CHECK(773, cudaMalloc(&d_mst_u,     (N - 1) * sizeof(int)));
  CUDA_CHECK(774, cudaMalloc(&d_mst_v,     (N - 1) * sizeof(int)));
  CUDA_CHECK(775, cudaMalloc(&d_mst_w,     (N - 1) * sizeof(double)));
  CUDA_CHECK(776, cudaMalloc(&d_mst_count, sizeof(int)));
  CUDA_CHECK(777, cudaMemset(d_mst_count, 0, sizeof(int)));

  // Per-component best-edge array (packed weight+index)
  unsigned long long *d_comp_best;
  CUDA_CHECK(781, cudaMalloc(&d_comp_best, N * sizeof(unsigned long long)));

  const int BLK_BORUVKA = 128;
  const int GRD_BORUVKA = (N + BLK_BORUVKA - 1) / BLK_BORUVKA;

  // Initialize UF: parent[i]=i, rank_uf[i]=0
  initUFKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(d_parent, d_rank_uf, N);
  CUDA_CHECK(788, cudaGetLastError());
  CUDA_CHECK(789, cudaDeviceSynchronize());

  // Initialize d_comp[i] = i
  CUDA_CHECK(792, cudaMemcpy(d_comp, d_parent, N * sizeof(int), cudaMemcpyDeviceToDevice));

  // ======================================================================
  // Step 1: Compute k and cell size
  // k = number of nearest neighbors per point used in the first iteration.
  // cell_size is chosen so that a single cell contains ~k points on average.
  // ======================================================================
  int k = max(8, (int)log2f((float)N));

  // Compute bounding box using Thrust reductions on device
  float x_min, x_max, y_min, y_max;
  {
    thrust::device_ptr<double> dx_ptr(d_x);
    thrust::device_ptr<double> dy_ptr(d_y);
    x_min = (float)*thrust::min_element(dx_ptr, dx_ptr + N);
    x_max = (float)*thrust::max_element(dx_ptr, dx_ptr + N);
    y_min = (float)*thrust::min_element(dy_ptr, dy_ptr + N);
    y_max = (float)*thrust::max_element(dy_ptr, dy_ptr + N);
  }

  // Compute cell size from average point spacing
  float bbox_area = (x_max - x_min) * (y_max - y_min);
  float avg_spacing = sqrtf(bbox_area / (float)N);
  float cell_size = avg_spacing * sqrtf((float)k);

  // Guard against degenerate cell_size (all points collinear or coincident)
  if (cell_size < 1e-6f) cell_size = 1.0f;

  // Compute grid dimensions
  int grid_w = (int)ceilf((x_max - x_min) / cell_size) + 1;
  int grid_h = (int)ceilf((y_max - y_min) / cell_size) + 1;
  int num_cells = grid_w * grid_h;

  if (g_verbose) std::cerr << "Hash grid: k=" << k
            << " cell_size=" << cell_size
            << " grid=" << grid_w << "x" << grid_h
            << " num_cells=" << num_cells << "\n";

  // ======================================================================
  // Step 2: Build spatial hash grid on GPU
  // ======================================================================
  int* d_cell_ids;
  int* d_point_ids;
  int* d_cell_start;
  int* d_cell_end;

  CUDA_CHECK(838, cudaMalloc(&d_cell_ids,   N * sizeof(int)));
  CUDA_CHECK(839, cudaMalloc(&d_point_ids,  N * sizeof(int)));
  CUDA_CHECK(840, cudaMalloc(&d_cell_start, num_cells * sizeof(int)));
  CUDA_CHECK(841, cudaMalloc(&d_cell_end,   num_cells * sizeof(int)));

  // Kernel 1 — Assign each point to a cell
  assignCellsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
      d_x, d_y, d_cell_ids, d_point_ids,
      x_min, y_min, cell_size, grid_w, grid_h, N
  );
  CUDA_CHECK(848, cudaGetLastError());
  CUDA_CHECK(849, cudaDeviceSynchronize());

  // Sort points by cell using Thrust
  {
    thrust::device_ptr<int> cell_ptr(d_cell_ids);
    thrust::device_ptr<int> point_ptr(d_point_ids);
    thrust::sort_by_key(cell_ptr, cell_ptr + N, point_ptr);
  }

  // Kernel 2 — Find start and end of each cell in the sorted array
  CUDA_CHECK(859, cudaMemset(d_cell_start, -1, num_cells * sizeof(int)));
  CUDA_CHECK(860, cudaMemset(d_cell_end,   -1, num_cells * sizeof(int)));

  findCellBoundsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
      d_cell_ids, d_cell_start, d_cell_end, N
  );
  CUDA_CHECK(865, cudaGetLastError());
  CUDA_CHECK(866, cudaDeviceSynchronize());

  if (g_verbose) std::cerr << "Spatial hash grid built.\n";

  // ======================================================================
  // Step 3: Borůvka main loop with spiral search KNN (first iteration only)
  // ======================================================================
  int phase = 0;
  int h_mst_count = 0;

  bool use_knn = true;  // flag: use KNN kernel or full scan kernel
  int boruvka_iter = 0;
  int r_max = max(grid_w, grid_h);  // maximum possible ring radius

  // Allocate KNN output buffers
  int* d_knn_indices = nullptr;
  double* d_knn_dists = nullptr;
  if (use_knn) {
    CUDA_CHECK(884, cudaMalloc(&d_knn_indices, (size_t)N * k * sizeof(int)));
    CUDA_CHECK(885, cudaMalloc(&d_knn_dists,   (size_t)N * k * sizeof(double)));
  }

  // Timing events for spiral search kernel
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

    // Choose between spiral search KNN and full O(n²) scan
    if (use_knn) {
        // Launch spiral search KNN kernel
        knnSpiralSearchKernel<<<GRD_SPIRAL, BLK_SPIRAL>>>(
            d_x, d_y,
            d_point_ids, d_cell_start, d_cell_end,
            d_knn_indices, d_knn_dists,
            x_min, y_min, cell_size,
            grid_w, grid_h,
            k, r_max, N
        );
        CUDA_CHECK(913, cudaGetLastError());

        // Extract cheapest cross-component neighbor from k-NN list
        extractCheapestFromKNN<<<GRD_BORUVKA, BLK_BORUVKA>>>(
            d_knn_indices, d_knn_dists, d_comp,
            d_cheapest_to, d_cheapest_w,
            k, N
        );
        CUDA_CHECK(921, cudaGetLastError());

        // Validation: fallback for points with no valid neighbor
        bool any_missed = checkMissedEdges(d_cheapest_w, N);
        if (any_missed) {
            // Launch fallback kernel for degenerate points
            knnFallbackKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
                d_x, d_y, d_comp,
                d_cheapest_to, d_cheapest_w,
                d_knn_indices,
                k, N
            );
            CUDA_CHECK(933, cudaGetLastError());
            CUDA_CHECK(934, cudaDeviceSynchronize());
            if (g_verbose) std::cerr << "Fallback kernel launched for missed edges\n";
        }

        // After first KNN iteration, switch to full scan for remaining iterations
        use_knn = false;

    } else {
        // Use full O(n²) scan kernel
        boruvkaFindCheapest<<<GRD_BORUVKA, BLK_BORUVKA>>>(
            d_x, d_y, d_comp,
            d_cheapest_to, d_cheapest_w,
            N);
        CUDA_CHECK(947, cudaGetLastError());
        CUDA_CHECK(948, cudaDeviceSynchronize());
    }
    boruvka_iter++;

    // Per-component best-edge reduction
    CUDA_CHECK(953, cudaMemset(d_comp_best, 0xFF, N * sizeof(unsigned long long)));
    findCompBestKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
        d_comp, d_cheapest_to, d_cheapest_w, d_comp_best, N);
    CUDA_CHECK(956, cudaGetLastError());
    CUDA_CHECK(957, cudaDeviceSynchronize());

    // Only merge per-component best edges
    int prev_count = h_mst_count;
    mergeComponentsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(
        d_parent, d_rank_uf, d_comp, d_cheapest_to, d_cheapest_w,
        d_comp_best,
        d_mst_u, d_mst_v, d_mst_w, d_mst_count,
        N, vrp.toRound);
    CUDA_CHECK(966, cudaGetLastError());
    CUDA_CHECK(967, cudaDeviceSynchronize());

    // Update component labels from parent array
    updateComponentsKernel<<<GRD_BORUVKA, BLK_BORUVKA>>>(d_comp, d_parent, N);
    CUDA_CHECK(971, cudaGetLastError());
    CUDA_CHECK(972, cudaDeviceSynchronize());

    // Read current MST edge count from device
    CUDA_CHECK(975, cudaMemcpy(&h_mst_count, d_mst_count, sizeof(int), cudaMemcpyDeviceToHost));

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

  if (g_verbose) std::cerr << "Spiral search KNN total time: " << knn_total_ms << " ms\n";

  // Free KNN buffers and timing events
  if (d_knn_indices) { CUDA_CHECK(995, cudaFree(d_knn_indices)); d_knn_indices = nullptr; }
  if (d_knn_dists)   { CUDA_CHECK(996, cudaFree(d_knn_dists));   d_knn_dists = nullptr; }
  cudaEventDestroy(knn_start_evt);
  cudaEventDestroy(knn_stop_evt);

  // ======================================================================
  // Step 4: Memory management for hash grid
  // Free hash grid early if GPU memory headroom is insufficient for later
  // allocations; otherwise keep alive to avoid re-allocation overhead.
  // ======================================================================
  {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t hash_grid_size = (size_t)N * sizeof(int)           // d_cell_ids
                          + (size_t)N * sizeof(int)           // d_point_ids
                          + (size_t)num_cells * sizeof(int)   // d_cell_start
                          + (size_t)num_cells * sizeof(int);  // d_cell_end

    if (free_mem < hash_grid_size * 2) {
        if (g_verbose) std::cerr << "Freeing hash grid (low memory headroom).\n";
        cudaFree(d_cell_ids);
        cudaFree(d_point_ids);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
        d_cell_ids = nullptr;
        d_point_ids = nullptr;
        d_cell_start = nullptr;
        d_cell_end = nullptr;
    } else {
        if (g_verbose) std::cerr << "Keeping hash grid alive (sufficient memory).\n";
    }
  }

  // ======================================================================
  // Step 5: GPU-resident CSR construction from accumulated MST edges
  //   1. Count degrees per node.
  //   2. Exclusive scan to build row pointer array.
  //   3. Scatter edges into col_idx and weights arrays.
  // ======================================================================
  int num_mst_edges = h_mst_count;
  mst_nnz = 2 * num_mst_edges;  // each MST edge appears twice (bidirectional)

  // Allocate CSR arrays on device
  int* d_degree;
  CUDA_CHECK(1040, cudaMalloc(&d_degree,   N * sizeof(int)));
  CUDA_CHECK(1041, cudaMemset(d_degree, 0, N * sizeof(int)));
  CUDA_CHECK(1042, cudaMalloc(&d_csr_row, (N + 1) * sizeof(int)));
  CUDA_CHECK(1043, cudaMalloc(&d_csr_col,  mst_nnz * sizeof(int)));

  // Also build weights on device (double to match weight_t)
  double* d_csr_weights;
  CUDA_CHECK(1047, cudaMalloc(&d_csr_weights, mst_nnz * sizeof(double)));

  // Step 1 — degree counting kernel
  int grd_sel = (num_mst_edges + BLK_BORUVKA - 1) / BLK_BORUVKA;
  if (grd_sel == 0) grd_sel = 1;  // guard against zero-edge case
  countDegreesKernel<<<grd_sel, BLK_BORUVKA>>>(
      d_mst_u, d_mst_v, d_degree, num_mst_edges);
  CUDA_CHECK(1054, cudaGetLastError());
  CUDA_CHECK(1055, cudaDeviceSynchronize());

  // Step 2 — build row pointer via Thrust exclusive scan
  {
    thrust::device_ptr<int> deg_ptr(d_degree);
    thrust::device_ptr<int> row_ptr(d_csr_row);
    thrust::exclusive_scan(deg_ptr, deg_ptr + N, row_ptr, 0);
  }
  // Write final value: csr_row_ptr[N] = mst_nnz
  CUDA_CHECK(1064, cudaMemcpy(d_csr_row + N, &mst_nnz, sizeof(int), cudaMemcpyHostToDevice));

  // Step 3 — scatter kernel to fill col_idx and weights
  int* d_cursor;
  CUDA_CHECK(1068, cudaMalloc(&d_cursor, N * sizeof(int)));
  CUDA_CHECK(1069, cudaMemcpy(d_cursor, d_csr_row, N * sizeof(int), cudaMemcpyDeviceToDevice));

  scatterEdgesKernel<<<grd_sel, BLK_BORUVKA>>>(
      d_mst_u, d_mst_v, d_mst_w,
      d_cursor, d_csr_col, d_csr_weights,
      num_mst_edges);
  CUDA_CHECK(1075, cudaGetLastError());
  CUDA_CHECK(1076, cudaDeviceSynchronize());

  // ---- CLEANUP temporary GPU buffers ----
  CUDA_CHECK(1079, cudaFree(d_cursor));
  CUDA_CHECK(1080, cudaFree(d_degree));
  CUDA_CHECK(1081, cudaFree(d_mst_u));
  CUDA_CHECK(1082, cudaFree(d_mst_v));
  CUDA_CHECK(1083, cudaFree(d_mst_w));
  CUDA_CHECK(1084, cudaFree(d_mst_count));
  CUDA_CHECK(1085, cudaFree(d_parent));
  CUDA_CHECK(1086, cudaFree(d_rank_uf));
  CUDA_CHECK(1087, cudaFree(d_comp_best));
  CUDA_CHECK(1088, cudaFree(d_x));
  CUDA_CHECK(1089, cudaFree(d_y));
  CUDA_CHECK(1090, cudaFree(d_comp));
  CUDA_CHECK(1091, cudaFree(d_cheapest_to));
  CUDA_CHECK(1092, cudaFree(d_cheapest_w));
  CUDA_CHECK(1093, cudaFree(d_csr_weights));  // not used by Part 2
  // Free hash grid arrays if they were not already freed
  if (d_cell_ids)   cudaFree(d_cell_ids);
  if (d_point_ids)  cudaFree(d_point_ids);
  if (d_cell_start) cudaFree(d_cell_start);
  if (d_cell_end)   cudaFree(d_cell_end);
  // d_csr_row and d_csr_col are OUTPUT — freed by caller
}

// =========================================================================
//  PART 2: 1k Route Search Loop — One CUDA thread per iteration
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
// Route search kernel: one thread per iteration.
// Each thread: shuffle MST neighbors, iterative DFS, compute CVRP cost.
// Per-thread buffers are laid out in column-major order for memory coalescing.
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
//  PART 3: Post-processing — CPU
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

// 2-opt local search for a single route
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

// Nearest-neighbor greedy tour construction for a single route
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

// Apply both nearest-neighbor and 2-opt post-processing, keep the better result per route.
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
    std::cerr << "gpuMDS\nUsage: " << argv[0]
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
  // PART 1: Borůvka's MST on GPU
  // Outputs device CSR pointers directly (freed after Part 2).
  // ------------------------------------------------------------------
  int *d_csr_row, *d_csr_col;
  int mst_nnz;
  BoruvkaMST(vrp, N, d_csr_row, d_csr_col, mst_nnz);
  if (g_verbose) std::cerr << "MST nnz (bidirectional): " << mst_nnz << "\n";

  auto t_mst  = std::chrono::high_resolution_clock::now();
  double time_mst = std::chrono::duration<double>(t_mst - t_start).count();
  if (g_verbose) std::cerr << "Part 1 (MST) time: " << time_mst << " s\n";

  // ------------------------------------------------------------------
  // PART 2: 1k route search loop on GPU
  // ------------------------------------------------------------------

  // Copy node data to GPU
  std::vector<double> hx(N), hy(N), hd(N);
  for (int i = 0; i < N; ++i) {
    hx[i] = vrp.node[i].x;
    hy[i] = vrp.node[i].y;
    hd[i] = vrp.node[i].demand;
  }

  double *d_x, *d_y, *d_demand;
  CUDA_CHECK(1429, cudaMalloc(&d_x,      N * sizeof(double)));
  CUDA_CHECK(1430, cudaMalloc(&d_y,      N * sizeof(double)));
  CUDA_CHECK(1431, cudaMalloc(&d_demand, N * sizeof(double)));
  CUDA_CHECK(1432, cudaMemcpy(d_x,      hx.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(1433, cudaMemcpy(d_y,      hy.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(1434, cudaMemcpy(d_demand, hd.data(), N*sizeof(double), cudaMemcpyHostToDevice));

  const int N_ITER = 1000;
  const int BLK    = 16;
  const int GRD    = (N_ITER + BLK - 1) / BLK;

  // Per-thread cost array
  double* d_costs;
  CUDA_CHECK(1442, cudaMalloc(&d_costs, N_ITER * sizeof(double)));

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
  CUDA_CHECK(1462, cudaMalloc(&d_buf_adj,  buf_adj_sz  * sizeof(int)));
  CUDA_CHECK(1463, cudaMalloc(&d_buf_ptr,  buf_ptr_sz  * sizeof(int)));
  CUDA_CHECK(1464, cudaMalloc(&d_buf_stk,  buf_stk_sz  * sizeof(int)));
  CUDA_CHECK(1465, cudaMalloc(&d_buf_vis,  buf_vis_sz  * sizeof(bool)));
  CUDA_CHECK(1466, cudaMalloc(&d_buf_tour, buf_tour_sz * sizeof(int)));

  if (g_verbose) std::cerr << "Launching 1k route search kernel...\n";

  routeSearchKernelV2<<<GRD, BLK>>>(
      d_x, d_y, d_demand, (double)vrp.getCapacity(), N, mst_nnz,
      d_csr_row, d_csr_col, d_costs, (unsigned long long)time(nullptr),
      d_buf_adj, d_buf_ptr, d_buf_stk, d_buf_vis, d_buf_tour,
      N_ITER);
  CUDA_CHECK(1475, cudaGetLastError());
  CUDA_CHECK(1476, cudaDeviceSynchronize());

  // Find best thread
  thrust::device_ptr<double> dp_costs(d_costs);
  auto best_it  = thrust::min_element(thrust::device, dp_costs, dp_costs + N_ITER);
  int  best_tid = (int)(best_it - dp_costs);
  double best_cost_gpu;
  CUDA_CHECK(1483, cudaMemcpy(&best_cost_gpu, d_costs + best_tid,
                         sizeof(double), cudaMemcpyDeviceToHost));

  if (g_verbose) std::cerr << "Best GPU cost: " << best_cost_gpu
            << " (thread " << best_tid << ")\n";

  // Copy best tour back to host
  std::vector<int> best_tour(N);

  for(int i=0; i<N; i++) {
    CUDA_CHECK(1493, cudaMemcpy(&best_tour[i],
                           d_buf_tour + (long long)i * N_ITER + best_tid,
                           sizeof(int), cudaMemcpyDeviceToHost));
  }
  
  auto t_loop  = std::chrono::high_resolution_clock::now();
  double time_loop = std::chrono::duration<double>(t_loop - t_start).count();

  auto minRoute = convertToVrpRoutes(vrp, best_tour);
  weight_t minCost = totalCost(vrp, minRoute);

  // ------------------------------------------------------------------
  // PART 3: Post-processing on CPU
  // ------------------------------------------------------------------
  auto postRoutes = postProcessIt(vrp, minRoute, minCost);

  auto t_end   = std::chrono::high_resolution_clock::now();
  double time_total = std::chrono::duration<double>(t_end - t_start).count();

  bool valid = verify_sol(vrp, postRoutes);
  std::ofstream ofs("output.txt", std::ios::app);
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
  CUDA_CHECK(1528, cudaFree(d_x)); CUDA_CHECK(1528, cudaFree(d_y));
  CUDA_CHECK(1529, cudaFree(d_demand));
  CUDA_CHECK(1530, cudaFree(d_csr_row)); CUDA_CHECK(1530, cudaFree(d_csr_col));
  CUDA_CHECK(1531, cudaFree(d_costs));
  CUDA_CHECK(1532, cudaFree(d_buf_adj)); CUDA_CHECK(1532, cudaFree(d_buf_ptr));
  CUDA_CHECK(1533, cudaFree(d_buf_stk)); CUDA_CHECK(1533, cudaFree(d_buf_vis));
  CUDA_CHECK(1534, cudaFree(d_buf_tour));

  return 0;
}
