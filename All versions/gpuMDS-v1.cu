// gpuMDS-v1.cu
// Hybrid: Part 1 = seqMDS Prim's MST (CPU, precomputed dist table)
//         Part 2 = CUDA parallel 1k route search (one thread per iteration)
//         Part 3 = seqMDS 2-opt post-processing (CPU)
//
// Compile: nvcc -O3 -std=c++17 -o gpuMDS-v1.out gpuMDS-v1.cu -lcurand

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
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

// -----------------------------------------------------------------------
// Error checking macro
// -----------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = (call);                                               \
    if (err != cudaSuccess) {                                               \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                << " — " << cudaGetErrorString(err) << "\n";               \
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

const node_t DEPOT = 0;

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
    if (i == j) return 0.0;
    if (i > j) std::swap(i, j);
    size_t myoffset   = ((2 * i * size) - (i * i) + i) / 2;
    size_t correction = 2 * i + 1;
    return dist[myoffset + j - correction];
  }

  // Build complete graph + precompute distance table (same as seqMDS)
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
// VRP::cal_graph_dist  (identical to seqMDS — precomputes dist[])
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
      nG[i].push_back(Edge(j, dist[k]));
      nG[j].push_back(Edge(i, dist[k]));
      k++;
    }
  }
  return nG;
}

// -----------------------------------------------------------------------
// Prim's MST  (identical to seqMDS)
// -----------------------------------------------------------------------
std::vector<std::vector<Edge>>
PrimsAlgo(const VRP& vrp, std::vector<std::vector<Edge>>& graph) {
  auto N = graph.size();
  std::vector<weight_t> key(N, INT_MAX);
  std::vector<weight_t> toEdges(N, -1);
  std::vector<bool>     visited(N, false);
  std::set<std::pair<weight_t, node_t>> active;
  std::vector<std::vector<Edge>> nG(N);

  key[0] = 0.0;
  active.insert({0.0, 0});

  while (!active.empty()) {
    auto where = active.begin()->second;
    active.erase(active.begin());
    if (visited[where]) continue;
    visited[where] = true;
    for (Edge E : graph[where]) {
      if (!visited[E.to] && E.length < key[E.to]) {
        key[E.to] = E.length;
        active.insert({key[E.to], E.to});
        toEdges[E.to] = where;
      }
    }
  }

  node_t u = 0;
  for (auto v : toEdges) {
    if (v != -1) {
      weight_t w = vrp.get_dist(u, (node_t)v);
      nG[u].push_back(Edge((node_t)v, w));
      nG[(node_t)v].push_back(Edge(u, w));
    }
    u++;
  }
  return nG;
}

// -----------------------------------------------------------------------
// Convert MST adjacency list → CSR  (CPU)
// -----------------------------------------------------------------------
struct CSR {
  std::vector<int>   row_ptr;
  std::vector<int>   col_idx;
  std::vector<float> weights;
};

CSR adjListToCSR(const std::vector<std::vector<Edge>>& mst) {
  int N = (int)mst.size();
  CSR csr;
  csr.row_ptr.resize(N + 1, 0);

  for (int v = 0; v < N; ++v)
    csr.row_ptr[v + 1] = (int)mst[v].size();
  for (int v = 1; v <= N; ++v)
    csr.row_ptr[v] += csr.row_ptr[v - 1];

  int nnz = csr.row_ptr[N];
  csr.col_idx.resize(nnz);
  csr.weights.resize(nnz);

  for (int v = 0; v < N; ++v) {
    int off = csr.row_ptr[v];
    for (int k = 0; k < (int)mst[v].size(); ++k) {
      csr.col_idx[off + k] = mst[v][k].to;
      csr.weights[off + k] = (float)mst[v][k].length;
    }
  }
  return csr;
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

  //   curandState rng = rng_states[tid];

  // Slice per-thread buffers
  int*  local_adj = d_buf_adj  + (long long)tid * mst_nnz;
  int*  nextChild = d_buf_ptr  + (long long)tid * (N + 1);
  int*  stk       = d_buf_stk  + (long long)tid * N;
  bool* visited   = d_buf_vis  + (long long)tid * N;
  int*  tour      = d_buf_tour + (long long)tid * N;

  // Copy MST neighbors and Fisher-Yates shuffle
  for (int v = 0; v < N; ++v) {
    int start = csr_row_ptr[v];
    int deg   = csr_row_ptr[v + 1] - start;
    for (int k = 0; k < deg; ++k)
      local_adj[start + k] = csr_col_idx[start + k];
    for (int k = deg - 1; k > 0; --k) {
      unsigned rval = curand(&rng);
      int j = rval % (unsigned)(k + 1);
      int tmp = local_adj[start + k];
      local_adj[start + k] = local_adj[start + j];
      local_adj[start + j] = tmp;
    }
    nextChild[v] = start;
  }

  // Iterative DFS from depot (node 0)
  for (int i = 0; i < N; ++i) visited[i] = false;
  int tour_len = 0;

  int top = 0;
  stk[top++] = 0;
  visited[0]  = true;
  tour[tour_len++] = 0;

  while (top > 0) {
    int v = stk[top - 1];
    bool pushed = false;
    while (nextChild[v] < csr_row_ptr[v + 1]) {
      int u = local_adj[nextChild[v]++];
      if (!visited[u]) {
        visited[u] = true;
        tour[tour_len++] = u;
        stk[top++] = u;
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
    int v = tour[i];
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
//   rng_states[tid] = rng;
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
    std::cerr << "gpuMDS-v1\nUsage: " << argv[0]
              << " input.vrp [-round 0|1]\n";
    return 1;
  }

  VRP vrp;
  for (int ii = 2; ii < argc; ii += 2) {
    if (std::string(argv[ii]) == "-round")
      vrp.toRound = (bool)atoi(argv[ii+1]);
  }

  vrp.read(argv[1]);
  int N = (int)vrp.getSize();
  std::cerr << "N=" << N << " capacity=" << vrp.getCapacity() << "\n";

  auto t_start = std::chrono::high_resolution_clock::now();

  // ------------------------------------------------------------------
  // PART 1: Prim's MST on CPU (same as seqMDS)
  // ------------------------------------------------------------------
  auto cG   = vrp.cal_graph_dist();   // builds dist[] table + adj list
  auto mstG = PrimsAlgo(vrp, cG);    // Prim's MST → adjacency list

  // Convert MST adj list → CSR for GPU
  CSR mst    = adjListToCSR(mstG);
  int mst_nnz = (int)mst.col_idx.size();  // 2*(N-1)
  std::cerr << "MST edges (Prim's): " << mst_nnz/2 << "\n";

  auto t_mst  = std::chrono::high_resolution_clock::now();
  double time_mst = std::chrono::duration<double>(t_mst - t_start).count();
  std::cerr << "Part 1 (MST) time: " << time_mst << " s\n";

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
  CUDA_CHECK(cudaMalloc(&d_x,      N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_y,      N * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_demand, N * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(d_x,      hx.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y,      hy.data(), N*sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_demand, hd.data(), N*sizeof(double), cudaMemcpyHostToDevice));

  // Copy MST CSR to GPU
  int *d_csr_row, *d_csr_col;
  CUDA_CHECK(cudaMalloc(&d_csr_row, (N+1)    * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_csr_col,  mst_nnz * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_csr_row, mst.row_ptr.data(), (N+1)*sizeof(int),    cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_csr_col, mst.col_idx.data(), mst_nnz*sizeof(int),  cudaMemcpyHostToDevice));

  const int N_ITER = 10000;
  const int BLK    = 16;
  const int GRD    = (N_ITER + BLK - 1) / BLK;

  // RNG states
//   curandState* d_rng;
//   CUDA_CHECK(cudaMalloc(&d_rng, N_ITER * sizeof(curandState)));
//   initRNG<<<GRD, BLK>>>(d_rng, (unsigned long long)time(nullptr), N_ITER);
//   CUDA_CHECK(cudaGetLastError());
//   CUDA_CHECK(cudaDeviceSynchronize());

  // Per-thread cost array
  double* d_costs;
  CUDA_CHECK(cudaMalloc(&d_costs, N_ITER * sizeof(double)));

  // Per-thread scratchpad buffers
  long long buf_adj_sz  = (long long)N_ITER * mst_nnz;
  long long buf_ptr_sz  = (long long)N_ITER * (N + 1);
  long long buf_stk_sz  = (long long)N_ITER * N;
  long long buf_vis_sz  = (long long)N_ITER * N;
  long long buf_tour_sz = (long long)N_ITER * N;

  std::cerr << "Allocating per-thread GPU buffers...\n";
  std::cerr << "  adj:  " << buf_adj_sz  * sizeof(int)  / (1LL<<30) << " GB\n";
  std::cerr << "  ptr:  " << buf_ptr_sz  * sizeof(int)  / (1LL<<30) << " GB\n";
  std::cerr << "  stk:  " << buf_stk_sz  * sizeof(int)  / (1LL<<30) << " GB\n";
  std::cerr << "  vis:  " << buf_vis_sz  * sizeof(bool) / (1LL<<30) << " GB\n";
  std::cerr << "  tour: " << buf_tour_sz * sizeof(int)  / (1LL<<30) << " GB\n";

  int*  d_buf_adj,  *d_buf_ptr, *d_buf_stk, *d_buf_tour;
  bool* d_buf_vis;
  CUDA_CHECK(cudaMalloc(&d_buf_adj,  buf_adj_sz  * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_buf_ptr,  buf_ptr_sz  * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_buf_stk,  buf_stk_sz  * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_buf_vis,  buf_vis_sz  * sizeof(bool)));
  CUDA_CHECK(cudaMalloc(&d_buf_tour, buf_tour_sz * sizeof(int)));

  std::cerr << "Launching 100k route search kernel...\n";

  routeSearchKernelV2<<<GRD, BLK>>>(
      d_x, d_y, d_demand, (double)vrp.getCapacity(), N, mst_nnz,
      d_csr_row, d_csr_col, d_costs, (unsigned long long)time(nullptr),
      d_buf_adj, d_buf_ptr, d_buf_stk, d_buf_vis, d_buf_tour,
      N_ITER);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Find best thread
  thrust::device_ptr<double> dp_costs(d_costs);
  auto best_it  = thrust::min_element(thrust::device, dp_costs, dp_costs + N_ITER);
  int  best_tid = (int)(best_it - dp_costs);
  double best_cost_gpu;
  CUDA_CHECK(cudaMemcpy(&best_cost_gpu, d_costs + best_tid,
                         sizeof(double), cudaMemcpyDeviceToHost));

  std::cerr << "Best GPU cost: " << best_cost_gpu
            << " (thread " << best_tid << ")\n";

  // Copy best tour back
  std::vector<int> best_tour(N);
  CUDA_CHECK(cudaMemcpy(best_tour.data(),
                         d_buf_tour + (long long)best_tid * N,
                         N * sizeof(int), cudaMemcpyDeviceToHost));

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
  std::ofstream ofs("v1.txt", std::ios::app);
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
  // std::cerr << argv[1]
  //           << " Cost " << best_cost_gpu << " " << minCost
  //           << " Time(seconds) " << time_mst
  //           << " " << time_loop - time_mst
  //           << " " << time_total - time_loop
  //           << " " << time_total
  //           << (valid ? " VALID" : " INVALID") << "\n";

  // printOutput(vrp, postRoutes);

  // Cleanup
  CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_demand));
  CUDA_CHECK(cudaFree(d_csr_row)); CUDA_CHECK(cudaFree(d_csr_col));
//   CUDA_CHECK(cudaFree(d_rng)); 
  CUDA_CHECK(cudaFree(d_costs));
  CUDA_CHECK(cudaFree(d_buf_adj)); CUDA_CHECK(cudaFree(d_buf_ptr));
  CUDA_CHECK(cudaFree(d_buf_stk)); CUDA_CHECK(cudaFree(d_buf_vis));
  CUDA_CHECK(cudaFree(d_buf_tour));

  return 0;
}
