// File rct.cu

#include <ot/cuda/rct.cuh>
#include <ot/cuda/sort.cuh>
#include <ot/cuda/utils.cuh>
#include <ot/timer/_prof.hpp>

#define MAX_SPLIT_TRAN 4 
const int chunk = 64;

template <typename RctCUDAType>
__global__ void compute_net_timing(RctCUDAType rct) {
  unsigned int net_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int el_rf_offset = threadIdx.y;
  if(net_id >= rct.num_nets) return;
  
  int st = rct.rct_nodes_start[net_id], ed = rct.rct_nodes_start[net_id + 1];
  int st4 = st * MAX_SPLIT_TRAN + el_rf_offset, ed4 = ed * MAX_SPLIT_TRAN + el_rf_offset;
  int rst4 = ed4 - MAX_SPLIT_TRAN, red4 = st4;   // red4 = st4, jumping over the root

  // update load from cap
  // and init array
  for(int i = st4; i < ed4; i += MAX_SPLIT_TRAN) {
    rct.load[i] = rct.cap[i];
    rct.delay[i] = rct.ldelay[i] = rct.impulse[i] = 0;
  }

  // update load from downstream to upstream
  for(int i = rst4, j = ed - 1; i > red4; i -= MAX_SPLIT_TRAN, --j) {
    int prev = i - rct.rct_pid[j] * MAX_SPLIT_TRAN;
    rct.load[prev] += rct.load[i];
  }

  // update delay from upstream to downstream
  for(int i = st4 + MAX_SPLIT_TRAN, j = st + 1; i < ed4; i += MAX_SPLIT_TRAN, ++j) {
    int prev = i - rct.rct_pid[j] * MAX_SPLIT_TRAN;
    rct.delay[i] += rct.delay[prev] + rct.load[i] * rct.pres[j];
  }

  // update cap*delay from downstream to upstream
  for(int i = rst4, j = ed - 1; i > red4; i -= MAX_SPLIT_TRAN, --j) {
    int prev = i - rct.rct_pid[j] * MAX_SPLIT_TRAN;
    rct.ldelay[i] += rct.cap[i] * rct.delay[i];
    rct.ldelay[prev] += rct.ldelay[i];
  }
  rct.ldelay[st4] += rct.cap[st4] * rct.delay[st4];

  // update beta from upstream to downstream
  for(int i = st4 + MAX_SPLIT_TRAN, j = st + 1; i < ed4; i += MAX_SPLIT_TRAN, ++j) {
    int prev = i - rct.rct_pid[j] * MAX_SPLIT_TRAN;
    rct.impulse[i] += rct.impulse[prev] + rct.ldelay[i] * rct.pres[j];
  }

  // beta -> impulse
  for(int i = st4; i < ed4; i += MAX_SPLIT_TRAN) {
    float t = rct.delay[i];
    rct.impulse[i] = 2 * rct.impulse[i] - t * t;
  }
}

#ifdef RCT_BASELINE

template <typename RctCUDAType>
__device__ void dfs_load(RctCUDAType &rct, int st, int u, int k) {
  rct.load[(st + u) * 4 + k] = rct.cap[(st + u) * 4 + k];
  for(int i = (u == 0 ? 0 : rct.rct_edgecount[st + u - 1]);
      i < rct.rct_edgecount[st + u];
      ++i) {
    int v = rct.rct_edgeadj[st + i];
    dfs_load(rct, st, v, k);
    rct.load[(st + u) * 4 + k] += rct.load[(st + v) * 4 + k];
  }
}

template <typename RctCUDAType>
__device__ void dfs_delay(RctCUDAType &rct, int st, int u, int k) {
  for(int i = (u == 0 ? 0 : rct.rct_edgecount[st + u - 1]);
      i < rct.rct_edgecount[st + u];
      ++i) {
    int v = rct.rct_edgeadj[st + i];
    rct.delay[(st + v) * 4 + k] = rct.delay[(st + u) * 4 + k] + rct.pres[st + v] * rct.load[(st + v) * 4 + k];
    dfs_delay(rct, st, v, k);
  }
}

template <typename RctCUDAType>
__device__ void dfs_ldelay(RctCUDAType &rct, int st, int u, int k) {
  rct.ldelay[(st + u) * 4 + k] = rct.cap[(st + u) * 4 + k] * rct.delay[(st + u) * 4 + k];
  for(int i = (u == 0 ? 0 : rct.rct_edgecount[st + u - 1]);
      i < rct.rct_edgecount[st + u];
      ++i) {
    int v = rct.rct_edgeadj[st + i];
    dfs_ldelay(rct, st, v, k);
    rct.ldelay[(st + u) * 4 + k] += rct.ldelay[(st + v) * 4 + k];
  }
}

template <typename RctCUDAType>
__device__ void dfs_response(RctCUDAType &rct, int st, int u, int k) {
  for(int i = (u == 0 ? 0 : rct.rct_edgecount[st + u - 1]);
      i < rct.rct_edgecount[st + u];
      ++i) {
    int v = rct.rct_edgeadj[st + i];
    rct.impulse[(st + v) * 4 + k] = rct.impulse[(st + u) * 4 + k] + rct.pres[st + v] * rct.ldelay[(st + v) * 4 + k];
    dfs_response(rct, st, v, k);
  }
  // beta -> impulse
  float t = rct.delay[(st + u) * 4 + k];
  rct.impulse[(st + u) * 4 + k] = 2 * rct.impulse[(st + u) * 4 + k] - t * t;
}

template <typename RctCUDAType>
__global__ void compute_net_timing_dfs(RctCUDAType rct) {
  unsigned int net_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int el_rf_offset = threadIdx.y;
  if(net_id >= rct.num_nets) return;
  
  int st = rct.rct_nodes_start[net_id];
  rct.delay[st * 4 + el_rf_offset] = 0;
  rct.impulse[st * 4 + el_rf_offset] = 0;
  
  dfs_load(rct, st, 0, el_rf_offset);
  dfs_delay(rct, st, 0, el_rf_offset);
  dfs_ldelay(rct, st, 0, el_rf_offset);
  dfs_response(rct, st, 0, el_rf_offset);
}

#endif

RctCUDA copy_cpu_to_gpu(const RctCUDA& data_cpu) {
  RctCUDA data_gpu;
  data_gpu.num_nets = data_cpu.num_nets;
  data_gpu.total_num_nodes = data_cpu.total_num_nodes;
#define COPY_DATA(arr, sz) allocateCopyCUDA(data_gpu.arr, data_cpu.arr, sz)
  COPY_DATA(rct_pid, data_cpu.total_num_nodes);
  COPY_DATA(pres, data_cpu.total_num_nodes);
  COPY_DATA(cap, data_cpu.total_num_nodes * MAX_SPLIT_TRAN);
  COPY_DATA(rct_nodes_start, data_cpu.num_nets + 1);
#undef COPY_DATA
#define MALLOC_RESULTS(arr) allocateCUDA(data_gpu.arr, data_cpu.total_num_nodes * MAX_SPLIT_TRAN, float)
  MALLOC_RESULTS(load);
  MALLOC_RESULTS(delay);
  MALLOC_RESULTS(ldelay);
  MALLOC_RESULTS(impulse);
#undef MALLOC_RESULTS
  return data_gpu;
}

void copy_gpu_to_cpu(const RctCUDA data_gpu, RctCUDA data_cpu) {
#define COPY_RESULTS(arr) memcpyDeviceHostCUDA(data_cpu.arr, data_gpu.arr, data_gpu.total_num_nodes * MAX_SPLIT_TRAN)
  COPY_RESULTS(load);
  COPY_RESULTS(delay);
  COPY_RESULTS(ldelay);
  COPY_RESULTS(impulse);
#undef COPY_RESULTS
}

void free_gpu(RctCUDA data_gpu) {
#define FREEG(arr) checkCUDA(cudaFree(data_gpu.arr))
  FREEG(rct_nodes_start);
  FREEG(rct_pid);
  FREEG(pres);
  FREEG(cap);
  FREEG(load);
  FREEG(delay);
  FREEG(ldelay);
  FREEG(impulse);
#undef FREEG
}

void rct_compute_cuda(RctCUDA data_cpu) {
  printf("entered rct_compute_cuda();\n");
  _prof::setup_timer("rct_compute_cuda__copy_c2g");
  RctCUDA data_gpu = copy_cpu_to_gpu(data_cpu);
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_compute_cuda__copy_c2g");
  _prof::setup_timer("rct_compute_cuda__compute");
  compute_net_timing<<<(data_cpu.num_nets + chunk - 1) / chunk,
    dim3(chunk, MAX_SPLIT_TRAN)>>>(data_gpu);
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_compute_cuda__compute");
  _prof::setup_timer("rct_compute_cuda__copy_g2c");
  copy_gpu_to_cpu(data_gpu, data_cpu);
  checkCUDA(cudaDeviceSynchronize());
  free_gpu(data_gpu);
  _prof::stop_timer("rct_compute_cuda__copy_g2c");
}

template <int BlockDim>
__global__ void compute_net_bfs(RctEdgeArrayCUDA data_gpu) {
    // one block processes one net
    assert(blockDim.x == BlockDim);
    //const int net_id = blockIdx.x; 
    const int tid = threadIdx.x; 
    __shared__ int nodes_offset; 
    __shared__ int edges_offset; 
    __shared__ int num_edges; 
    __shared__ int num_nodes; 
    __shared__ RctEdgeCUDA* edges; // modified inplace from undirected edges to directed according to BFS order 
    __shared__ int* distances; 
    __shared__ int* sort_counts; 
    __shared__ int* orders; 
    __shared__ int* pid; 
    __shared__ int root; 
    __shared__ int level; 
    __shared__ int count; // count the number of same distances for counting sort algorithm 
                        // I also use count as the change flag 

    for (int net_id = blockIdx.x; net_id < data_gpu.num_nets; net_id += gridDim.x) {

        if (tid == 0) {
            nodes_offset = data_gpu.rct_nodes_start[net_id]; 
            edges_offset = nodes_offset - net_id; 
            num_nodes = data_gpu.rct_nodes_start[net_id + 1] - nodes_offset; 
            num_edges = num_nodes - 1; 
            edges = data_gpu.rct_edges + edges_offset; 
            distances = data_gpu.rct_distances + nodes_offset; 
            sort_counts = data_gpu.rct_sort_counts + nodes_offset; 
            orders = data_gpu.rct_node2bfs_order + nodes_offset; 
            pid = data_gpu.rct_pid + nodes_offset; 
            root = data_gpu.rct_roots[net_id]; 
            level = 0; 
            count = 1; 
        }
        __syncthreads();
#ifdef DEBUG
        assert(nodes_offset + num_nodes <= data_gpu.total_num_nodes);
        assert(edges_offset + num_edges <= data_gpu.total_num_edges);
#endif

        // initialize distance and sort_counts for root 
        for (int u = tid; u < num_nodes; u += blockDim.x) {
            distances[u] = (u == root)? 0 : INT_MAX; 
            sort_counts[u] = (u == 0)? 1 : 0;
        }

        for (int iter = 0; iter < num_nodes; ++iter) {
            if (tid == 0) {
                count = 0; 
            }
            __syncthreads(); 
            // traverse edges, using all blockDim.x threads  
            for (int e = tid; e < num_edges; e += blockDim.x) {
                RctEdgeCUDA& edge = edges[e]; 
#ifdef DEBUG
                assert(edge.s < num_nodes); 
                assert(edge.t < num_nodes);
#endif
                // assume the input edges are undirected 
                // we need to determine the direction  
                int& ds = distances[edge.s]; 
                int& dt = distances[edge.t]; 
                if (ds == level) {
                    if (level + 1 < dt) {
                        dt = level + 1; 
                        atomicAdd(&count, 1);
                    }
                }
                else if (dt == level) {
                    if (level + 1 < ds) {
                        ds = level + 1; 
                        atomicAdd(&count, 1);
                        // swap s -> t to t -> s 
                        int tmp = edge.s; 
                        edge.s = edge.t; 
                        edge.t = tmp; 
                    }
                }
            }
            __syncthreads(); 
            if (count) {
                if (tid == 0) {
#ifdef DEBUG
                    assert(level < num_nodes); 
                    assert(level >= 0);
                    assert(nodes_offset + level < data_gpu.total_num_nodes);
#endif
                    // next level 
                    level += 1; 
                    sort_counts[level] = count; 
                }
            } 
            else
            {
                break; 
            }
            __syncthreads(); 
#ifdef DEBUG
            if (tid < num_edges && count == 0) {
                for (int u = 0; u < num_nodes; ++u) {
                    assert(distances[u] < num_nodes);
                }
            }
#endif
        } 
        __syncthreads(); 

        // argsort nodes according to distances to root 
        block_couting_sort<BlockDim>(distances, sort_counts, orders, nullptr, num_nodes, 0, num_nodes-1, false); 
        

        // construct pid 
        for (int e = tid; e < num_edges; e += blockDim.x) {
            RctEdgeCUDA const& edge = edges[e]; 
#ifdef DEBUG
            assert(edge.s < num_nodes); 
            assert(edge.t < num_nodes); 
#endif
            int order_s = orders[edge.s]; 
            int order_t = orders[edge.t]; 
            pid[order_t] = order_t - order_s;
        }
    }
}

RctEdgeArrayCUDA copy_cpu_to_gpu(const RctEdgeArrayCUDA& data_cpu) {
  RctEdgeArrayCUDA data_gpu;
  data_gpu.num_nets = data_cpu.num_nets;
  data_gpu.total_num_nodes = data_cpu.total_num_nodes; 
  data_gpu.total_num_edges = data_cpu.total_num_edges;
#define COPY_DATA(arr, sz) allocateCopyCUDA(data_gpu.arr, data_cpu.arr, sz)
  COPY_DATA(rct_edges, data_cpu.total_num_edges);
  COPY_DATA(rct_roots, data_cpu.num_nets);
  COPY_DATA(rct_nodes_start, data_cpu.num_nets + 1);
  COPY_DATA(rct_edges_res, data_cpu.total_num_edges); 
  COPY_DATA(rct_nodes_cap, data_cpu.total_num_nodes * MAX_SPLIT_TRAN); 
#undef COPY_DATA
#define MALLOC_RESULTS(arr) allocateCUDA(data_gpu.arr, data_cpu.total_num_nodes, int)
  MALLOC_RESULTS(rct_distances);
  MALLOC_RESULTS(rct_sort_counts); 
  MALLOC_RESULTS(rct_node2bfs_order); 
  MALLOC_RESULTS(rct_pid);
#ifdef RCT_BASELINE
  MALLOC_RESULTS(rct_edgecount);
  MALLOC_RESULTS(rct_edgeadj);
#endif
#undef MALLOC_RESULTS
#define MALLOC_RESULTS(arr, sz) allocateCUDA(data_gpu.arr, sz, float)
  MALLOC_RESULTS(pres, data_cpu.total_num_nodes); 
#undef MALLOC_RESULTS
#define MALLOC_RESULTS(arr) allocateCUDA(data_gpu.arr, data_cpu.total_num_nodes * MAX_SPLIT_TRAN, float)
  MALLOC_RESULTS(cap); 
  MALLOC_RESULTS(load);
  MALLOC_RESULTS(delay);
  MALLOC_RESULTS(ldelay);
  MALLOC_RESULTS(impulse);
#undef MALLOC_RESULTS
  return data_gpu;
}

void copy_gpu_to_cpu(const RctEdgeArrayCUDA data_gpu, RctEdgeArrayCUDA data_cpu) {
#define COPY_RESULTS(arr, sz) memcpyDeviceHostCUDA(data_cpu.arr, data_gpu.arr, sz)
  COPY_RESULTS(rct_edges, data_gpu.total_num_edges); 
  COPY_RESULTS(rct_node2bfs_order, data_gpu.total_num_nodes);
  COPY_RESULTS(rct_pid, data_gpu.total_num_nodes);
#undef COPY_RESULTS
#define COPY_RESULTS(arr) memcpyDeviceHostCUDA(data_cpu.arr, data_gpu.arr, data_gpu.total_num_nodes * MAX_SPLIT_TRAN)
  COPY_RESULTS(load);
  COPY_RESULTS(delay);
  COPY_RESULTS(ldelay);
  COPY_RESULTS(impulse);
#undef COPY_RESULTS
}

void free_gpu(RctEdgeArrayCUDA data_gpu) {
#define FREEG(arr) destroyCUDA(data_gpu.arr)
  FREEG(rct_edges);
  FREEG(rct_roots);
  FREEG(rct_distances);
  FREEG(rct_sort_counts); 
  FREEG(rct_node2bfs_order); 
  FREEG(rct_nodes_start);
  FREEG(rct_pid);
#ifdef RCT_BASELINE
  FREEG(rct_edgecount);
  FREEG(rct_edgeadj);
#endif
  FREEG(rct_edges_res); 
  FREEG(rct_nodes_cap);
  FREEG(pres); 
  FREEG(cap); 
  FREEG(load); 
  FREEG(delay); 
  FREEG(ldelay); 
  FREEG(impulse);
#undef FREEG
}

/// @brief Prepare resistance and capacitance by copying from rct_edges_res and rct_nodes_cap. 
/// Also convert from the original order to BFS order. 
__global__ void prepare_res_cap(RctEdgeArrayCUDA data_gpu) {
    const int net_id = blockIdx.x;
    const int tid = threadIdx.x; 

    __shared__ int nodes_offset; 
    __shared__ int edges_offset; 
    __shared__ int num_edges; 
    __shared__ int num_nodes; 
    __shared__ RctEdgeCUDA* edges; // modified inplace from undirected edges to directed according to BFS order 
    __shared__ int* orders; 
    __shared__ float* edges_res; // edge resistance, in original order 
    __shared__ float* nodes_cap; // node capacitance, in original order 
    __shared__ float* pres; // resistance to sink 
    __shared__ float* cap; 

    if (net_id < data_gpu.num_nets) {
        if (tid == 0) {
            nodes_offset = data_gpu.rct_nodes_start[net_id]; 
            edges_offset = nodes_offset - net_id; 
            num_nodes = data_gpu.rct_nodes_start[net_id + 1] - nodes_offset; 
            num_edges = num_nodes - 1; 
            edges = data_gpu.rct_edges + edges_offset; 
            orders = data_gpu.rct_node2bfs_order + nodes_offset; 
            edges_res = data_gpu.rct_edges_res + edges_offset; 
            nodes_cap = data_gpu.rct_nodes_cap + nodes_offset * MAX_SPLIT_TRAN; 
            pres = data_gpu.pres + nodes_offset; 
            cap = data_gpu.cap + nodes_offset * MAX_SPLIT_TRAN; 
        }
        __syncthreads();

        // set resistance 
        for (int e = tid; e < num_edges; e += blockDim.x) {
            auto const& edge = edges[e]; 
            pres[orders[edge.t]] = edges_res[e];
        }
        // set capacitance 
        for (int u = tid; u < num_nodes; u += blockDim.x) {
            int offset = orders[u] * MAX_SPLIT_TRAN;
            int offset_u = u * MAX_SPLIT_TRAN; 
            #pragma unroll
            for (int i = 0; i < MAX_SPLIT_TRAN; ++i) {
                cap[offset + i] = nodes_cap[offset_u + i]; 
            }
        }
    }
}

#ifdef RCT_BASELINE
/// @brief Simple Adjacency list construction
__global__ void prepare_adjacency_list(RctEdgeArrayCUDA data_gpu) {
  unsigned int net_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(net_id >= data_gpu.num_nets) return;

  int st = data_gpu.rct_nodes_start[net_id], ed = data_gpu.rct_nodes_start[net_id + 1];
  int ste = st - net_id, ede = ed - net_id - 1;
  const int *orders = data_gpu.rct_node2bfs_order + st;
  
  // init edgecount
  for(int i = st; i < ed; ++i) {
    data_gpu.rct_edgecount[i] = 0;
  }
  for(int i = ste; i < ede; ++i) {
    auto const &edge = data_gpu.rct_edges[i];
    ++data_gpu.rct_edgecount[st + orders[edge.s]];
  }
  for(int i = st + 1; i < ed; ++i) {
    data_gpu.rct_edgecount[i] += data_gpu.rct_edgecount[i - 1];
  }

  for(int i = ste; i < ede; ++i) {
    auto const &edge = data_gpu.rct_edges[i];
    int index = --data_gpu.rct_edgecount[st + orders[edge.s]];
    data_gpu.rct_edgeadj[index] = orders[edge.t];
  }
  
  // re-init edgecount again
  for(int i = st; i < ed; ++i) {
    data_gpu.rct_edgecount[i] = 0;
  }
  for(int i = ste; i < ede; ++i) {
    auto const &edge = data_gpu.rct_edges[i];
    ++data_gpu.rct_edgecount[st + orders[edge.s]];
  }
  for(int i = st + 1; i < ed; ++i) {
    data_gpu.rct_edgecount[i] += data_gpu.rct_edgecount[i - 1];
  }
}
#endif

void rct_bfs_and_compute_cuda(RctEdgeArrayCUDA data_cpu) {
  // copy data 
  _prof::setup_timer("rct_bfs_and_compute_cuda__copy_c2g");
  RctEdgeArrayCUDA data_gpu = copy_cpu_to_gpu(data_cpu); 
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_and_compute_cuda__copy_c2g");

  // BFS 
  _prof::setup_timer("rct_bfs_and_compute_cuda__bfs");
  // the threads for one net 
  constexpr int threads = 128; 
  compute_net_bfs<threads><<<data_gpu.num_nets, threads>>>(data_gpu); 
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_and_compute_cuda__bfs");

  // reorder and copy resistance and capacitance
  _prof::setup_timer("rct_bfs_and_compute_cuda__res_cap");
  checkCUDA(cudaMemset(data_gpu.pres, 0, sizeof(float) * data_gpu.total_num_nodes));
  checkCUDA(cudaMemset(data_gpu.cap, 0, sizeof(float) * data_gpu.total_num_nodes * MAX_SPLIT_TRAN));
  prepare_res_cap<<<data_gpu.num_nets, threads>>>(data_gpu); 
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_and_compute_cuda__res_cap");

#ifdef RCT_BASELINE

  // construct adjacency list
  _prof::setup_timer("rct_bfs_and_compute_cuda__BASELINE_adjacency_list");
  prepare_adjacency_list<<<(data_cpu.num_nets + chunk - 1) / chunk, chunk>>>
    (data_gpu);
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_and_compute_cuda__BASELINE_adjacency_list");

  // compute RC delay (BASELINE)
  _prof::setup_timer("rct_bfs_and_compute_cuda__BASELINE_do");
  compute_net_timing_dfs<<<(data_cpu.num_nets + chunk - 1) / chunk,
    dim3(chunk, MAX_SPLIT_TRAN)>>>(data_gpu);
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_and_compute_cuda__BASELINE_do");

#else
  
  // compute RC delay 
  _prof::setup_timer("rct_bfs_and_compute_cuda__compute");
  compute_net_timing<<<(data_cpu.num_nets + chunk - 1) / chunk,
    dim3(chunk, MAX_SPLIT_TRAN)>>>(data_gpu);
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_and_compute_cuda__compute");
  
#endif

  // copy back 
  _prof::setup_timer("rct_bfs_and_compute_cuda__copy_g2c");
  copy_gpu_to_cpu(data_gpu, data_cpu); 
  checkCUDA(cudaDeviceSynchronize());
  free_gpu(data_gpu); 
  _prof::stop_timer("rct_bfs_and_compute_cuda__copy_g2c");
}
