// File rct.cu

#include <ot/cuda/rct.cuh>
#include <ot/cuda/sort.cuh>
#include <ot/cuda/utils.cuh>
#include <ot/timer/_prof.hpp>

//const MAX_SPLIT_TRAN = 4;
const int chunk = 64;

__global__ void compute_net_timing(RctCUDA rct) {
  unsigned int net_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int el_rf_offset = threadIdx.y;
  if(net_id >= rct.num_nets) return;
  
  int st = rct.rct_nodes_start[net_id], ed = rct.rct_nodes_start[net_id + 1];
  int st4 = st * 4 + el_rf_offset, ed4 = ed * 4 + el_rf_offset;
  int rst4 = ed4 - 4, red4 = st4;   // red4 = st4, jumping over the root

  // update load from cap
  // and init array
  for(int i = st4; i < ed4; i += 4) {
    rct.load[i] = rct.cap[i];
    rct.delay[i] = rct.ldelay[i] = rct.impulse[i] = 0;
  }

  // update load from downstream to upstream
  for(int i = rst4, j = ed - 1; i > red4; i -= 4, --j) {
    int prev = i - rct.rct_pid[j] * 4;
    rct.load[prev] += rct.load[i];
  }

  // update delay from upstream to downstream
  for(int i = st4 + 4, j = st + 1; i < ed4; i += 4, ++j) {
    int prev = i - rct.rct_pid[j] * 4;
    rct.delay[i] += rct.delay[prev] + rct.load[i] * rct.pres[j];
  }

  // update cap*delay from downstream to upstream
  for(int i = rst4, j = ed - 1; i > red4; i -= 4, --j) {
    int prev = i - rct.rct_pid[j] * 4;
    rct.ldelay[i] += rct.cap[i] * rct.delay[i];
    rct.ldelay[prev] += rct.ldelay[i];
  }
  rct.ldelay[st4] += rct.cap[st4] * rct.delay[st4];

  // update beta from upstream to downstream
  for(int i = st4 + 4, j = st + 1; i < ed4; i += 4, ++j) {
    int prev = i - rct.rct_pid[j] * 4;
    rct.impulse[i] += rct.impulse[prev] + rct.ldelay[i] * rct.pres[j];
  }

  // beta -> impulse
  for(int i = st4; i < ed4; i += 4) {
    float t = rct.delay[i];
    rct.impulse[i] = 2 * rct.impulse[i] - t * t;
  }
}

RctCUDA copy_cpu_to_gpu(const RctCUDA& data_cpu) {
  RctCUDA data_gpu;
  data_gpu.num_nets = data_cpu.num_nets;
  data_gpu.total_num_nodes = data_cpu.total_num_nodes;
#define COPY_DATA(arr, sz) allocateCopyCUDA(data_gpu.arr, data_cpu.arr, sz)
  COPY_DATA(rct_pid, data_cpu.total_num_nodes);
  COPY_DATA(pres, data_cpu.total_num_nodes);
  COPY_DATA(cap, data_cpu.total_num_nodes * 4);
  COPY_DATA(rct_nodes_start, data_cpu.num_nets + 1);
#undef COPY_DATA
#define MALLOC_RESULTS(arr) allocateCUDA(data_gpu.arr, data_cpu.total_num_nodes * 4, float)
  MALLOC_RESULTS(load);
  MALLOC_RESULTS(delay);
  MALLOC_RESULTS(ldelay);
  MALLOC_RESULTS(impulse);
#undef MALLOC_RESULTS
  return data_gpu;
}

void copy_gpu_to_cpu(const RctCUDA data_gpu, RctCUDA data_cpu) {
#define COPY_RESULTS(arr) memcpyDeviceHostCUDA(data_cpu.arr, data_gpu.arr, data_gpu.total_num_nodes * 4)
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
    dim3(chunk, 4)>>>(data_gpu);
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
#undef COPY_DATA
#define MALLOC_RESULTS(arr) allocateCUDA(data_gpu.arr, data_cpu.total_num_nodes, int)
  MALLOC_RESULTS(rct_distances);
  MALLOC_RESULTS(rct_sort_counts); 
  MALLOC_RESULTS(rct_node2bfs_order); 
  MALLOC_RESULTS(rct_pid); 
#undef MALLOC_RESULTS
  return data_gpu;
}

void copy_gpu_to_cpu(const RctEdgeArrayCUDA data_gpu, RctEdgeArrayCUDA data_cpu) {
#define COPY_RESULTS(arr, sz) memcpyDeviceHostCUDA(data_cpu.arr, data_gpu.arr, sz)
  COPY_RESULTS(rct_edges, data_gpu.total_num_edges); 
  COPY_RESULTS(rct_node2bfs_order, data_gpu.total_num_nodes);
  COPY_RESULTS(rct_pid, data_gpu.total_num_nodes);
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
#undef FREEG
}

void rct_bfs_cuda(RctEdgeArrayCUDA data_cpu) {
  _prof::setup_timer("rct_bfs_cuda__copy_c2g");
  RctEdgeArrayCUDA data_gpu = copy_cpu_to_gpu(data_cpu); 
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_cuda__copy_c2g");
  _prof::setup_timer("rct_bfs_cuda__compute");
  // the threads for one net 
  constexpr int threads = 128; 
  compute_net_bfs<threads><<<data_gpu.num_nets, threads>>>(data_gpu); 
  checkCUDA(cudaDeviceSynchronize());
  _prof::stop_timer("rct_bfs_cuda__compute");
  _prof::setup_timer("rct_bfs_cuda__copy_g2c");
  copy_gpu_to_cpu(data_gpu, data_cpu); 
  checkCUDA(cudaDeviceSynchronize());
  free_gpu(data_gpu); 
  _prof::stop_timer("rct_bfs_cuda__copy_g2c");
}
