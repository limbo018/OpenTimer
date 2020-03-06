// File rct.cu

#include <ot/cuda/rct.cuh>
#include <ot/cuda/utils.cuh>
#include <ot/timer/_prof.hpp>

//const MAX_SPLIT_TRAN = 4;
const int chunk = 64;

__global__ void compute_net_timing(RctCUDA rct) {
  unsigned int net_id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int el_rf_offset = threadIdx.y;
  if(net_id >= rct.num_nets) return;
  
  int st = rct.arr_starts[net_id], ed = rct.arr_starts[net_id + 1];
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
    int prev = i - rct.pid[j] * 4;
    rct.load[prev] += rct.load[i];
  }

  // update delay from upstream to downstream
  for(int i = st4 + 4, j = st + 1; i < ed4; i += 4, ++j) {
    int prev = i - rct.pid[j] * 4;
    rct.delay[i] += rct.delay[prev] + rct.load[i] * rct.pres[j];
  }

  // update cap*delay from downstream to upstream
  for(int i = rst4, j = ed - 1; i > red4; i -= 4, --j) {
    int prev = i - rct.pid[j] * 4;
    rct.ldelay[i] += rct.cap[i] * rct.delay[i];
    rct.ldelay[prev] += rct.ldelay[i];
  }
  rct.ldelay[st4] += rct.cap[st4] * rct.delay[st4];

  // update beta from upstream to downstream
  for(int i = st4 + 4, j = st + 1; i < ed4; i += 4, ++j) {
    int prev = i - rct.pid[j] * 4;
    rct.impulse[i] += rct.impulse[prev] + rct.ldelay[i] * rct.pres[j];
  }

  // beta -> impulse
  for(int i = st4; i < ed4; i += 4) {
    float t = rct.delay[i];
    rct.impulse[i] = 2 * rct.impulse[i] - t * t;
  }
}

RctCUDA copy_cpu_to_gpu(const RctCUDA data_cpu) {
  RctCUDA data_gpu;
  data_gpu.num_nets = data_cpu.num_nets;
  data_gpu.total_num_nodes = data_cpu.total_num_nodes;
#define COPY_DATA(arr, sz) allocateCopyCUDA(data_gpu.arr, data_cpu.arr, sz)
  COPY_DATA(pid, data_cpu.total_num_nodes);
  COPY_DATA(pres, data_cpu.total_num_nodes);
  COPY_DATA(cap, data_cpu.total_num_nodes * 4);
  COPY_DATA(arr_starts, data_cpu.num_nets + 1);
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
  FREEG(arr_starts);
  FREEG(pid);
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

__global__ void compute_net_bfs(RctEdgeArrayCUDA data_gpu) {
    // one block may process multiple nets 
    int net_id = blockIdx.x; 
    int tid = threadIdx.x; 
    __shared__ int edges_offset; 
    __shared__ int nodes_offset; 
    __shared__ int num_edges; 
    __shared__ int num_nodes; 
    __shared__ RctEdgeCUDA* edges; 
    //__shared__ int* parents; 
    __shared__ int* distances; 
    __shared__ int root; 
    __shared__ int level; 
    __shared__ int change_flag; 

    if (net_id < data_gpu.num_nets) {

        if (tid == 0) {
            edges_offset = data_gpu.rct_edges_start[net_id]; 
            nodes_offset = data_gpu.rct_nodes_start[net_id]; 
            num_edges = data_gpu.rct_edges_start[net_id + 1] - edges_offset; 
            num_nodes = data_gpu.rct_nodes_start[net_id + 1] - nodes_offset; 
            edges = data_gpu.rct_edges + edges_offset; 
            //parents = data_gpu.rct_parents + nodes_offset; 
            distances = data_gpu.rct_distances + nodes_offset; 
            root = data_gpu.rct_roots[net_id]; 
            level = 0; 
            change_flag = true; 
        }
        __syncthreads();

        // initialize distance by traversing nodes, using all blockDim.x threads 
        for (int u = tid; u < num_nodes; u += blockDim.x) {
            int& du = distances[u]; 
            // initialize 
            du = (u == root)? 0 : INT_MAX; 
        }
        __syncthreads(); 

        while (change_flag) {
            if (tid == 0) {
                change_flag = false; 
            }
            __syncthreads(); 
            // traverse edges, using all blockDim.x threads  
            for (int e = tid; e < num_edges; e += blockDim.x) {
                RctEdgeCUDA& edge = edges[e]; 
                int& ds = distances[edge.s]; 
                if (ds == level) {
                    int& dt = distances[edge.t]; 
                    if (level + 1 < dt) {
                        dt = level + 1; 
                        //parents[edge.t] = edge.s; 
                        atomicExch(&change_flag, true); 
                    }
                }
            }
            if (tid == 0) {
                level += 1; 
            }
            __syncthreads(); 
        }
    }
}

RctEdgeArrayCUDA copy_cpu_to_gpu(const RctEdgeArrayCUDA data_cpu) {
  RctEdgeArrayCUDA data_gpu;
  data_gpu.num_nets = data_cpu.num_nets;
  data_gpu.total_num_nodes = data_cpu.total_num_nodes; 
  data_gpu.total_num_edges = data_cpu.total_num_edges;
#define COPY_DATA(arr, sz) allocateCopyCUDA(data_gpu.arr, data_cpu.arr, sz)
  COPY_DATA(rct_edges, data_cpu.total_num_nodes);
  COPY_DATA(rct_edges_start, data_cpu.num_nets + 1);
  COPY_DATA(rct_roots, data_cpu.num_nets + 1);
  COPY_DATA(rct_nodes_start, data_cpu.num_nets + 1);
#undef COPY_DATA
#define MALLOC_RESULTS(arr) allocateCUDA(data_gpu.arr, data_cpu.total_num_nodes, int)
  //MALLOC_RESULTS(rct_parents);
  MALLOC_RESULTS(rct_distances);
#undef MALLOC_RESULTS
  return data_gpu;
}

void copy_gpu_to_cpu(const RctEdgeArrayCUDA data_gpu, RctEdgeArrayCUDA data_cpu) {
#define COPY_RESULTS(arr) memcpyDeviceHostCUDA(data_cpu.arr, data_gpu.arr, data_gpu.total_num_nodes)
  COPY_RESULTS(rct_distances);
#undef COPY_RESULTS
}

void free_gpu(RctEdgeArrayCUDA data_gpu) {
#define FREEG(arr) checkCUDA(cudaFree(data_gpu.arr))
  FREEG(rct_edges);
  FREEG(rct_edges_start);
  FREEG(rct_roots);
  //FREEG(rct_parents);
  FREEG(rct_distances);
  FREEG(rct_nodes_start);
#undef FREEG
}

void rct_bfs_cuda(RctEdgeArrayCUDA data_cpu) {
  RctEdgeArrayCUDA data_gpu = copy_cpu_to_gpu(data_cpu); 
  checkCUDA(cudaDeviceSynchronize());
  // the threads for one net 
  int threads = 256; 
  compute_net_bfs<<<(data_gpu.num_nets + threads - 1) / threads, threads>>>(data_gpu); 
  checkCUDA(cudaDeviceSynchronize());
  copy_gpu_to_cpu(data_gpu, data_cpu); 
  checkCUDA(cudaDeviceSynchronize());
  free_gpu(data_gpu); 
}
