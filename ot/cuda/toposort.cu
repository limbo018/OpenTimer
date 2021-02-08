//#include <ot/cuda/toposort.cuh>
#include <ot/cuda/prop.cuh>
#include <ot/cuda/utils.cuh>

__global__ void toposort_init(int *out, int *frontiers, int* first_size, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
      if (out[id] == 0) {
          int index = atomicAdd(first_size, 1); 
          frontiers[index] = id; 
      }
  }
}

__global__ void toposort_advance(int *edgelist_start, FlatArc *edgelist,
                                 int *out,
                                 int *frontiers,
                                 int last_size, int *new_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id >= last_size) return;
  
  int *next_frontiers = frontiers + last_size;
  int u = frontiers[id];
  int edge_st = edgelist_start[u], edge_ed = edgelist_start[u + 1];
  
  for(int i = edge_st; i < edge_ed; ++i) {
    int v = edgelist[i].other;
    if(1 == atomicAdd(&out[v], -1)) {
      next_frontiers[atomicAdd(new_size, 1)] = v;
    }
  }
}

const int thread_per_block = 128;

void toposort_compute_cuda(PropCUDA& prop_data_cpu, PropCUDA& prop_data_cuda, 
                           std::vector<int, ot_cuda_allocator<int>> &frontiers_ends)
{
    int n = prop_data_cuda.fanin_graph.num_nodes; 
    int num_edges = prop_data_cuda.fanin_graph.num_edges;
  // Step 1: copy to GPU
  int *size_gpu = nullptr;

  allocateCUDA(size_gpu, 2, int);
  cudaMemset(size_gpu, 0, sizeof(int) * 2); 
  checkCUDA(cudaDeviceSynchronize());

  int* new_size_gpu = size_gpu; 
  int* first_size_gpu = size_gpu + 1; 

  // Step 1.1: initialize first size 
  toposort_init<<<(n + thread_per_block - 1) / thread_per_block, thread_per_block>>>(
          prop_data_cuda.fanout_degrees, prop_data_cuda.frontiers, first_size_gpu, n
          ); 
  int first_size = 0; 
  memcpyDeviceHostCUDA(&first_size, first_size_gpu, 1); 
  checkCUDA(cudaDeviceSynchronize());
  frontiers_ends.push_back(0); 
  frontiers_ends.push_back(first_size); 

  // Step 2: do the computation
  int total_size_wo_last = 0, last_size = first_size;
  while(true) {
    toposort_advance<<<(last_size + thread_per_block - 1) / thread_per_block,
      thread_per_block>>>(prop_data_cuda.fanin_graph.adjacency_list_start, prop_data_cuda.fanin_graph.adjacency_list, prop_data_cuda.fanout_degrees,
                          prop_data_cuda.frontiers + total_size_wo_last,
                          last_size, new_size_gpu);
    int current_size;
    memcpyDeviceHostCUDA(&current_size, new_size_gpu, 1);
    checkCUDA(cudaDeviceSynchronize());
    if(current_size) {
      total_size_wo_last += last_size;
      last_size = current_size;
      frontiers_ends.push_back(total_size_wo_last + current_size);
      
      current_size = 0;
      memcpyHostDeviceCUDA(new_size_gpu, &current_size, 1);
      checkCUDA(cudaDeviceSynchronize());
    }
    else break;
  }

  // Step 3: copy back to CPU
  memcpyDeviceHostCUDA(prop_data_cpu.frontiers, prop_data_cuda.frontiers, n);
  destroyCUDA(size_gpu);
  checkCUDA(cudaDeviceSynchronize());
}
