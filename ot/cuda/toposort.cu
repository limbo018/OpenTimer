#include <ot/cuda/toposort.cuh>
#include <ot/cuda/utils.cuh>

__global__ void toposort_advance(int *edgelist_start, int *edgelist,
                                 int *out,
                                 int *frontiers,
                                 int last_size, int *new_size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id >= last_size) return;
  
  int *next_frontiers = frontiers + last_size;
  int u = frontiers[id];
  int edge_st = edgelist_start[u], edge_ed = edgelist_start[u + 1];
  
  for(int i = edge_st; i < edge_ed; ++i) {
    int v = edgelist[i];
    if(1 == atomicAdd(&out[v], -1)) {
      next_frontiers[atomicAdd(new_size, 1)] = v;
    }
  }
}

const int thread_per_block = 128;

void toposort_compute_cuda(
  int n, int num_edges, int first_size,
  int *edgelist_start, int *edgelist, int *out, int *frontiers,
  std::vector<int> &frontiers_ends)
{
  // Step 1: copy to GPU
  int *edgelist_start_gpu, *edgelist_gpu, *out_gpu;
  int *frontiers_gpu;
  int *new_size_gpu;

  allocateCopyCUDA(edgelist_start_gpu, edgelist_start, n + 1);
  allocateCopyCUDA(edgelist_gpu, edgelist, num_edges);
  allocateCopyCUDA(out_gpu, out, n);
  allocateCopyCUDA(frontiers_gpu, frontiers, n);
  allocateCUDA(new_size_gpu, 1, int);
  checkCUDA(cudaDeviceSynchronize());

  // Step 2: do the computation
  int total_size_wo_last = 0, last_size = first_size;
  while(true) {
    toposort_advance<<<(last_size + thread_per_block - 1) / thread_per_block,
      thread_per_block>>>(edgelist_start_gpu, edgelist_gpu, out_gpu,
                          frontiers_gpu + total_size_wo_last,
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
  memcpyDeviceHostCUDA(frontiers, frontiers_gpu, n);
  destroyCUDA(edgelist_start_gpu);
  destroyCUDA(edgelist_gpu);
  destroyCUDA(out_gpu);
  destroyCUDA(frontiers_gpu);
  destroyCUDA(new_size_gpu);
  checkCUDA(cudaDeviceSynchronize());
}
