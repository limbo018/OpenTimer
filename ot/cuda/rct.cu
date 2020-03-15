// File rct.cu

#include <ot/cuda/rct.cuh>
#include <ot/cuda/utils.cuh>
#include <ot/timer/_prof.hpp>

//const MAX_SPLIT_TRAN = 4;
const int chunk = 64;

__global__ static void compute_net_timing(RctCUDA rct) {
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

static RctCUDA copy_cpu_to_gpu(const RctCUDA data_cpu) {
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

static void copy_gpu_to_cpu(const RctCUDA data_gpu, RctCUDA data_cpu) {
#define COPY_RESULTS(arr) memcpyDeviceHostCUDA(data_cpu.arr, data_gpu.arr, data_gpu.total_num_nodes * 4)
  COPY_RESULTS(load);
  COPY_RESULTS(delay);
  COPY_RESULTS(ldelay);
  COPY_RESULTS(impulse);
#undef COPY_RESULTS
}

static void free_gpu(RctCUDA data_gpu) {
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

extern "C"
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
