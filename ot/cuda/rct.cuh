// File rct.cuh
// CUDA Ports

struct RctCUDA {
  int num_nets;
  int total_num_nodes;
  int *arr_starts, *pid;
  float *pres, *cap, *load, *delay, *ldelay, *impulse;
};

extern "C"
void rct_compute_cuda(RctCUDA data_cpu);
