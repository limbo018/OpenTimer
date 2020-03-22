/**
 * @file   prop.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include <cstdio>
#include <ot/cuda/utils.cuh>
#include <ot/cuda/prop.cuh>
#include <ot/cuda/prop_print.cuh>

#define MAX_SPLIT_TRAN 4 
#define MAX_SPLIT 2
#define MAX_TRAN 2

__device__ float interpolate(float x1, float x2, float d1, float d2, float x) {
  return d1 + (d2 - d1) * (x - x1) / (x2 - x1);
}

__device__ float lut_lookup(int n, int m, float *xs, float *ys, float **data,
                            float x, float y) {
  float (*arr)[m] = data;
  int i = 1, j = 1;
  while(i + 1 < n && xs[i + 1] <= x) ++i;
  while(j + 1 < m && xs[j + 1] <= x) ++j;
  float r1, r2, r;
  r1 = interpolate(ys[j - 1], ys[j], arr[i - 1][j - 1], arr[i - 1][j], y);
  r2 = interpolate(ys[j - 1], ys[j], arr[i][j - 1], arr[i][j], y);
  r = interpolate(xs[i - 1], xs[i], r1, r2, x);
  return r;
}

__device__ float lut_lookup(const FlatTableCUDA &ft, int lutidx, float x, float y) {
  int xsl = ft.xs_st[lutidx], xsr = ft.xs_st[lutidx + 1];
  int ysl = ft.ys_st[lutidx], ysr = ft.ys_st[lutidx + 1];
  int datal = ft.data_st[lutidx];
  return lut_lookup(xsr - xsl, ysr - ysl, ft.xs + xsl, ft.ys + ysl,
                    ft.data_st + datal, x, y);
}

__device__ void write_info(InfoPinCUDA &ipc, float num,
                            int el, int rf, int arcidx) {
  if(ipc.num == num && 0 == atomicExch(ipc.flag, 1)) {
    ipc.fr_el = el;
    ipc.fr_rf = rf;
    ipc.fr_arcidx = arcidx;
  }
}

/*
step = true: update by atomicMax/atomicMin
step = false: write fr_el, fr_rf, fr_arcidx
 */

__global__ void slew_update_netarcs(PropCUDA prop, int level_l, int level_r,
                                    bool step) {
  int arcp = blockIdx.x * blockDim.x + threadIdx.x + level_l;
  int el_rf = threadIdx.y;
  if(arcp >= level_r) return;

  int a = prop.netarcs[arcp].from, b = prop.netarcs[arcp].to;

  if(step) {
    float a_slew = prop.slew_pins4[a * 4 + el_rf];
    float arc_impulse = impulse_netarcs4[arcp * 4 + el_rf];
    float b_slew = sqrtf(a_slew * a_slew + arc_impulse) * (a_slew < 0 ? -1 : 1);
    tmpslew_netarcs4[arcp * 4 + el_rf] = b_slew;
    ((el_rf >> 1) ? atomicMax : atomicMin)
      (&prop.slew_pins4[b * 4 + el_rf].num, b_slew);
  }
  else {
    float b_slew = tmpslew_netarcs4[arcp * 4 + el_rf];
    write_info(prop.slew_pins4[b * 4 + el_rf], b_slew,
               el_rf >> 1, el_rf & 1, prop.netarcs[arcp].idx);
  }
}

__global__ void slew_update_cellarcs(PropCUDA prop, int level_l, int level_r,
                                     bool step) {
  int arcp = blockIdx.x * blockDim.x + threadIdx.x + level_l;
  int el_frf_trf = threadIdx.y;
  if(arcp >= level_r) return;
  int el_frf = el_frf_trf >> 1;
  int el_trf = ((el_frf_trf >> 2) << 1) | (el_frf_trf & 1);

  int a = prop.cellarcs[arcp].from, b = prop.cellarcs[arcp].to;
  int lutidx = prop.lutidx_slew_cellarcs8[arcp * 8 + el_frf_trf];
  if(lutidx < 0) return;

  if(step) {
    float a_slew = prop.slew_pins4[a * 4 + el_frf];
    float b_load = prop.netload_pins4[b * 4 + el_trf];

    float b_slew = lut_lookup(prop.ft, lutidx, a_slew, b_load);
    prop.tmpslew_cellarcs8[arcp * 8 + el_frf_trf] = b_slew;
    ((el_frf_trf >> 2) ? atomicMax : atomicMin)
      (&prop.slew_pins4[b * 4 + el_trf].num, b_slew);
  }
  else {
    float b_slew = prop.tmpslew_cellarcs8[arcp * 8 + el_frf_trf];
    write_info(prop.slew_pins4[b * 4 + el_trf], b_slew,
               el_frf_trf >> 2, el_frf & 1, prop.cellarcs[arcp].idx);
  }
}

__global__ void calc_delay_cellarcs(PropCUDA prop, int level_l, int level_r) {
  int arcp = blockIdx.x * blockDim.x + threadIdx.x + level_l;
  int el_frf_trf = threadIdx.y;
  if(arcp >= level_r) return;

  int lutidx = prop.lutidx_delay_cellarcs8[arcp * 8 + el_frf_trf];
  if(lutidx < 0) return;

}

__global__ void at_update_netarcs(PropCUDA prop, int level_l, int level_r,
                                  bool step) {
  int arcp = blockIdx.x * blockDim.x + threadIdx.x + level_l;
  int el_rf = threadIdx.x;
  if(arcp >= level_r) return;

  int a = prop.netarcs[arcp].from, b = prop.netarcs[arcp].to;
  float a_at = prop.at_pins4[a * 4 + el_rf];
  float b_at = a_at + prop.delay_netarcs4[arcp * 4 + el_rf];

  if(step) {
    ((el_rf >> 1) ? atomicMax : atomicMin)
      (&prop.at_pins4[b * 4 + el_rf], b_at);
  }
  else {
    write_info(prop.at_pins4[b * 4 + el_rf], b_at,
               el_rf >> 1, el_rf & 1, prop.netarcs[arcp].arcidx);
  }
}

__global__ void at_update_cellarcs(PropCUDA prop, int level_l, int level_r,
                                   bool step) {
  int arcp = blockIdx.x * blockDim.x + threadIdx.x + level_l;
  if(arcp >= level_r) return;
  int el_frf_trf = threadIdx.x;
  int el_frf = el_frf_trf >> 1;
  int el_trf = ((el_frf_trf >> 2) << 1) | (el_frf_trf & 1);
  
  int lutidx = prop.lutidx_delay_cellarcs8[arcp * 8 + el_frf_trf];
  if(lutidx < 0) return;

  int a = prop.cellarcs[arcp].from, b = prop.cellarcs[arcp].to;f];

  if(step) {
    float a_slew = prop.slew_pins4[a * 4 + el_frf];
    float b_load = prop.netload_pins4[b * 4 + el_trf];
    float delay = lut_lookup(prop.ft, lutidx, a_slew, b_load);

    prop.delay_cellarcs8[arcp * 8 + el_frf_trf] = delay;
    float a_at = prop.at_pins4[a * 4 + el_frf];
    float b_at = a_at + delay;
  
    ((el_rf >> 1) ? atomicMax : atomicMin)
      (&prop.at_pins4[b * 4 + el_trf], b_at);
  }
  else {
    float a_at = prop.at_pins4[a * 4 + el_frf];
    float b_at = a_at + prop.delay_cellarcs8[arcp * 8 + el_frf_tr];
    
    write_info(prop.at_pins4[b * 4 + el_trf], b_at,
               el_frf_trf >> 2, el_frf & 1, prop.cellarcs[arcp].arcidx);
  }
}

PropCUDA copy2gpu(PropCUDA &cpu) {
  PropCUDA gpu;
#define COPY_DATA(arr, sz) allocateCopyCUDA(gpu.arr, cpu.arr, sz)
  COPY_DATA(netload_pins4, cpu.num_pins * 4);
  COPY_DATA(slew_pins4, cpu.num_pins * 4);
  COPY_DATA(at_pins4, cpu.num_pins * 4);
  
  COPY_DATA(netarcs, cpu.total_num_netarcs);
  COPY_DATA(cellarcs, cpu.total_num_cellarcs);
  COPY_DATA(impulse_netarcs4, cpu.total_num_netarcs * 4);
  COPY_DATA(delay_netarcs4, cpu.total_num_netarcs * 4);
  COPY_DATA(lutidx_slew_cellarcs8, cpu.total_num_cellarcs * 8);
  COPY_DATA(lutidx_delay_cellarcs8, cpu.total_num_cellarcs * 8);

  COPY_DATA(ft.xs, cpu.ft.total_num_xs);
  COPY_DATA(ft.ys, cpu.ft.total_num_ys);
  COPY_DATA(ft.data, cpu.ft.total_num_xsys);
  COPY_DATA(ft.xs_st, cpu.ft.total_num_xs);
  COPY_DATA(ft.ys_st, cpu.ft.total_num_ys);
  COPY_DATA(ft.data_st, cpu.ft.total_num_xsys);
#undef COPY_DATA
  
  allocateCUDA(gpu.tmpslew_netarcs4, cpu.total_num_netarcs * 4, float);
  allocateCUDA(gpu.tmpslew_cellarcs8, cpu.total_num_cellarcs * 8, float);
  allocateCUDA(gpu.delay_cellarcs8, cpu.total_num_cellarcs * 8, float);
}

void freegpu(RctCUDA data_gpu) {
#define FREEG(arr) checkCUDA(cudaFree(data_gpu.arr))
  FREEG(netload_pins4);
  FREEG(slew_pins4);
  FREEG(at_pins4);
  FREEG(netarcs);
  FREEG(cellarcs);
  FREEG(impulse_netarcs4);
  FREEG(delay_netarcs4);
  FREEG(lutidx_slew_cellarcs8);
  FREEG(lutidx_delay_cellarcs8);
  FREEG(ft.xs);
  FREEG(ft.ys);
  FREEG(ft.data);
  FREEG(ft.xs_st);
  FREEG(ft.ys_st);
  FREEG(ft.data_st);
#undef FREEG
}

void copy2cpu(const PropCUDA &gpu, PropCUDA &cpu) {
#define COPY_RESULTS(arr) memcpyDeviceHostCUDA(cpu.arr, gpu.arr, cpu.num_pins * 4);
  COPY_RESULTS(slew_pins4);
  COPY_RESULTS(at_pins4);
#endif
}

void fprop_cuda(PropCUDA &prop_cpu) {
  const int chunk = 64;
  PropCUDA prop_gpu = copy2gpu(prop_cpu);
#define INVOKE(func, l, r, step) \
  func<<<(r - l + 1 + chunk - 1) / chunk, chunk>>>(prop_gpu, l, r, step)
#define INVOKE_NETARCS(func, level, step) \
  INVOKE(func, prop_cpu.netarc_ends[i], prop_cpu.netarc_ends[i + 1], step)
#define INVOKE_CELLARCS(func, level, step) \
  INVOKE(func, prop_cpu.cellarc_ends[i], prop_cpu.cellarc_ends[i + 1], step)
  for(int i = 0; i < prop_cpu.num_levels; ++i) {
    INVOKE_NETARCS(slew_update_netarcs, i, true);
    INVOKE_CELLARCS(slew_update_cellarcs, i, true);
    INVOKE_NETARCS(slew_update_netarcs, i, false);
    INVOKE_CELLARCS(slew_update_cellarcs, i, false);
    INVOKE_NETARCS(at_update_netarcs, i, true);
    INVOKE_CELLARCS(at_update_cellarcs, i, true);
    INVOKE_NETARCS(at_update_netarcs, i, false);
    INVOKE_CELLARCS(at_update_cellarcs, i, false);
  }
#undef INVOKE_CELLARCS
#undef INVOKE_NETARCS
#undef INVOKE
  copy2cpu(prop_gpu, prop_cpu);
  freegpu(prop_gpu);
}
