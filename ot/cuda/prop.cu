/**
 * @file   prop.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include <cstdio>
#include <ot/cuda/prop.cuh>
#include "ot/cuda/utils.cuh"

enum Split {
  MIN = 0,
  MAX = 1
};

enum Tran {
  RISE = 0,
  FALL = 1
};

#define MAX_SPLIT_TRAN 4 
#define MAX_SPLIT 2
#define MAX_TRAN 2

template <typename T>
void print(const T* data, int n, const char* msg) {
    printf("%s[%d] = {", msg, n);
    for (int i = 0; i < n; ++i) {
        printf("%g ", (double)data[i]);
    }
    printf("}\n");
}

void print(FlatTableCUDA const& ft, const char* msg) {
    printf("%s[%u] = {\n", msg, ft.num_tables);
    for (int i = 0; i < ft.num_tables; ++i) {
        printf("slew[%u][%u x %u]\n", i, ft.xs_st[i + 1] - ft.xs_st[i], ft.ys_st[i + 1] - ft.ys_st[i]);
        printf("ft.xs: ");
        for (int j = ft.xs_st[i]; j < ft.xs_st[i + 1]; ++j) {
            printf("%g ", ft.xs[j]); 
        }
        printf("\n");
        printf("ft.ys: ");
        for (int j = ft.ys_st[i]; j < ft.ys_st[i + 1]; ++j) {
            printf("%g ", ft.ys[j]); 
        }
        printf("\n");
        printf("slew table\n");
        for (int j = ft.data_st[i]; j < ft.data_st[i + 1]; ++j) {
            printf("%g ", ft.data[j]); 
            if ((j % (ft.ys_st[i + 1] - ft.ys_st[i])) == (ft.ys_st[i + 1] - ft.ys_st[i] - 1)) {
                printf("\n");
            }
        }
        printf("\n");
    }
    printf("}\n");
}

void PropCUDA::destroy_device() {
    fanin_graph.destroy_device();
    slew_ft.destroy_device(); 
    delay_ft.destroy_device();

    destroyCUDA(net_arc_delays); 
    destroyCUDA(net_arc_impulses); 
    destroyCUDA(pin_loads); 
    destroyCUDA(arc2ftid); 
    destroyCUDA(frontiers); 
    destroyCUDA(frontiers_ends); 
    destroyCUDA(pin_slews); 
    destroyCUDA(pin_ats); 
    destroyCUDA(cell_arc_delays); 
}

void PropCUDA::copy2device(PropCUDA& rhs) const {
    fanin_graph.copy2device(rhs.fanin_graph); 
    slew_ft.copy2device(rhs.slew_ft); 
    delay_ft.copy2device(rhs.delay_ft); 

    rhs.num_levels = num_levels; 
    rhs.num_pins = num_pins; 
    rhs.num_arcs = num_arcs; 
    allocateCopyCUDA(rhs.net_arc_delays, net_arc_delays, num_arcs * MAX_SPLIT_TRAN);
    allocateCopyCUDA(rhs.net_arc_impulses, net_arc_impulses, num_arcs * MAX_SPLIT_TRAN);
    allocateCopyCUDA(rhs.pin_loads, pin_loads, num_arcs * MAX_SPLIT_TRAN);
    allocateCopyCUDA(rhs.arc2ftid, arc2ftid, num_arcs * MAX_SPLIT_TRAN * MAX_TRAN);
    allocateCopyCUDA(rhs.frontiers, frontiers, num_pins);
    allocateCopyCUDA(rhs.frontiers_ends, frontiers_ends, num_levels + 1);
    allocateCopyCUDA(rhs.pin_slews, pin_slews, num_pins * MAX_SPLIT_TRAN); 
    allocateCopyCUDA(rhs.pin_ats, pin_ats, num_pins * MAX_SPLIT_TRAN); 
    allocateCopyCUDA(rhs.cell_arc_delays, cell_arc_delays, num_arcs * MAX_SPLIT_TRAN * MAX_TRAN);
}

__device__ float interpolate(float x1, float x2, float d1, float d2, float x) {
    if (x1 == x2) {
        return d1; 
    }
    else {
        return d1 + (d2 - d1) * (x - x1) / (x2 - x1);
    }
}

__device__ float lut_lookup(int n, int m, 
        const float *xs, const float *ys, const float *data,
        float x, float y) {
#define AT_DATA_2D(i, j) data[(i) * m + (j)]

  int i_1 = 0; 
  int i = min(1, n - 1);
  while(i + 1 < n && xs[i] <= x) {
      i_1 = i++;
  }
  int j_1 = 0; 
  int j = min(1, m - 1);
  while(j + 1 < m && ys[j] <= y) {
      j_1 = j++;
  }
  float r1 = interpolate(ys[j_1], ys[j], AT_DATA_2D(i_1, j_1), AT_DATA_2D(i_1, j), y);
  float r2 = interpolate(ys[j_1], ys[j], AT_DATA_2D(i, j_1), AT_DATA_2D(i, j), y);
  float r = interpolate(xs[i_1], xs[i], r1, r2, x);

#undef AT_DATA_2D
  return r;
}

__device__ float lut_lookup(const FlatTableCUDA &ft, int lutidx, float x, float y) {
  int xsl = ft.xs_st[lutidx], xsr = ft.xs_st[lutidx + 1];
  int ysl = ft.ys_st[lutidx], ysr = ft.ys_st[lutidx + 1];
  int datal = ft.data_st[lutidx];
  return lut_lookup(xsr - xsl, ysr - ysl, 
          ft.xs + xsl, ft.ys + ysl, ft.data + datal, 
          x, y);
}

__device__ int ftid(PropCUDA const& prop, int arc_id, int el, int irf, int orf) {
    int id = arc_id * (MAX_SPLIT_TRAN * MAX_TRAN) + el * (MAX_TRAN * MAX_TRAN)
        + irf * MAX_TRAN + orf; 
    return prop.arc2ftid[id]; 
}

__device__ void update_slew_or_at(PinInfoCUDA& to_slew, FlatArc const& arc, int el, int rf, float slew) {
    if (el == Split::MAX) {
        if (to_slew.value < slew) {
            to_slew.value = slew; 
            to_slew.from_el = el;
            to_slew.from_rf = rf; 
            to_slew.from_arcidx = (arc.idx >> 1);
        }
    } 
    else {
        if (to_slew.value > slew) {
            to_slew.value = slew; 
            to_slew.from_el = el;
            to_slew.from_rf = rf; 
            to_slew.from_arcidx = (arc.idx >> 1);
        }
    }
}

__global__ void fprop_slew_cuda(PropCUDA prop, int level_l, int level_r) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = tid + level_l; 
    if (offset < level_r) {
        int el_trf = threadIdx.y; // trf is short for rf for to 
        int el = (el_trf >> 1);
        int trf = (el_trf & 1); 
        int pin_to = prop.frontiers[offset];
        int pin_to_offset = pin_to * MAX_SPLIT_TRAN + el_trf;
        auto to_load = prop.pin_loads[pin_to_offset];
        auto& to_slew = prop.pin_slews[pin_to_offset];
        auto& to_at = prop.pin_ats[pin_to_offset];
        int bgn = prop.fanin_graph.adjacency_list_start[pin_to]; 
        int end = prop.fanin_graph.adjacency_list_start[pin_to + 1]; 
        for (int e = bgn; e < end; ++e) {
            FlatArc const& arc = prop.fanin_graph.adjacency_list[e]; 
            int arc_idx = (arc.idx >> 1);
            int pin_from = arc.other;
            auto arc_type = (arc.idx & 1); 
            if (arc_type) { // cell arc 
                int pin_from_offset = pin_from * MAX_SPLIT_TRAN + el * MAX_TRAN;
                int arc_offset = arc_idx * MAX_SPLIT_TRAN * MAX_TRAN + el * MAX_TRAN * MAX_TRAN + trf; 
                for (int frf = 0; frf < MAX_TRAN; ++frf) {
                    auto const& from_slew = prop.pin_slews[pin_from_offset + frf]; 
                    auto const& from_at = prop.pin_ats[pin_from_offset + frf]; 
                    auto& arc_delay = prop.cell_arc_delays[arc_offset + frf * MAX_TRAN];
                    int lutidx = ftid(prop, arc_idx, el, frf, trf); 
                    if (lutidx < prop.slew_ft.num_tables) {
                        float cur_to_slew = lut_lookup(prop.slew_ft, lutidx, from_slew.value, to_load);
                        float delay = lut_lookup(prop.delay_ft, lutidx, from_slew.value, to_load);
                        float cur_to_at = from_at.value + delay;
                        arc_delay = delay; 
                        update_slew_or_at(to_slew, arc, el, trf, cur_to_slew);
                        update_slew_or_at(to_at, arc, el, trf, cur_to_at);
                    }
                }
            }
            else { // net arc 
                int pin_from_offset = pin_from * MAX_SPLIT_TRAN + el_trf;
                auto const& from_slew = prop.pin_slews[pin_from_offset];
                auto const& from_at = prop.pin_ats[pin_from_offset];
                int arc_offset = arc_idx * MAX_SPLIT_TRAN + el_trf;
                float arc_impulse = prop.net_arc_impulses[arc_offset];
                float arc_delay = prop.net_arc_delays[arc_offset];
                int sign = (from_slew.value < 0? -1 : 1);
                float cur_to_slew = sqrtf(from_slew.value * from_slew.value + arc_impulse) * sign; 
                float cur_to_at = from_at.value + arc_delay; 
                update_slew_or_at(to_slew, arc, el, trf, cur_to_slew);
                update_slew_or_at(to_at, arc, el, trf, cur_to_at);
            }
        }
    }
}

__global__ void print(PinInfoCUDA const* pin_slews, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < n; ++i) {
            for (int el = 0; el < MAX_SPLIT; ++el) {
                for (int rf = 0; rf < MAX_TRAN; ++rf) {
                    int idx = i * MAX_SPLIT_TRAN + el * MAX_TRAN + rf;
                    printf("data_cuda.pin[%u][%d][%d] slew %.6f\n", i, el, rf, pin_slews[idx].value);
                }
            }
        }
    }
}

void prop_cuda(PropCUDA& data_cpu) {
    //print(prop_data_cpu.arcs, prop_data_cpu.num_arcs); 
    //print(prop_data_cpu.net_arc_delays, prop_data_cpu.num_arcs, "net_arc_delays"); 
    //print(prop_data_cpu.net_arc_impulses, prop_data_cpu.num_arcs, "net_arc_impulses"); 
    //print(prop_data_cpu.pin_loads, prop_data_cpu.num_pins, "pin_loads"); 
    //print(prop_data_cpu.arc2ftid, prop_data_cpu.num_arcs, "arc2ftid"); 
    //print(prop_data_cpu.slew_ft, "slew_ft");
    //print(prop_data_cpu.delay_ft, "delay_ft");
    // kernel propagation  
    PropCUDA data_cuda;
    data_cpu.copy2device(data_cuda);

    constexpr int chunk = 32; 
    for (int i = data_cpu.num_levels - 1; i >= 0; --i) {
        int l = data_cpu.frontiers_ends[i]; 
        int r = data_cpu.frontiers_ends[i + 1];
        //printf("frontiers[%d]: ", data_cpu.num_levels - 1 - i);
        //for (int k = l; k < r; ++k) {
        //    printf("%d, ", data_cpu.frontiers[k]);
        //}
        //printf("\n");
        int block_dim = (r - l + chunk - 1) / chunk;
        fprop_slew_cuda<<<block_dim, {chunk, MAX_SPLIT_TRAN}>>>(data_cuda, l, r); 
        //fprop_at_cuda<<<block_dim, {chunk, MAX_SPLIT_TRAN}>>>(data_cuda, l, r); 
    }

    //print<<<1, 1>>>(data_cuda.pin_slews, data_cuda.num_pins); 

    checkCUDA(cudaMemcpy(data_cpu.pin_slews, data_cuda.pin_slews, sizeof(PinInfoCUDA) * data_cpu.num_pins * MAX_SPLIT_TRAN, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(data_cpu.pin_ats, data_cuda.pin_ats, sizeof(PinInfoCUDA) * data_cpu.num_pins * MAX_SPLIT_TRAN, cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(data_cpu.cell_arc_delays, data_cuda.cell_arc_delays, sizeof(float) * data_cpu.num_arcs * MAX_SPLIT_TRAN * MAX_TRAN, cudaMemcpyDeviceToHost));
    checkCUDA(cudaDeviceSynchronize());

    //for (int i = 0; i < data_cpu.num_pins; ++i) {
    //    for (int el = 0; el < MAX_SPLIT; ++el) {
    //        for (int rf = 0; rf < MAX_TRAN; ++rf) {
    //            int idx = i * MAX_SPLIT_TRAN + el * MAX_TRAN + rf;
    //            printf("data_cpu.pin[%u][%d][%d] slew %.6f\n", i, el, rf, data_cpu.pin_slews[idx].value);
    //        }
    //    }
    //}
    data_cuda.destroy_device(); 
}
