/**
 * @file   flat_table.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "ot/cuda/flat_table.cuh"
#include "ot/cuda/utils.cuh"

void FlatTableCUDA::destroy_device() {
  destroyCUDA(xs); 
  destroyCUDA(ys); 
  destroyCUDA(data); 
  destroyCUDA(xs_st); 
  destroyCUDA(ys_st); 
  destroyCUDA(data_st); 
}

void FlatTableCUDA::copy2device(FlatTableCUDA& rhs) const {
    rhs.num_tables = num_tables; 
    rhs.total_num_xs = total_num_xs; 
    rhs.total_num_ys = total_num_ys; 
    rhs.total_num_data = total_num_data;

    allocateCopyCUDA(rhs.xs, xs, total_num_xs); 
    allocateCopyCUDA(rhs.ys, ys, total_num_ys);
    allocateCopyCUDA(rhs.data, data, total_num_data); 
    allocateCopyCUDA(rhs.xs_st, xs_st, num_tables + 1); 
    allocateCopyCUDA(rhs.ys_st, ys_st, num_tables + 1); 
    allocateCopyCUDA(rhs.data_st, data_st, num_tables + 1); 
}

