/**
 * @file   flat_table.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "ot/cuda/flat_table.cuh"
#include "ot/cuda/utils.cuh"

extern cudaStream_t streams[]; 

void FlatTableCUDA::destroy_device() {
  destroyCUDA(xs); 
  destroyCUDA(ys); 
  destroyCUDA(data); 
  destroyCUDA(xs_st); 
  destroyCUDA(ys_st); 
  destroyCUDA(data_st); 
}

void FlatTableCUDA::copy2device(FlatTableCUDA& rhs, int stream_id) const {
    rhs.num_tables = num_tables; 
    rhs.total_num_xs = total_num_xs; 
    rhs.total_num_ys = total_num_ys; 
    rhs.total_num_data = total_num_data;

    allocateCopyCUDAAsync(rhs.xs, xs, total_num_xs, streams[stream_id]); 
    allocateCopyCUDAAsync(rhs.ys, ys, total_num_ys, streams[stream_id]);
    allocateCopyCUDAAsync(rhs.data, data, total_num_data, streams[stream_id]); 
    allocateCopyCUDAAsync(rhs.xs_st, xs_st, num_tables + 1, streams[stream_id]); 
    allocateCopyCUDAAsync(rhs.ys_st, ys_st, num_tables + 1, streams[stream_id]); 
    allocateCopyCUDAAsync(rhs.data_st, data_st, num_tables + 1, streams[stream_id]); 
}

