/**
 * @file   graph.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "ot/cuda/graph.cuh"
#include "ot/cuda/utils.cuh"

extern cudaStream_t streams[]; 

void FlatArcGraphCUDA::destroy_device() {
    destroyCUDA(adjacency_list);
    destroyCUDA(adjacency_list_start);
}

void FlatArcGraphCUDA::copy2device(FlatArcGraphCUDA& rhs, int stream_id) const {
    rhs.num_nodes = num_nodes; 
    rhs.num_edges = num_edges;
    allocateCopyCUDAAsync(rhs.adjacency_list, adjacency_list, adjacency_list_start[num_nodes], streams[stream_id]); 
    allocateCopyCUDAAsync(rhs.adjacency_list_start, adjacency_list_start, num_nodes + 1, streams[stream_id]); 
}
