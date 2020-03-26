/**
 * @file   graph.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include "ot/cuda/graph.cuh"
#include "ot/cuda/utils.cuh"

void FlatArcGraphCUDA::destroy_device() {
    destroyCUDA(adjacency_list);
    destroyCUDA(adjacency_list_start);
}

void FlatArcGraphCUDA::copy2device(FlatArcGraphCUDA& rhs) const {
    rhs.num_nodes = num_nodes; 
    rhs.num_edges = num_edges;
    allocateCopyCUDA(rhs.adjacency_list, adjacency_list, adjacency_list_start[num_nodes]); 
    allocateCopyCUDA(rhs.adjacency_list_start, adjacency_list_start, num_nodes + 1); 
}
