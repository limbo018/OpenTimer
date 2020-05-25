// File rct.cuh
// CUDA Ports

#ifndef OT_CUDA_RCT_CUH_
#define OT_CUDA_RCT_CUH_

struct RctCUDA {
  int num_nets;
  int total_num_nodes;
  int *rct_nodes_start; 
  int *rct_pid;
  float *pres, *cap, *load, *delay, *ldelay, *impulse;
};

struct RctEdgeCUDA {
    int s; 
    int t; 
}; 

struct RctEdgeArrayCUDA {
    int num_nets; ///< input
    int total_num_nodes; ///< input rct_nodes_start[num_nets]
    int total_num_edges; ///< input rct_edges_start[num_nets]
    int* rct_nodes_start = nullptr; ///< input length of (num_nets + 1)
    RctEdgeCUDA* rct_edges = nullptr; ///< input length of rct_edges_start[num_nets]
    int* rct_roots = nullptr; ///< input length of num_nets, record the root 
    float* rct_edges_res = nullptr; ///< input length of total_num_edges, edge resistance 
    float* rct_nodes_cap = nullptr; ///< input length of total_num_nodes* MAX_SPLIT_TRAN, node capacitance, in original order 

    int* rct_distances = nullptr; ///< intermediate, distances to root 
    int* rct_sort_counts = nullptr; ///< intermediate, length of total_num_nodes, count of same distances for counting sort algorithm 
    float* pres = nullptr; ///< intermediate, length of total_num_nodes; in BFS order 
    float* cap = nullptr; ///< intermediate, length of total_num_nodes * MAX_SPLIT_TRAN; in BFS order 

    int* rct_node2bfs_order = nullptr; ///< output, length of total_num_nodes, given node i, should be at location order[i]; same as bfs_reverse_order_map
    int* rct_pid = nullptr; ///< output, length of total_num_nodes; record how far away its parent locates. 
                            ///< For example, the parent of node i is i - pid[i]; the array itself is in BFS order.
#ifdef RCT_BASELINE
    int *rct_edgecount = nullptr; ///< rct edge count
    int *rct_edgeadj = nullptr; ///< rct edgelist
#endif
    float* load = nullptr; ///< output, length of total_num_nodes * MAX_SPLIT_TRAN
    float* delay = nullptr; ///< output, length of total_num_nodes * MAX_SPLIT_TRAN
    float* ldelay = nullptr; ///< output, length of total_num_nodes * MAX_SPLIT_TRAN 
    float* impulse = nullptr; ///< output, length of total_num_nodes * MAX_SPLIT_TRAN
};

void rct_compute_cuda(RctCUDA data_cpu);
void rct_bfs_and_compute_cuda(RctEdgeArrayCUDA data_cpu); 

#endif
