// File rct.cuh
// CUDA Ports

struct RctCUDA {
  int num_nets;
  int total_num_nodes;
  int *rct_nodes_start; 
  int *pid;
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
    int* rct_nodes_start; ///< input length of (num_nets + 1)
    RctEdgeCUDA* rct_edges; ///< input length of rct_edges_start[num_nets]
    int* rct_roots; ///< input length of num_nets, record the root 

    int* rct_distances; ///< intermediate, distances to root 
    int* rct_sort_counts; ///< intermediate, length of total_num_nodes, count of same distances for counting sort algorithm 

    int* rct_node2bfs_order; ///< output, length of total_num_nodes, given node i, should be at location order[i]; same as bfs_reverse_order_map
    int* rct_pid; ///< output, length of total_num_nodes; record how far away its parent locates. 
                ///< For example, the parent of node i is i - pid[i]; the array itself is in BFS order. 
};

void rct_compute_cuda(RctCUDA data_cpu);
void rct_bfs_cuda(RctEdgeArrayCUDA data_cpu); 
