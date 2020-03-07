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
    int num_nets; 
    int total_num_nodes; ///< rct_nodes_start[num_nets]
    int total_num_edges; ///< rct_edges_start[num_nets]
    RctEdgeCUDA* rct_edges; ///< length of rct_edges_start[num_nets]
    int* rct_edges_start; ///< length of (num_nets + 1)

    int* rct_roots; ///< length of num_nets, record the root 

    int* rct_distances; ///< distances to root 
    int* rct_sort_counts; ///< length of total_num_nodes, count of same distances for counting sort algorithm 
    int* rct_orders; ///< length of total_num_nodes, given node i, should be at location order[i]; same as bfs_reverse_order_map
    int* rct_pid; ///< length of total_num_nodes; record how far away its parent locates. 
                ///< For example, the parent of node i is i - pid[i]; the array itself is in BFS order. 
    int* rct_nodes_start; ///< length of (num_nets + 1)
};

void rct_compute_cuda(RctCUDA data_cpu);
void rct_bfs_cuda(RctEdgeArrayCUDA data_cpu); 
