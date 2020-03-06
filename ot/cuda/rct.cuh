// File rct.cuh
// CUDA Ports

struct RctCUDA {
  int num_nets;
  int total_num_nodes;
  int *arr_starts, *pid;
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

    //int* rct_parents; ///< parents of each node in BFS order  
    int* rct_distances; ///< distances to root 
    int* rct_nodes_start; ///< length of (num_nets + 1)
};

void rct_compute_cuda(RctCUDA data_cpu);
void rct_bfs_cuda(RctEdgeArrayCUDA data_cpu); 
