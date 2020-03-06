/**
 * @file   Net.cuh
 * @author Yibo Lin
 * @date   Feb 2020
 */

#ifndef _NET_CUH
#define _NET_CUH

#define MAX_SPLIT 2 ///< early/late 
#define MAX_TRANS 2 ///< rising/fall 

template <typename T>
struct NetArray 
{
    struct Net 
    {
        unsigned int num_nodes; 
        unsigned char* adjacency_matrix; 
        T* node_res[MAX_SPLIT][MAX_TRANS]; ///< resistance of the input edge of a node 
        T* node_cap[MAX_SPLIT][MAX_TRANS]; ///< total load capacitance, initially only has capacitance at the leaves 
        T* node_delay[MAX_SPLIT][MAX_TRANS]; ///< delay at each node 
        T* node_cap_delay[MAX_SPLIT][MAX_TRANS]; ///< capacitance * delay, similar propagation as node_cap 
        T* node_beta[MAX_SPLIT][MAX_TRANS]; ///< used to compute slew 
        T* node_impulse[MAX_SPLIT][MAX_TRANS]; ///< used to compute \hat{s_{oT}}

        __device__ bool edge(unsigned int i, unsigned int j) const 
        {
            return adjacency_matrix[i*num_nodes + j];
        }
    };

    unsigned char* net_adjacency_matrices; ///< length of sum (#nodes in net)^2 for all nets 
    unsigned int* net_adjacency_matrices_start; ///< length of number of nets + 1
    T* net_node_res[MAX_SPLIT][MAX_TRANS]; ///< length of total number of nodes in all nets 
    T* net_node_cap[MAX_SPLIT][MAX_TRANS]; ///< length of total number of nodes in all nets 
    T* net_node_delay[MAX_SPLIT][MAX_TRANS]; 
    T* net_node_cap_delay[MAX_SPLIT][MAX_TRANS];
    T* net_node_beta[MAX_SPLIT][MAX_TRANS];
    T* net_node_impulse[MAX_SPLIT][MAX_TRANS];
    unsigned int* net_node_attr_start; ///< length of number of nets + 1
    unsigned int num_nets; ///< number of nets 

    __device__ Net getNet(unsigned int net_id) 
    {
        Net net; 
        unsigned int offset = net_adjacency_matrices_start[net_id];
        net.adjacency_matrix = net_adjacency_matrices + offset; 
        offset = net_node_attr_start[net_id];
        net.num_nodes = net_node_attr_start[net_id + 1] - offset; 
        //printf("offset1 = %u, offset2 = %u, num_nodes = %u\n", net_adjacency_matrices_start[net_id], net_node_attr_start[net_id], net.num_nodes);
        
        for (int i = 0; i < MAX_SPLIT; ++i)
        {
            for (int j = 0; j < MAX_TRANS; ++j)
            {
                net.node_res[i][j] = net_node_res[i][j] + offset;
                net.node_cap[i][j] = net_node_cap[i][j] + offset;
                net.node_delay[i][j] = net_node_delay[i][j] + offset; 
                net.node_cap_delay[i][j] = net_node_cap_delay[i][j] + offset; 
                net.node_beta[i][j] = net_node_beta[i][j] + offset; 
                net.node_impulse[i][j] = net_node_impulse[i][j] + offset; 
            }
        }

        return net; 
    }
};

template <typename T>
__global__ void compute_net_timing(NetArray<T> net_array)
{
    unsigned int net_id = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int el = threadIdx.y; 
    unsigned int rf = threadIdx.z; 

    if (net_id < net_array.num_nets)
    {
        // assume the adjacency matrix of each net is ordered by BFS 
        typename NetArray<T>::Net net = net_array.getNet(net_id);

        // update load from downstream to upstream 
        for (int i = net.num_nodes - 1; i >= 0; --i)
        {
            for (int j = i - 1; j >= 0; --j) // parent of i must be earlier than i  
            {
                if (net.edge(i, j))
                {
                    net.node_cap[el][rf][j] += net.node_cap[el][rf][i];
                    break; 
                }
            }
        }


        // update delay from upstream to downstream 
        for (int i = 0; i < net.num_nodes; ++i)
        {
            for (int j = i + 1; j < net.num_nodes; ++j) // children of i must be later than i 
            {
                net.node_delay[el][rf][j] += net.node_delay[el][rf][i] + net.node_cap[el][rf][j] * net.node_res[el][rf][j];
            }
        }

        // update cap*delay from downstream to upstream 
        for (int i = net.num_nodes - 1; i >= 0; --i)
        {
            bool leaf_flag = true; 
            for (int j = i + 1; j < net.num_nodes; ++j)
            {
                if (net.edge(i, j)) // not leaf
                {
                    leaf_flag = false; 
                    break; 
                }
            }
            if (leaf_flag)
            {
                net.node_cap_delay[el][rf][i] = net.node_cap[el][rf][i] * net.node_delay[el][rf][i];
            }
            for (int j = i - 1; j >= 0; --j) // parent of i must be earlier than i  
            {
                if (net.edge(i, j))
                {
                    net.node_cap_delay[el][rf][j] += net.node_cap_delay[el][rf][i];
                    break; 
                }
            }
        }

        // update beta from upstream to downstream 
        for (int i = 0; i < net.num_nodes; ++i)
        {
            for (int j = i + 1; j < net.num_nodes; ++j) // children of i must be later than i 
            {
                net.node_beta[el][rf][j] += net.node_beta[el][rf][i] + net.node_cap_delay[el][rf][j] * net.node_res[el][rf][j];
            }
        }
    }
}

#endif
