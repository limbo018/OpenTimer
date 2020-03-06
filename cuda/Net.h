/**
 * @file   Net.h
 * @author Yibo Lin
 * @date   Feb 2020
 */
#ifndef _NET_H
#define _NET_H

template <typename T>
struct NetCPU 
{
    unsigned int num_nodes; 
    std::vector<unsigned char> adjacency_matrix; 
    std::vector<T> node_res[MAX_SPLIT][MAX_TRANS]; ///< resistance of the input edge of a node 
    std::vector<T> node_cap[MAX_SPLIT][MAX_TRANS]; ///< total load capacitance, initially only has capacitance at the leaves 
    std::vector<T> node_delay[MAX_SPLIT][MAX_TRANS]; ///< delay at each node 
    std::vector<T> node_cap_delay[MAX_SPLIT][MAX_TRANS]; ///< capacitance * delay, similar propagation as node_cap 
    std::vector<T> node_beta[MAX_SPLIT][MAX_TRANS]; ///< used to compute slew 
    std::vector<T> node_slew[MAX_SPLIT][MAX_TRANS]; ///< slew at each node, the slew[0] is initialized to the input slew  

    bool edge(unsigned int i, unsigned int j) const 
    {
        return adjacency_matrix[i*num_nodes + j];
    }
};

template <typename T>
void compute_net_timing_cpu(std::vector<NetCPU<T> >& net_array)
{
    for (int el = 0; el < MAX_SPLIT; ++el)
    {
        for (int rf = 0; rf < MAX_TRANS; ++rf)
        {
#pragma omp parallel for num_threads(10)
            for (unsigned int net_id = 0; net_id < net_array.size(); ++net_id)
            {
                auto& net = net_array[net_id];
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
    }
}

#endif
