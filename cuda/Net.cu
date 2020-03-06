/**
 * @file   Net.cu
 * @author Yibo Lin
 * @date   Feb 2020
 */

#include <vector>
#include <deque>
#include <algorithm>
#include <random>
#include <cassert>
#include <cstring>
#include "Util.cuh"
#include "Net.cuh"
#include "Net.h"

template <typename T>
NetCPU<T> randomNet(int degree)
{
    std::vector<unsigned char> adjacency_matrix (degree * degree, 0); 

    std::deque<int> queue; 
    std::vector<unsigned char> visited (degree, 0); 
    std::vector<int> remaining_nodes (degree); 
    std::iota(remaining_nodes.begin(), remaining_nodes.end(), 0);

    visited[0] = true; 
    remaining_nodes.erase(std::find(remaining_nodes.begin(), remaining_nodes.end(), 0));
    queue.push_back(0); 

    while (!queue.empty())
    {
        int s = queue.front();
        queue.pop_front(); 
        
        int fanouts = (std::rand() % (degree / 2 + 1)) + 1; 
        for (int i = 0; i < fanouts; ++i)
        {
            if (remaining_nodes.empty())
            {
                break; 
            }
            int target = remaining_nodes.at(std::rand() % remaining_nodes.size()); 
            if (!adjacency_matrix[s * degree + target] && !visited[target])
            {
                adjacency_matrix[s * degree + target] = 1; 
                adjacency_matrix[target * degree + s] = 1; 
                //printf("add edge %u --> %u, degree %d, fanouts %d\n", s, target, degree, fanouts);

                visited[target] = true; 
                remaining_nodes.erase(std::find(remaining_nodes.begin(), remaining_nodes.end(), target));
                queue.push_back(target); 
            }
        }
    }

    // do BFS again, I cannot guarantee the graph is in BFS order 
    std::fill(visited.begin(), visited.end(), false);
    visited[0] = true; 
    queue.push_back(0); 

    std::vector<unsigned int> bfs_order; 
    while (!queue.empty())
    {
        int s = queue.front();
        bfs_order.push_back(s);
        queue.pop_front(); 
        
        for (unsigned int i = 0; i < degree; ++i)
        {
            if (adjacency_matrix[s * degree + i])
            {
                if (!visited[i])
                {
                    visited[i] = true; 
                    queue.push_back(i);
                }
            }
        }
    }

    std::vector<unsigned char> new_adjacency_matrix (adjacency_matrix.size(), 0);
    for (unsigned int i = 0; i < bfs_order.size(); ++i)
    {
        for (unsigned int j = 0; j < bfs_order.size(); ++j)
        {
            new_adjacency_matrix[ i * degree + j ] = adjacency_matrix[ bfs_order[i] * degree + bfs_order[j] ];
        }
    }

    NetCPU<T> net; 
    net.num_nodes = degree; 
    net.adjacency_matrix = new_adjacency_matrix; 
    for (int i = 0; i < degree; ++i)
    {
        int num_neighbors = std::count(net.adjacency_matrix.begin() + i * degree, net.adjacency_matrix.begin() + (i + 1) * degree, 1);
        assert(num_neighbors);
    }
    for (int k = 0; k < MAX_SPLIT; ++k)
    {
        for (int h = 0; h < MAX_TRANS; ++h)
        {
            net.node_res[k][h].resize(degree); 
            net.node_cap[k][h].resize(degree); 
            net.node_delay[k][h].resize(degree); 
            net.node_cap_delay[k][h].resize(degree); 
            net.node_beta[k][h].resize(degree); 
            for (int i = 0; i < degree; ++i)
            {
                int num_neighbors = std::count(net.adjacency_matrix.begin() + i * degree, net.adjacency_matrix.begin() + (i + 1) * degree, 1);
                net.node_res[k][h][i] = std::rand() % 1000; 
                net.node_cap[k][h][i] = (num_neighbors == 1)? std::rand() % 1000 : 0; 
                net.node_delay[k][h][i] = 0; 
                net.node_cap_delay[k][h][i] = 0; 
                net.node_beta[k][h][i] = 0; 
            }
        }
    }

#if 0
    printf("net[%u]\n", degree);
    for (unsigned int i = 0; i < degree; ++i)
    {
        for (unsigned int j = i + 1; j < degree; ++j)
        {
            if (net.adjacency_matrix[i * degree + j])
            {
                printf("edge %u --> %u\n", i, j);
            }
        }
    }
#endif

    return net; 
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("usage: ./timing #nets\n");
        return -1; 
    }
    std::srand(123);
    unsigned int num_nets = atoi(argv[1]);
    int avg_degree = 8; 
    using datatype = float; 

    hr_clock_rep start, stop; 
    start = get_globaltime(); 
    std::vector<NetCPU<datatype>> nets (num_nets); 
    for (unsigned int i = 0; i < num_nets; ++i)
    {
        unsigned int degree = (std::rand() % avg_degree) + avg_degree; 
        nets[i] = randomNet<datatype>(degree);
    }
    stop = get_globaltime(); 
    printf("generate nets time %g ms\n", (stop - start) * get_timer_period());

    start = get_globaltime();
    unsigned int count = 0; 
    unsigned int count2 = 0; 
    for (unsigned int i = 0; i < num_nets; ++i)
    {
        count += nets[i].adjacency_matrix.size(); 
        count2 += nets[i].num_nodes; 
    }
    printf("count1 = %u, count2 = %u, mem %u (%u + %u) integers\n", count, count2, 
            count / 4 + (num_nets + 1) + count2 * 5 * MAX_SPLIT * MAX_TRANS + (num_nets + 1), 
            count / 4 + (num_nets + 1), 
            count2 * 5 * MAX_SPLIT * MAX_TRANS + (num_nets + 1)
            );
    std::vector<unsigned char> net_adjacency_matrices (count); 
    std::vector<unsigned int> net_adjacency_matrices_start (num_nets + 1);
    std::vector<datatype> net_node_res[MAX_SPLIT][MAX_TRANS]; ///< length of total number of nodes in all nets 
    std::vector<datatype> net_node_cap[MAX_SPLIT][MAX_TRANS]; ///< length of total number of nodes in all nets 
    std::vector<datatype> net_node_delay[MAX_SPLIT][MAX_TRANS]; 
    std::vector<datatype> net_node_cap_delay[MAX_SPLIT][MAX_TRANS];
    std::vector<datatype> net_node_beta[MAX_SPLIT][MAX_TRANS];
    std::vector<unsigned int> net_node_attr_start (num_nets + 1); ///< length of number of nets + 1
    for (int k = 0; k < MAX_SPLIT; ++k)
    {
        for (int h = 0; h < MAX_TRANS; ++h)
        {
            net_node_res[k][h].resize(count2); 
            net_node_cap[k][h].resize(count2); 
            net_node_delay[k][h].resize(count2); 
            net_node_cap_delay[k][h].resize(count2); 
            net_node_beta[k][h].resize(count2);
        }
    }

    count = 0; 
    count2 = 0; 
    for (unsigned int i = 0; i < num_nets; ++i)
    {
        auto const& net = nets[i]; 
        net_adjacency_matrices_start[i] = count; 
        std::copy(net.adjacency_matrix.begin(), net.adjacency_matrix.end(), net_adjacency_matrices.begin() + count); 
        count += net.adjacency_matrix.size(); 

        net_node_attr_start[i] = count2; 
        for (int k = 0; k < MAX_SPLIT; ++k)
        {
            for (int h = 0; h < MAX_TRANS; ++h)
            {
                std::copy(net.node_res[k][h].begin(), net.node_res[k][h].end(), net_node_res[k][h].begin() + count2); 
                std::copy(net.node_cap[k][h].begin(), net.node_cap[k][h].end(), net_node_cap[k][h].begin() + count2); 
                std::copy(net.node_delay[k][h].begin(), net.node_delay[k][h].end(), net_node_delay[k][h].begin() + count2); 
                std::copy(net.node_cap_delay[k][h].begin(), net.node_cap_delay[k][h].end(), net_node_cap_delay[k][h].begin() + count2); 
                std::copy(net.node_beta[k][h].begin(), net.node_beta[k][h].end(), net_node_beta[k][h].begin() + count2); 
            }
        }
        count2 += net.num_nodes; 
    }
    net_adjacency_matrices_start[num_nets] = count; 
    net_node_attr_start[num_nets] = count2; 
    stop = get_globaltime(); 
    printf("flat cpu time %g ms\n", (stop - start) * get_timer_period());

    start = get_globaltime();
    NetArray<datatype> net_array; 
    net_array.num_nets = num_nets; 
    allocateCopyCUDA(net_array.net_adjacency_matrices, net_adjacency_matrices.data(), net_adjacency_matrices.size());
    allocateCopyCUDA(net_array.net_adjacency_matrices_start, net_adjacency_matrices_start.data(), net_adjacency_matrices_start.size());
    for (int k = 0; k < MAX_SPLIT; ++k)
    {
        for (int h = 0; h < MAX_TRANS; ++h)
        {
            allocateCopyCUDA(net_array.net_node_res[k][h], net_node_res[k][h].data(), net_node_res[k][h].size());
            allocateCopyCUDA(net_array.net_node_cap[k][h], net_node_cap[k][h].data(), net_node_cap[k][h].size());
            allocateCopyCUDA(net_array.net_node_delay[k][h], net_node_delay[k][h].data(), net_node_delay[k][h].size());
            allocateCopyCUDA(net_array.net_node_cap_delay[k][h], net_node_cap_delay[k][h].data(), net_node_cap_delay[k][h].size());
            allocateCopyCUDA(net_array.net_node_beta[k][h], net_node_beta[k][h].data(), net_node_beta[k][h].size());
        }
    }
    allocateCopyCUDA(net_array.net_node_attr_start, net_node_attr_start.data(), net_node_attr_start.size());
    checkCUDA(cudaDeviceSynchronize());
    stop = get_globaltime(); 
    printf("init cuda time %g ms\n", (stop - start) * get_timer_period());

    start = get_globaltime(); 
    dim3 threads; 
    threads.x = 64; 
    threads.y = MAX_SPLIT; 
    threads.z = MAX_TRANS; 
    compute_net_timing<<<CPUCeilDiv(num_nets, threads.x), threads>>>(net_array);

    checkCUDA(cudaDeviceSynchronize());

    stop = get_globaltime(); 
    printf("cuda kernel time %g ms\n", (stop - start) * get_timer_period());

    start = get_globaltime(); 
    compute_net_timing_cpu(nets); 
    stop = get_globaltime(); 
    printf("cpu kernel time %g ms\n", (stop - start) * get_timer_period());

    return 0; 
}
