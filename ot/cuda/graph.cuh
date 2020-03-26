/**
 * @file   flat_graph.cuh
 * @author Yibo Lin
 * @date   Mar 2020
 */

#pragma once 

#include <vector>

struct FlatGraph {
    /// a flat storage of std::vector<std::vector<int>> adjacency_list;
    std::vector<int> adjacency_list; ///< elements in the adjacency_list
    std::vector<int> adjacency_list_start; ///< length of num_nodes + 1, last element is the number of edges 
    int num_nodes; ///< number of nodes

    void set_num_nodes(int n) {
        num_nodes = n;
        adjacency_list_start.assign(n + 1, 0);
    }
};

/// only record arc idx and to pin 
/// the pin can fanin or fanout 
struct FlatArc {
    int idx; ///< encoded with arc type, the lowest bit 0 for net arc, 1 for cell arc 
    int other; ///< the other node; for fanin, it is from; for fanout, it is to
};

struct FlatArcGraph {
    /// a flat storage of std::vector<std::vector<int>> adjacency_list;
    std::vector<FlatArc> adjacency_list; ///< elements in the adjacency_list
    std::vector<int> adjacency_list_start; ///< length of num_nodes + 1, last element is the number of arcs 
    int num_nodes; ///< number of nodes

    void set_num_nodes(int n) {
        num_nodes = n;
        adjacency_list_start.assign(n + 1, 0);
    }
};

struct FlatArcGraphCUDA {
    /// a flat storage of std::vector<std::vector<int>> adjacency_list;
    FlatArc* adjacency_list = nullptr; ///< elements in the adjacency_list
    int* adjacency_list_start = nullptr; ///< length of num_nodes + 1, last element is the number of arcs 
    int num_nodes; ///< number of nodes

    /// destroy on cuda 
    void destroy_device();

    /// copy to device, the object itself must be on host 
    /// Assume rhs has not been allocated yet 
    void copy2device(FlatArcGraphCUDA& rhs) const; 
};
