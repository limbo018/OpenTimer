/**
 * @file   prop.cuh
 * @author Yibo Lin
 * @date   Mar 2020
 */

#pragma once 

#include <vector>
#include "ot/cuda/graph.cuh"
#include "ot/cuda/flat_table.cuh"

struct PinInfoCUDA {
    float value = 0;
    int from_arcidx;
    unsigned char from_el; 
    unsigned char from_rf; 
};

struct NetArcInfo {
    float delays[4]; ///< MAX_SPLIT_TRAN 
    float impulses[4]; ///< MAX_SPLIT_TRAN 
};

struct CellArcInfo {
    float delays[8]; ///< MAX_SPLIT_TRAN * MAX_TRAN
};

/// combine net arc and cell arc together 
union ArcInfo {
    NetArcInfo net_arc; 
    CellArcInfo cell_arc; 
};

struct PropCUDA {
    float* pin_loads = nullptr; ///< length of num_pins * MAX_SPLIT_TRAN, only entries for the driver of net arcs are valid; 
                            ///< we only need the loads of output pin of a cell, which is the driver/root of a net
    int* arc2ftid = nullptr; ///< length of num_arcs * {MAX_SPLIT * MAX_TRAN * MAX_TRAN}; 
                  ///< if infinity, then it is a net arc; 
                  ///< else if equal to ft.num_tables, then cell arc without timing table; 
                  ///< else, a cell arc with timing table 

    int* frontiers = nullptr; ///< it might be faster to switch from pins to arcs 
    int* frontiers_ends = nullptr; ///< length of num_levels + 1
    int num_levels; 
    int num_pins; 
    int num_arcs; 

    FlatArcGraphCUDA fanin_graph;
    FlatTableCUDA slew_ft;
    FlatTableCUDA delay_ft;

    int* fanout_degrees; ///< length of num_pins, for toposort, will be modified 

    PinInfoCUDA* pin_slews = nullptr; ///< length of num_pins * MAX_SPLIT_TRAN, both input and output 
    PinInfoCUDA* pin_ats = nullptr; ///< length of num_pins * MAX_SPLIT_TRAN, both input and output  
    ArcInfo* arc_infos = nullptr; ///< length of num_arcs, combine net arcs and cell arcs together to save memory 

    /// init on cuda 
    void init_device(); 
    /// destroy on cuda 
    void destroy_device();

    /// copy to device, the object itself must be on host 
    /// Assume rhs has not been allocated yet 
    void copy2device(PropCUDA& rhs) const;
    void copy_fanin_graph(FlatArcGraphCUDA const& host_data);
    void copy_slew_ft(FlatTableCUDA const& host_data);
    void copy_delay_ft(FlatTableCUDA const& host_data);
    void copy_fanout_degrees(std::vector<int> const& host_fanout_degrees); 
    void copy_pin_loads(std::vector<float> const& host_pin_loads);
    void copy_arc2ftid(std::vector<int> const& host_arc2ftid);
    void alloc_frontiers(int n);
    void copy_frontiers_ends(std::vector<int> const& host_frontiers_ends);
    void copy_pin_slews(std::vector<PinInfoCUDA> const& host_pin_slews); 
    void copy_pin_ats(std::vector<PinInfoCUDA> const& host_pin_ats); 
    void copy_arc_infos(std::vector<ArcInfo> const& host_arc_infos);
};

void prop_cuda(PropCUDA& data_cpu, PropCUDA& data_cuda); 

void toposort_compute_cuda(
        PropCUDA& prop_data_cpu, PropCUDA& prop_data_cuda, 
        std::vector<int> &frontiers_ends
  );
