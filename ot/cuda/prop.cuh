/**
 * @file   prop.cuh
 * @author Yibo Lin
 * @date   Mar 2020
 */

#pragma once 

#include "ot/cuda/graph.cuh"
#include "ot/cuda/flat_table.cuh"

struct PinInfoCUDA {
    float value = 0;
    int from_arcidx;
    unsigned char from_el; 
    unsigned char from_rf; 
};

struct PropCUDA {
    float* net_arc_delays = nullptr; ///< length of num_arcs * MAX_SPLIT_TRAN, only entries for net arcs are valid 
    float* net_arc_impulses = nullptr; ///< length of num_arcs * MAX_SPLIT_TRAN, only entries for net arcs are valid 
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

    PinInfoCUDA* pin_slews = nullptr; ///< length of num_pins * MAX_SPLIT_TRAN, both input and output 
    PinInfoCUDA* pin_ats = nullptr; ///< length of num_pins * MAX_SPLIT_TRAN, both input and output  
    float* cell_arc_delays = nullptr; ///< length of num_arcs * MAX_SPLIT_TRAN * MAX_TRAN, output 

    /// destroy on cuda 
    void destroy_device();

    /// copy to device, the object itself must be on host 
    /// Assume rhs has not been allocated yet 
    void copy2device(PropCUDA& rhs) const;
};

void prop_cuda(PropCUDA& data_cpu); 
