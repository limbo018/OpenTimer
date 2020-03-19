/**
 * @file   prop.cuh
 * @author Yibo Lin
 * @date   Mar 2020
 */

#pragma once 

struct ArcCUDA {
    unsigned from; ///< from pin 
    unsigned to; ///< to pin 
};

struct FlatTableCUDA {
  unsigned num_tables; // number of valid tables, only count for existing ones 

  float* slew_indices1;  
  float* slew_indices2;  
  float* slew_table;           
  
  float* delay_indices1;  
  float* delay_indices2;  
  float* delay_table;           

  unsigned* slew_indices1_start;
  unsigned* slew_indices2_start;
  unsigned* slew_table_start;

  unsigned* delay_indices1_start;
  unsigned* delay_indices2_start;
  unsigned* delay_table_start;
};

struct PropCUDA {
    ArcCUDA* arcs; 
    float* arc_delays; ///< length of num_arcs * MAX_SPLIT_TRAN, only entries for net arcs are valid 
    float* arc_impulses; ///< length of num_arcs * MAX_SPLIT_TRAN, only entries for net arcs are valid 
    float* arc_loads; ///< length of num_arcs * MAX_SPLIT_TRAN, only entries for net arcs are valid; 
                            ///< we only need the loads of output pin of a cell, which is the driver/root of a net
    unsigned* arc2ftid; ///< length of num_arcs * {MAX_SPLIT * MAX_TRAN * MAX_TRAN}; 
                              ///< if infinity, then it is a net arc; 
                              ///< else if equal to ft.num_tables, then cell arc without timing table; 
                              ///< else, a cell arc with timing table 

    int* frontiers; ///< it might be faster to switch from pins to arcs 
    int* frontiers_ends; ///< length of num_levels + 1

    FlatTableCUDA ft;

    unsigned num_arcs; 
    unsigned num_levels; 

    /// @brief get the FlatTableCUDA table id 
    unsigned ftid(unsigned arc_id, unsigned el, unsigned irf, unsigned orf) const; 
};

void prop_cuda(PropCUDA& prop_data_cpu); 
