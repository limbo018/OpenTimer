/**
 * @file   prop.cuh
 * @author Yibo Lin
 * @date   Mar 2020
 */

#pragma once 


struct FlatTableCUDA {
  int num_tables; // number of valid tables, only count for existing ones
  int total_num_xs, total_num_ys, total_num_xsys;
  float *xs, *ys;
  float *data;
  int *xs_st, *ys_st, *data_st;
};

// it should be even with timer.hpp
struct InfoPinCUDA {
  float num;
  int flag;
  int fr_el, fr_rf, fr_arcidx;
};
struct ArcCUDA {
  int from, to, idx;
};

struct PropCUDA {
  // pin info
  // slew_pins, at_pins need to be initialized to \pm \infinity for max/min.
  // for primary inputs and outputs, also need to init their slew/at_pins.
  // flag should be set to zero
  // all scalars such as num_pins, total_xxx need not be copied to GPU.
  int num_pins;
  float *netload_pins4;
  InfoPinCUDA *slew_pins4, *at_pins4;

  // arc info
  // two CSRs
  // the xx_ends array need not be copied to GPU.
  int num_levels, total_num_netarcs, total_num_cellarcs;
  int *netarc_ends, *cellarc_ends;
  ArcCUDA *netarcs, *cellarcs;
  float *impulse_netarcs4, *delay_netarcs4;

  // lut
  int *lutidx_slew_cellarcs8, *lutidx_delay_cellarcs8;
  FlatTableCUDA ft;

  // generated when running kernel
  // currently need not be allocated on CPU counterpart
  float *tmpslew_netarcs4, *tmpslew_cellarcs8;
  float *delay_cellarcs8;  // if need this, one can copy it to cpu
};

void fprop_cuda(PropCUDA &prop_data_cpu); 
