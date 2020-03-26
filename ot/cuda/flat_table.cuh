/**
 * @file   flat_table.cuh
 * @author Yibo Lin
 * @date   Mar 2020
 */

#pragma once 

struct FlatTableCUDA {
  int num_tables; // number of valid tables, only count for existing ones
  int total_num_xs, total_num_ys, total_num_data;
  float *xs = nullptr, *ys = nullptr;
  float *data = nullptr;
  int *xs_st = nullptr, *ys_st = nullptr, *data_st = nullptr;

  /// destroy on cuda 
  void destroy_device();

  /// copy to device, the object itself must be on host 
  /// Assume rhs has not been allocated yet 
  void copy2device(FlatTableCUDA& rhs) const; 
};

