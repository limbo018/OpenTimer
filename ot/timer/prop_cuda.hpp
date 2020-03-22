
#pragma once

namespace ot {
  // it should be even with prop.cuh
  struct InfoPinCUDA {
    float num;
    int flag;
    int fr_el, fr_rf, fr_arcidx;
  };
  struct ArcCUDA {
    int from, to, idx;
  };

  struct FlatTableStorage {
    int num_tables, total_num_xs, total_num_ys, total_num_xsys;
    std::vector<float> xs, ys, data;
    std::vector<int> xs_st, ys_st, data_st;
  };

  struct PropStorage {
    int num_pins;
    std::vector<float> netload_pins4;
    std::vector<InfoPinCUDA> slew_pins4, at_pins4;

    int num_levels, total_num_netarcs, total_num_cellarcs;
    std::vector<int> netarc_ends, cellarc_ends;
    std::vector<ArcCUDA> netarcs, cellarcs;
    std::vector<float> impulse_netarcs4, delay_netarcs4;

    std::vector<int> lutidx_slew_cellarcs8, lutidx_delay_cellarcs8;
    FlatTableStorage fts;
  };
}
