#include <ot/timer/timer.hpp>
#include <ot/timer/prop_cuda.hpp>
#include <ot/cuda/prop.cuh>

namespace ot {
  
  // Procedure: _flatten_liberty
  void Timer::_flatten_liberty() {
    // TODO......
    // put slew lut and delay lut in a unified {x_start, y_start, data_start}-like CSR
    // generate 8 index of slew lut and 8 index of delay lut for each cell arc
    // fix var-type order (x, y) = <slew, load>, transpose the matrix when necessary
    // expand the matrix to be at least 2x2, adding dummy rows and columns, and don't let the dummy rows and columns' coordinate coincide with the old one, you can, e.g., set x' = x + 1, y' = y + 1
    // the old code is in commit history... probably needs some modification
  }

  // Procedure: _fprop_cuda_init
  void Timer::_fprop_cuda_init() {
    _prop_stor.emplace();
    _prop_stor->num_pins = _pins.size();
    _prop_stor->netload_pins.resize(_pins.size() * 4);
    _prop_stor->slew_pins4.resize(_pins.size() * 4);
    _prop_stor->at_pins4.resize(_pins.size() * 4);

    _prop_stor->num_levels = _prop_frontiers_ends.size() - 1;
  }

  // Procedure: _fprop_cuda_init_pin
  void Timer::_fprop_cuda_init_pin(Pin &pin) {
    float MAX = 1e10, MIN = -1e10;
    FOR_EACH_EL_RF(el, rf) {
      _prop_stor->netload_pins4[pin._idx * 4 + el * 2 + rf] = pin._net.load(el, rf);
      slew_pins4[pin._idx * 4 + el * 2 + rf] = {el ? MIN : MAX, 0, 0, 0, 0};
    }
    if(auto pi = pin.primary_input(); pi) {
      FOR_EACH_EL_RF_IF(el, rf, pi->_slew[el][rf]) {
        slew_pins4[pin._idx * 4 + el * 2 + rf] = {*(pi->_slew[el][rf]),
                                                  0, el, rf, -1};
      }
      FOR_EACH_EL_RF_IF(el, rf, pi->_at[el][rf]) {
        at_pins4[pin._idx * 4 + el * 2 + rf] = {*(pi->_at[el][rf]),
                                                0, el, rf, -1};
      }
    }
  }

  // Procedure: _fprop_cuda_init_arc
  // not implemented...

  // Procedure: _fprop_cuda_action
  void Timer::_fprop_cuda_action() {
    // Step 1: vectors to vectors::data()
    PropStorage s = *_prop_stor;
    PropCUDA propCUDA = {
      s.num_pins,
      s.netload_pins4.data(),
      s.slew_pins4.data(), s.at_pins4.data(),

      s.num_levels, s.total_num_netarcs, s.total_num_cellarcs,
      s.netarc_ends.data(), s.cellarc_ends.data(),
      s.netarcs.data(), s.cellarcs.data(),
      s.impulse_netarcs4.data(), s.delay_netarcs4.data(),

      s.lutidx_slew_cellarcs8.data(), s.lutidx_delay.cellarcs8.data(),
      /* ft */ {
        s.fts.num_tables, s.fts.total_num_xs, s.fts.total_num_ys, s.fts.total_num_xsys,
        s.fts.xs.data(), s.fts.ys.data(), s.fts.data.data(),
        s.fts.xs_st.data(), s.fts.ys_st.data(), s.fts.data_st.data()
      },

      std::nullptr, std::nullptr,
      std::nullptr
    };

    fprop_cuda(propCUDA);
  }

  // Procedure: _fprop_cuda_writeback_pin
  void Timer::_fprop_cuda_writeback_pin(Pin &pin) {
    float MAX = 1e10, MIN = -1e10;
    FOR_EACH_EL_RF(el, rf) {
      InfoPinCUDA &islew = _prop_stor->slew_pins4[pin._idx * 4 + el * 2 + rf];
      InfoPinCUDA &iat = _prop_stor->at_pins4[pin._idx * 4 + el * 2 + rf];

      if(islew.num != MAX && islew.num != MIN) {
        pin._slew[el][rf].emplace(islew.fr_arcidx >= 0 ? _idx2arc[islew.fr_arcidx] : std::nullptr,
                                  islew.fr_el, islew.fr_rf, islew.num);
      }
      if(iat.num != MAX && iat.num != MIN) {
        pin._at[el][rf].emplace(iat.fr_arcidx >= 0 ? _idx2arc[iat.fr_arcidx] : std::nullptr,
                                iat.fr_el, iat.fr_rf, iat.num);
      }
    }

    /* don't know if it's necessary to set arc::delay as well, but I suspect so
       if it's necessary, then one should add PropStorage::delay_cellarcs8 
       and copy PropCUDA::delay_cellarcs8 from gpu */
  }
}

