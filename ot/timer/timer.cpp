#include <ot/timer/timer.hpp>
#include <ot/timer/_prof.hpp>
#include <ot/cuda/prop.cuh>

namespace ot {

// ------------------------------------------------------------------------------------------------

// Function: set_num_threads
Timer& Timer::set_num_threads(unsigned n) {
  std::scoped_lock lock(_mutex);
  unsigned w = (n == 0) ? 0 : n-1;
  OT_LOGI("using ", n, " threads (", w, " worker)");
  // TODO
  //_taskflow.num_workers(w);
  return *this;
}

// Procedure: _add_to_lineage
void Timer::_add_to_lineage(tf::Task task) {
  _lineage | [&] (auto& p) { p.precede(task); };
  _lineage = task;
}

// Function: _max_pin_name_size
size_t Timer::_max_pin_name_size() const {
  if(_pins.empty()) {
    return 0;
  }
  else {
    return std::max_element(_pins.begin(), _pins.end(), 
      [] (const auto& l, const auto& r) {
        return l.second._name.size() < r.second._name.size();
      }
    )->second._name.size();
  }
}

// Function: _max_net_name_size
size_t Timer::_max_net_name_size() const {
  if(_nets.empty()) {
    return 0;
  }
  else {
    return std::max_element(_nets.begin(), _nets.end(), 
      [] (const auto& l, const auto& r) {
        return l.second._name.size() < r.second._name.size();
      }
    )->second._name.size();
  }
}

// Function: repower_gate
// Change the size or level of an existing gate, e.g., NAND2_X2 to NAND2_X3. The gate's
// logic function and topology is guaranteed to be the same, along with the currently-connected
// nets. However, the pin capacitances of the new cell type might be different. 
Timer& Timer::repower_gate(std::string gate, std::string cell) {

  std::scoped_lock lock(_mutex);

  auto task = _taskflow.emplace([this, gate=std::move(gate), cell=std::move(cell)] () {
    _repower_gate(gate, cell);
  });
  
  _add_to_lineage(task);

  return *this;
}

// Procedure: _repower_gate
void Timer::_repower_gate(const std::string& gname, const std::string& cname) {
  
  OT_LOGE_RIF(!_celllib[MIN] || !_celllib[MAX], "celllib not found");

  // Insert the gate if it doesn't exist.
  if(auto gitr = _gates.find(gname); gitr == _gates.end()) {
    OT_LOGW("gate ", gname, " doesn't exist (insert instead)");
    _insert_gate(gname, cname);
    return;
  }
  else {

    auto cell = CellView {_celllib[MIN]->cell(cname), _celllib[MAX]->cell(cname)};

    OT_LOGE_RIF(!cell[MIN] || !cell[MAX], "cell ", cname, " not found");

    auto& gate = gitr->second;

    // Remap the cellpin
    for(auto pin : gate._pins) {
      FOR_EACH_EL(el) {
        assert(pin->cellpin(el));
        if(const auto cpin = cell[el]->cellpin(pin->cellpin(el)->name)) {
          pin->_remap_cellpin(el, *cpin);
        }
        else {
          OT_LOGE(
            "repower ", gname, " with ", cname, " failed (cellpin mismatched)"
          );  
        }
      }
    }
    
    gate._cell = cell;

    // reconstruct the timing and tests
    _remove_gate_arcs(gate);
    _insert_gate_arcs(gate);

    // Insert the gate to the frontier
    for(auto pin : gate._pins) {
      _insert_frontier(*pin);
      for(auto arc : pin->_fanin) {
        _insert_frontier(arc->_from);
      }
    }
  }
}

// Fucntion: insert_gate
// Create a new gate in the design. This newly-created gate is "not yet" connected to
// any other gates or wires. The gate to insert cannot conflict with existing gates.
Timer& Timer::insert_gate(std::string gate, std::string cell) {  
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, gate=std::move(gate), cell=std::move(cell)] () {
    _insert_gate(gate, cell);
  });

  _add_to_lineage(op);

  return *this;
}

// Function: _insert_gate
void Timer::_insert_gate(const std::string& gname, const std::string& cname) {

  OT_LOGE_RIF(!_celllib[MIN] || !_celllib[MAX], "celllib not found");

  if(_gates.find(gname) != _gates.end()) {
    OT_LOGW("gate ", gname, " already existed");
    return;
  }

  auto cell = CellView {_celllib[MIN]->cell(cname), _celllib[MAX]->cell(cname)};

  if(!cell[MIN] || !cell[MAX]) {
    OT_LOGE("cell ", cname, " not found in celllib");
    return;
  }
  
  auto& gate = _gates.try_emplace(gname, gname, cell).first->second;
  
  // Insert pins
  for(const auto& [cpname, ecpin] : cell[MIN]->cellpins) {

    CellpinView cpv {&ecpin, cell[MAX]->cellpin(cpname)};

    if(!cpv[MIN] || !cpv[MAX]) {
      OT_LOGF("cellpin ", cpname, " mismatched in celllib");
    }

    auto& pin = _insert_pin(gname + ':' + cpname);
    pin._handle = cpv;
    pin._gate = &gate;
    
    gate._pins.push_back(&pin);
  }
  
  _insert_gate_arcs(gate);
}

// Fucntion: remove_gate
// Remove a gate from the current design. This is guaranteed to be called after the gate has 
// been disconnected from the design using pin-level operations. The procedure iterates all 
// pins in the cell to which the gate was attached. Each pin that is being iterated is either
// a cell input pin or cell output pin. In the former case, the pin might have constraint arc
// while in the later case, the ot_pin.has no output connections and all fanin edges should be 
// removed here.
Timer& Timer::remove_gate(std::string gate) {  
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, gate=std::move(gate)] () {
    if(auto gitr = _gates.find(gate); gitr != _gates.end()) {
      _remove_gate(gitr->second);
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _remove_gate
void Timer::_remove_gate(Gate& gate) {

  // Disconnect this gate from the design.
  for(auto pin : gate._pins) {
    _disconnect_pin(*pin);
  }

  // Remove associated test
  for(auto test : gate._tests) {
    _remove_test(*test);
  }

  // Remove associated arcs
  for(auto arc : gate._arcs) {
    _remove_arc(*arc);
  }

  // Disconnect the gate and remove the pins from the gate
  for(auto pin : gate._pins) {
    _remove_pin(*pin);
  }

  // remove the gate
  _gates.erase(gate._name);
}

// Procedure: _remove_gate_arcs
void Timer::_remove_gate_arcs(Gate& gate) {

  // remove associated tests
  for(auto test : gate._tests) {
    _remove_test(*test);
  }
  gate._tests.clear();
  
  // remove associated arcs
  for(auto arc : gate._arcs) {
    _remove_arc(*arc);
  }
  gate._arcs.clear();
}

// Procedure: _insert_gate_arcs
void Timer::_insert_gate_arcs(Gate& gate) {

  assert(gate._tests.empty() && gate._arcs.empty());

  FOR_EACH_EL(el) {
    for(const auto& [cpname, cp] : gate._cell[el]->cellpins) {
      auto& to_pin = _insert_pin(gate._name + ':' + cpname);

      for(const auto& tm : cp.timings) {

        if(_is_redundant_timing(tm, el)) {
          continue;
        }

        TimingView tv{nullptr, nullptr};
        tv[el] = &tm;

        auto& from_pin = _insert_pin(gate._name + ':' + tm.related_pin);
        auto& arc = _insert_arc(from_pin, to_pin, tv);
        
        gate._arcs.push_back(&arc);
        if(tm.is_constraint()) {
          auto& test = _insert_test(arc);
          gate._tests.push_back(&test);
        }
      }
    }
  }
}

// Function: connect_pin
// Connect the pin to the corresponding net. The pin_name will either have the 
// <gate name>:<cell pin name> syntax (e.g., u4:ZN) or be a primary input. The net name
// will match an existing net read in from a .spef file.
Timer& Timer::connect_pin(std::string pin, std::string net) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, pin=std::move(pin), net=std::move(net)] () {
    auto p = _pins.find(pin);
    auto n = _nets.find(net);
    OT_LOGE_RIF(p==_pins.end() || n == _nets.end(),
      "can't connect pin ", pin,  " to net ", net, " (pin/net not found)"
    )
    _connect_pin(p->second, n->second);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _connect_pin
void Timer::_connect_pin(Pin& pin, Net& net) {
      
  // Connect the pin to the net and construct the edge connections.
  net._insert_pin(pin);
  
  // Case 1: the pin is the root of the net.
  if(&pin == net._root) {
    for(auto leaf : net._pins) {
      if(leaf != &pin) {
        _insert_arc(pin, *leaf, net);
      }
    }
  }
  // Case 2: the pin is not a root of the net.
  else {
    if(net._root) {
      _insert_arc(*net._root, pin, net);
    }
  }

  // TODO(twhuang) Enable the clock tree update?
}

// Procedure: disconnect_pin
// Disconnect the pin from the net it is connected to. The pin_name will either have the 
// <gate name>:<cell pin name> syntax (e.g., u4:ZN) or be a primary input.
Timer& Timer::disconnect_pin(std::string name) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    if(auto itr = _pins.find(name); itr != _pins.end()) {
      _disconnect_pin(itr->second);
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: disconnect_pin
// TODO (twhuang)
// try get rid of find_fanin which can be wrong under multiple arcs.
void Timer::_disconnect_pin(Pin& pin) {

  auto net = pin._net;

  if(net == nullptr) return;

  // Case 1: the pin is a root of the net (i.e., root of the rctree)
  if(&pin == net->_root) {
    // Iterate the pinlist and delete the corresponding edge. Notice here we cannot iterate
    // fanout of the node during removal.
    for(auto leaf : net->_pins) {
      if(leaf != net->_root) {
        auto arc = leaf->_find_fanin(*net->_root);
        assert(arc);
        _remove_arc(*arc);
      }
    }
  }
  // Case 2: the pin is not a root of the net.
  else {
    if(net->_root) {
      auto arc = pin._find_fanin(*net->_root);
      assert(arc);
      _remove_arc(*arc);
    }
  }
  
  // TODO: Enable the clock tree update.
  
  // Remove the pin from the net and enable the rc timing update.
  net->_remove_pin(pin);
}

// Function: insert_net
// Creates an empty net object with the input "net_name". By default, it will not be connected 
// to any pins and have no parasitics (.spef). This net will be connected to existing pins in 
// the design by the "connect_pin" and parasitics will be loaded by "spef".
Timer& Timer::insert_net(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    _insert_net(name);
  });

  _add_to_lineage(op);

  return *this;
}

// Function: _insert_net
Net& Timer::_insert_net(const std::string& name) {
  return _nets.try_emplace(name, name).first->second;
}

// Procedure: remove_net
// Remove a net from the current design, which by default removes all associated pins.
Timer& Timer::remove_net(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    if(auto itr = _nets.find(name); itr != _nets.end()) {
      _remove_net(itr->second);
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Function: _remove_net
void Timer::_remove_net(Net& net) {

  if(net.num_pins() > 0) {
    auto fetch = net._pins;
    for(auto pin : fetch) {
      _disconnect_pin(*pin);
    }
  }

  _nets.erase(net._name);
}

// Function: _insert_pin
Pin& Timer::_insert_pin(const std::string& name) {
  
  // pin already exists
  if(auto [itr, inserted] = _pins.try_emplace(name, name); !inserted) {
    return itr->second;
  }
  // inserted a new pon
  else {
    
    // Generate the pin idx
    auto& pin = itr->second;
    
    // Assign the idx mapping
    pin._idx = _pin_idx_gen.get();
    resize_to_fit(pin._idx + 1, _idx2pin);
    _idx2pin[pin._idx] = &pin;

    // insert to frontier
    _insert_frontier(pin);

    return pin;
  }
}

// Function: _remove_pin
void Timer::_remove_pin(Pin& pin) {

  assert(pin.num_fanouts() == 0 && pin.num_fanins() == 0 && pin.net() == nullptr);

  _remove_frontier(pin);

  // remove the id mapping
  _idx2pin[pin._idx] = nullptr;
  _pin_idx_gen.recycle(pin._idx);

  // remove the pin
  _pins.erase(pin._name);
}

Timer& Timer::cuda(bool flag) {
  if(flag) {
    OT_LOGI("enable cuda gpu acceleration");
    _insert_state(CUDA_ENABLED);
  }
  else {
    OT_LOGI("disable cuda gpu acceleration");
    _remove_state(CUDA_ENABLED);
  }

  return *this;
}

Timer& Timer::force_full_update(bool flag) {
  if(flag) {
    OT_LOGI("enable forced full update. subsequent call to update_timing will always launch a full update.");
    _insert_state(FULL_TIMING);
  }
  else {
    OT_LOGI("disable forced full update");
    _remove_state(FULL_TIMING);
  }

  return *this;
}

// Function: cppr
Timer& Timer::cppr(bool flag) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, flag] () {
    _cppr(flag);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _cppr
// Enable/Disable common path pessimism removal (cppr) analysis
void Timer::_cppr(bool enable) {
  
  // nothing to do.
  if((enable && _cppr_analysis) || (!enable && !_cppr_analysis)) {
    return;
  }

  if(enable) {
    OT_LOGI("enable cppr analysis");
    _cppr_analysis.emplace();
  }
  else {
    OT_LOGI("disable cppr analysis");
    _cppr_analysis.reset();
  }
    
  for(auto& test : _tests) {
    _insert_frontier(test._constrained_pin());
  }
}

// Function: clock
Timer& Timer::create_clock(std::string c, std::string s, float p) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, c=std::move(c), s=std::move(s), p] () {
    if(auto itr = _pins.find(s); itr != _pins.end()) {
      _create_clock(c, itr->second, p);
    }
    else {
      OT_LOGE("can't create clock ", c, " on source ", s, " (pin not found)");
    }
  });

  _add_to_lineage(op);
  
  return *this;
}

// Function: create_clock
Timer& Timer::create_clock(std::string c, float p) {
  
  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, c=std::move(c), p] () {
    _create_clock(c, p);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _create_clock
Clock& Timer::_create_clock(const std::string& name, Pin& pin, float period) {
  auto& clock = _clocks.try_emplace(name, name, pin, period).first->second;
  _insert_frontier(pin);
  return clock;
}

// Procedure: _create_clock
Clock& Timer::_create_clock(const std::string& name, float period) {
  auto& clock = _clocks.try_emplace(name, name, period).first->second;
  return clock;
}

// Function: insert_primary_input
Timer& Timer::insert_primary_input(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    _insert_primary_input(name);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _insert_primary_input
void Timer::_insert_primary_input(const std::string& name) {

  if(_pis.find(name) != _pis.end()) {
    OT_LOGW("can't insert PI ", name, " (already existed)");
    return;
  }

  assert(_pins.find(name) == _pins.end());

  // Insert the pin and and pi
  auto& pin = _insert_pin(name);
  auto& pi = _pis.try_emplace(name, pin).first->second;
  
  // Associate the connection.
  pin._handle = &pi;

  // Insert the pin to the frontier
  _insert_frontier(pin);

  // Create a net for the po and connect the pin to the net.
  auto& net = _insert_net(name); 
  
  // Connect the pin to the net.
  _connect_pin(pin, net);
}

// Function: insert_primary_output
Timer& Timer::insert_primary_output(std::string name) {

  std::scoped_lock lock(_mutex);

  auto op = _taskflow.emplace([this, name=std::move(name)] () {
    _insert_primary_output(name);
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _insert_primary_output
void Timer::_insert_primary_output(const std::string& name) {

  if(_pos.find(name) != _pos.end()) {
    OT_LOGW("can't insert PO ", name, " (already existed)");
    return;
  }

  assert(_pins.find(name) == _pins.end());

  // Insert the pin and and pi
  auto& pin = _insert_pin(name);
  auto& po = _pos.try_emplace(name, pin).first->second;
  
  // Associate the connection.
  pin._handle = &po;

  // Insert the pin to the frontier
  _insert_frontier(pin);

  // Create a net for the po and connect the pin to the net.
  auto& net = _insert_net(name); 

  // Connect the pin to the net.
  _connect_pin(pin, net);
}

// Procedure: _insert_test
Test& Timer::_insert_test(Arc& arc) {
  auto& test = _tests.emplace_front(arc);
  test._satellite = _tests.begin();
  test._pin_satellite = arc._to._tests.insert(arc._to._tests.end(), &test);
  return test;
}

// Procedure: _remove_test
void Timer::_remove_test(Test& test) {
  assert(test._satellite);
  if(test._pin_satellite) {
    test._arc._to._tests.erase(*test._pin_satellite);
  }
  _tests.erase(*test._satellite);
}

// Procedure: _remove_arc
// Remove an arc from the design. The procedure first disconnects the arc from its two ending
// pins, "from_pin" and "to_pin". Then it removes the arc from the design and insert both
// "from_pin" and "to_pin" into the pipeline.
void Timer::_remove_arc(Arc& arc) {

  assert(arc._satellite);
  
  arc._from._remove_fanout(arc);
  arc._to._remove_fanin(arc);

  // Insert the two ends to the frontier list.
  _insert_frontier(arc._from, arc._to);
  
  // remove the id mapping
  _idx2arc[arc._idx] = nullptr;
  _arc_idx_gen.recycle(arc._idx);

  // Remove this arc from the timer.
  _arcs.erase(*arc._satellite);
}

// Function: _insert_arc (net arc)
// Insert an net arc to the timer.
Arc& Timer::_insert_arc(Pin& from, Pin& to, Net& net) {

  OT_LOGF_IF(&from == &to, "net arc is a self loop at ", to._name);

  // Create a new arc
  auto& arc = _arcs.emplace_front(from, to, net);
  arc._satellite = _arcs.begin();

  from._insert_fanout(arc);
  to._insert_fanin(arc);

  // Insert frontiers
  _insert_frontier(from, to);
   
  // Assign the idx mapping
  arc._idx = _arc_idx_gen.get();
  resize_to_fit(arc._idx + 1, _idx2arc);
  _idx2arc[arc._idx] = &arc;

  return arc;
}

// Function: _insert_arc (cell arc)
// Insert a cell arc to the timing graph. A cell arc is a combinational link.
Arc& Timer::_insert_arc(Pin& from, Pin& to, TimingView tv) {
  
  //OT_LOGF_IF(&from == &to, "timing graph contains a self loop at ", to._name);

  // Create a new arc
  auto& arc = _arcs.emplace_front(from, to, tv);
  arc._satellite = _arcs.begin();
  from._insert_fanout(arc);
  to._insert_fanin(arc);

  // insert the arc into frontier list.
  _insert_frontier(from, to);
  
  // Assign the idx mapping
  arc._idx = _arc_idx_gen.get();
  resize_to_fit(arc._idx + 1, _idx2arc);
  _idx2arc[arc._idx] = &arc;

  return arc;
}

// Procedure: _fprop_rc_timing
void Timer::_fprop_rc_timing(Pin& pin) {
  if(auto net = pin._net; net) {
    net->_update_rc_timing();
  }
}

// Procedure: _fprop_slew
void Timer::_fprop_slew(Pin& pin) {
  
  // clear slew  
  pin._reset_slew();

  // PI
  if(auto pi = pin.primary_input(); pi) {
    FOR_EACH_EL_RF_IF(el, rf, pi->_slew[el][rf]) {
      pin._relax_slew(nullptr, el, rf, el, rf, *(pi->_slew[el][rf]));
    }
  }
  
  // Relax the slew from its fanin.
  for(auto arc : pin._fanin) {
    arc->_fprop_slew();
  }
}

// Procedure: _fprop_delay
void Timer::_fprop_delay(Pin& pin) {

  // clear delay
  for(auto arc : pin._fanin) {
    arc->_reset_delay();
  }

  // Compute the delay from its fanin.
  for(auto arc : pin._fanin) {
    arc->_fprop_delay();
  }
}

// Procedure: _fprop_at
void Timer::_fprop_at(Pin& pin) {
  
  // clear at
  pin._reset_at();

  // PI
  if(auto pi = pin.primary_input(); pi) {
    FOR_EACH_EL_RF_IF(el, rf, pi->_at[el][rf]) {
      pin._relax_at(nullptr, el, rf, el, rf, *(pi->_at[el][rf]));
    }
  }

  // Relax the at from its fanin.
  for(auto arc : pin._fanin) {
    arc->_fprop_at();
  }
}

// Procedure: _fprop_test
void Timer::_fprop_test(Pin& pin) {
  
  // reset tests
  for(auto test : pin._tests) {
    test->_reset();
  }
  
  // Obtain the rat
  if(!_clocks.empty()) {

    // Update the rat
    for(auto test : pin._tests) {
      // TODO: currently we assume a single clock...
      test->_fprop_rat(_clocks.begin()->second._period);
      
      // compute the cppr credit if any
      if(_cppr_analysis) {
        FOR_EACH_EL_RF_IF(el, rf, test->raw_slack(el, rf)) {
          test->_cppr_credit[el][rf] = _cppr_credit(*test, el, rf);
        }
      }
    }
  }
}

// Procedure: _bprop_rat
void Timer::_bprop_rat(Pin& pin) {

  pin._reset_rat();

  // PO
  if(auto po = pin.primary_output(); po) {
    FOR_EACH_EL_RF_IF(el, rf, po->_rat[el][rf]) {
      pin._relax_rat(nullptr, el, rf, el, rf, *(po->_rat[el][rf]));
    }
  }

  // Test
  for(auto test : pin._tests) {
    FOR_EACH_EL_RF_IF(el, rf, test->_rat[el][rf]) {
      if(test->_cppr_credit[el][rf]) {
        pin._relax_rat(
          &test->_arc, el, rf, el, rf, *test->_rat[el][rf] + *test->_cppr_credit[el][rf]
        );
      }
      else {
        pin._relax_rat(&test->_arc, el, rf, el, rf, *test->_rat[el][rf]);
      }
    }
  }

  // Relax the rat from its fanout.
  for(auto arc : pin._fanout) {
    arc->_bprop_rat();
  }
}

// Procedure: _build_fprop_cands
// Performs DFS to find all nodes in the fanout cone of frontiers.
void Timer::_build_fprop_cands(Pin& from) {
  
  assert(!from._has_state(Pin::FPROP_CAND) && !from._has_state(Pin::IN_FPROP_STACK));

  from._insert_state(Pin::FPROP_CAND | Pin::IN_FPROP_STACK);

  for(auto arc : from._fanout) {
    if(auto& to = arc->_to; !to._has_state(Pin::FPROP_CAND)) {
      _build_fprop_cands(to);
    }
    else if(to._has_state(Pin::IN_FPROP_STACK)) {
      _scc_analysis = true;
    }
  }
  
  _fprop_cands.push_front(&from);  // insert from front for scc traversal
  from._remove_state(Pin::IN_FPROP_STACK);
}

// Procedure: _build_bprop_cands
// Perform the DFS to find all nodes in the fanin cone of fprop candidates.
void Timer::_build_bprop_cands(Pin& to) {
  
  assert(!to._has_state(Pin::BPROP_CAND) && !to._has_state(Pin::IN_BPROP_STACK));

  to._insert_state(Pin::BPROP_CAND | Pin::IN_BPROP_STACK);

  // add pin to scc
  if(_scc_analysis && to._has_state(Pin::FPROP_CAND) && !to._scc) {
    _scc_cands.push_back(&to);
  }

  for(auto arc : to._fanin) {
    if(auto& from=arc->_from; !from._has_state(Pin::BPROP_CAND)) {
      _build_bprop_cands(from);
    }
  }
  
  _bprop_cands.push_front(&to);
  to._remove_state(Pin::IN_BPROP_STACK);
}

// Procedure: _build_prop_cands
void Timer::_build_prop_cands() {
  _prof::setup_timer("_build_prop_cands");

  _scc_analysis = false;

  // Discover all fprop candidates.
  for(const auto& ftr : _frontiers) {
    if(ftr->_has_state(Pin::FPROP_CAND)) {
      continue;
    }
    _build_fprop_cands(*ftr);
  }

  // Discover all bprop candidates.
  for(auto fcand : _fprop_cands) {

    if(fcand->_has_state(Pin::BPROP_CAND)) {
      continue;
    }

    _scc_cands.clear();
    _build_bprop_cands(*fcand);

    if(!_scc_analysis) {
      assert(_scc_cands.empty());
    }
    
    // here dfs returns with exacly one scc if exists
    if(auto& c = _scc_cands; c.size() >= 2 || (c.size() == 1 && c[0]->has_self_loop())) {
      auto& scc = _insert_scc(c);
      scc._unloop();
    }
  }
  
  _prof::stop_timer("_build_prop_cands");
}

void Timer::_build_rc_timing_tasks() {
  _prof::setup_timer("_build_rc_timing_tasks");
  // Emplace all rc timing tasks

  if(_has_state(CUDA_ENABLED)) {
    // Step 1: Allocate the space for FlatRct's
    auto& stor = _flat_rct_stor.emplace();
    
    stor.rct_nodes_start.reserve(_nets.size() + 1);
    
    int total_num_nodes = 0;
    int total_num_edges = 0; 
    int net_id = 0; 
    for(auto &p : _nets) {
      size_t sz = p.second._init_flat_rct(&stor, total_num_nodes, total_num_edges, net_id);
      if(!sz) continue;

      stor.rct_nodes_start.push_back(total_num_nodes);
      total_num_nodes += sz;
      total_num_edges += sz - 1; 
      net_id += 1; 
    }
    assert(total_num_edges + net_id == total_num_nodes);
    stor.rct_nodes_start.push_back(total_num_nodes);
    stor.total_num_nodes = total_num_nodes;
    stor.total_num_edges = total_num_edges; 
    stor.rct_edges.resize(total_num_edges); 
    stor.rct_edges_res.assign(total_num_edges, 0); 
    stor.rct_nodes_cap.assign(total_num_nodes*MAX_SPLIT_TRAN, 0); 
    stor.rct_roots.resize(_nets.size());
    stor.rct_node2bfs_order.resize(total_num_nodes);
    stor.rct_pinidx2id.assign(_pins.size(), -1);
    stor.rct_pid.resize(total_num_nodes);

    // Step 2: Create task for FlatRct make
    auto pf_pair = _taskflow.parallel_for(_nets.begin(), _nets.end(), [] (auto &p) {
        p.second._update_rc_timing_flat();
      }, 32);

    // Step 3: Create task for computing FlatRctStorage
    auto task_compute = _taskflow.emplace([this] () {
        _flat_rct_stor->_update_timing_cuda();
      });

    pf_pair.second.precede(task_compute);
    //task_init_omp.precede(task_compute);
  }
  else {
    _taskflow.parallel_for(_nets.begin(), _nets.end(), [] (auto &p) {
        p.second._update_rc_timing();
      }, 32);
  }

  _prof::stop_timer("_build_rc_timing_tasks");
}

// Procedure: _clear_prop_tasks
void Timer::_clear_rc_timing_tasks() {
  // no need to do anything
}

// Procedure: _build_prop_tasks_cuda
// do GPU-based levelization, and build level-wise parallel_for tasks
void Timer::_build_prop_tasks_cuda() {
  _prof::setup_timer("_build_prop_tasks_cuda");

  // new taskflow instance
  // to avoid the mix of "tasks that build tasks" with "tasks".
  tf::Taskflow _tf;
  tf::Executor _ex;

  // Step 1: init edgelist, degree and frontier arrays
  // this is for levelization, so we use fanin graph here.
  
  int n = _pins.size();
  int num_edges = std::numeric_limits<int>::max();

  FlatArcGraph &fanin_arc_graph = _prop_fanin_arc_graph.emplace();
  fanin_arc_graph.set_num_nodes(n);

  //////////// !!! CAUSION !!!! ///////////
  // Yibo: I try to make both toposort and prop_cuda use the same data 
  // such that the overhead for data copy can be minimized. 
  // Meanwhile, I try to add aynchronious copy with --default-stream=per-thread 
  // setting to NVCC. It is not yet verified if this works or not. 
  // The idea is that multiple streams will be used if cuda kernels are invoked inside a CPU thread. 
  // Maybe manually manage the streams will give better performance. 
  // This code is rather crappy for now. 
  // In the future, we shoudl have a better way to manage CPU and GPU data copies. 
  /////////////////////////////////////////
  _prof::setup_timer("bptc_memory_alloc");
  PropCUDA &prop_data_cpu = _prop_cuda_cpu.emplace();
  PropCUDA &prop_data_cuda = _prop_cuda_gpu.emplace();
  prop_data_cuda.init_device();

  // only used locally through levelization
  std::vector<int> fanout_degrees(n, 0);
  
  std::vector<int> &frontiers = _prop_frontiers.emplace(n, 0);
  std::vector<int> &frontiers_ends = _prop_frontiers_ends.emplace();

  std::vector<int> &arc2ftid = _prop_arc2ftid.emplace(_arcs.size() * (MAX_SPLIT_TRAN * MAX_TRAN), std::numeric_limits<int>::max());
  
  // for net arcs only 
  std::vector<ArcInfo> &arc_infos = _prop_net_arc_infos.emplace(_arcs.size()); 
  std::vector<float> &pin_loads = _prop_pin_loads.emplace(_pins.size() * MAX_SPLIT_TRAN, 0);
  std::vector<PinInfoCUDA> &pin_slews = _prop_pin_slews.emplace(_pins.size() * MAX_SPLIT_TRAN); 
  std::vector<PinInfoCUDA> &pin_ats = _prop_pin_ats.emplace(_pins.size() * MAX_SPLIT_TRAN); 
  
  _prof::stop_timer("bptc_memory_alloc");
  _prof::setup_timer("build_tasks_prepare");

  auto dummy_task1 = _tf.emplace([] () {
      _prof::setup_timer("fanin_count_edges_pair");
    });

  // count number of edges
  auto [fanin_count_edges_S, fanin_count_edges_T] = _tf.parallel_for(_fprop_cands.begin(), _fprop_cands.end(), [&] (Pin *pin) {
      int& szi = fanin_arc_graph.adjacency_list_start[pin->_idx + 1];
      for(auto arc: pin->_fanin) {
        if(!arc->_has_state(Arc::LOOP_BREAKER)
           && arc->_from._has_state(Pin::FPROP_CAND)) ++szi;
      }
      int &outi = fanout_degrees[pin->_idx];
      for(auto arc: pin->_fanout) {
        if(!arc->_has_state(Arc::LOOP_BREAKER)
           && arc->_to._has_state(Pin::FPROP_CAND)) ++outi;
      }
    }, 32);
  fanin_count_edges_S.succeed(dummy_task1);
  
  // sequential partial sum of fanin degree, for in-edge adjacency list
  auto fanin_prefix_sum = _tf.emplace([&](){
      _prof::stop_timer("fanin_count_edges_pair");
      _prof::setup_timer("fanin_prefix_sum");
      for(int i = 1; i <= n; ++i) {
        fanin_arc_graph.adjacency_list_start[i] += fanin_arc_graph.adjacency_list_start[i - 1];
      }
      num_edges = fanin_arc_graph.adjacency_list_start[n];
      fanin_arc_graph.adjacency_list.resize(num_edges);
      fanin_arc_graph.num_edges = num_edges; 
      _prof::stop_timer("fanin_prefix_sum");
      _prof::setup_timer("fanin_put_edges");
    });
  fanin_prefix_sum.succeed(fanin_count_edges_T);

  // put edges into edgelist
  auto [fanin_put_edges_S, fanin_put_edges_T] = _tf.parallel_for(_fprop_cands.begin(), _fprop_cands.end(), [&] (Pin *pin) {
      int st = fanin_arc_graph.adjacency_list_start[pin->_idx];
      for(auto arc: pin->_fanin) {
        if(!arc->_has_state(Arc::LOOP_BREAKER)
           && arc->_from._has_state(Pin::FPROP_CAND)) {
          // encode with arc type 
          fanin_arc_graph.adjacency_list[st++] = {(arc->idx() << 1) + arc->is_cell_arc(), arc->_from._idx};
        }
      }
    }, 32);
  fanin_put_edges_S.succeed(fanin_prefix_sum); 

  auto copy_fanin_graph = _tf.emplace([&](){
      _prof::stop_timer("fanin_put_edges");
      _prof::setup_timer("copy_fanin_graph");
      prop_data_cpu.fanin_graph.adjacency_list = fanin_arc_graph.adjacency_list.data();
      prop_data_cpu.fanin_graph.adjacency_list_start = fanin_arc_graph.adjacency_list_start.data();
      prop_data_cpu.fanin_graph.num_nodes = fanin_arc_graph.num_nodes;
      prop_data_cpu.fanin_graph.num_edges = fanin_arc_graph.num_edges;
      prop_data_cuda.copy_fanin_graph(prop_data_cpu.fanin_graph); 

      prop_data_cpu.fanout_degrees = fanout_degrees.data();
      prop_data_cuda.copy_fanout_degrees(fanout_degrees); 
      _prof::stop_timer("copy_fanin_graph");
    });
  copy_fanin_graph.succeed(fanin_put_edges_T);
  
  auto alloc_frontiers = _tf.emplace([&] () {
      prop_data_cpu.frontiers = frontiers.data();
      prop_data_cuda.alloc_frontiers(n);
    });

  // Step 2: call GPU function
  auto toposort = _tf.emplace([&](){
      _prof::setup_timer("toposort_compute_cuda");
      toposort_compute_cuda(
        prop_data_cpu, prop_data_cuda, 
        frontiers_ends);
      _prof::stop_timer("toposort_compute_cuda");
    });
  toposort.succeed(copy_fanin_graph, alloc_frontiers);

  auto copy_frontiers_ends = _tf.emplace([&](){
      prop_data_cpu.frontiers_ends = frontiers_ends.data();
      prop_data_cuda.copy_frontiers_ends(frontiers_ends);
    });
  copy_frontiers_ends.succeed(toposort);

  // prepare timing table and arc2ftid for propagation on GPU
  // Timing Graph here is constructed
  auto prepare_timing_table = _tf.emplace([&](){
      _prof::setup_timer("_flatten_liberty"); 
      _flatten_liberty();
      _prof::stop_timer("_flatten_liberty"); 
      _prof::setup_timer("_update_arc2ftid");
    });

  // copy content in flat table to prop_data_cpu
  auto copy_timing_table = _tf.emplace([&](){
      prop_data_cpu.slew_ft.num_tables = _ft.num_tables; 
      prop_data_cpu.slew_ft.total_num_xs = _ft.slew_indices1.size(); 
      prop_data_cpu.slew_ft.total_num_ys = _ft.slew_indices2.size(); 
      prop_data_cpu.slew_ft.total_num_data = _ft.slew_table.size(); 
      prop_data_cpu.slew_ft.xs = _ft.slew_indices1.data(); 
      prop_data_cpu.slew_ft.ys = _ft.slew_indices2.data(); 
      prop_data_cpu.slew_ft.data = _ft.slew_table.data(); 
      prop_data_cpu.slew_ft.xs_st = _ft.slew_indices1_start.data(); 
      prop_data_cpu.slew_ft.ys_st = _ft.slew_indices2_start.data(); 
      prop_data_cpu.slew_ft.data_st = _ft.slew_table_start.data(); 
      prop_data_cpu.delay_ft.num_tables = _ft.num_tables; 
      prop_data_cpu.delay_ft.total_num_xs = _ft.delay_indices1.size(); 
      prop_data_cpu.delay_ft.total_num_ys = _ft.delay_indices2.size(); 
      prop_data_cpu.delay_ft.total_num_data = _ft.delay_table.size(); 
      prop_data_cpu.delay_ft.xs = _ft.delay_indices1.data(); 
      prop_data_cpu.delay_ft.ys = _ft.delay_indices2.data(); 
      prop_data_cpu.delay_ft.data = _ft.delay_table.data(); 
      prop_data_cpu.delay_ft.xs_st = _ft.delay_indices1_start.data(); 
      prop_data_cpu.delay_ft.ys_st = _ft.delay_indices2_start.data(); 
      prop_data_cpu.delay_ft.data_st = _ft.delay_table_start.data(); 

      prop_data_cuda.copy_slew_ft(prop_data_cpu.slew_ft); 
      prop_data_cuda.copy_delay_ft(prop_data_cpu.delay_ft); 
    });
  copy_timing_table.succeed(prepare_timing_table);

  // map the id to each arc
  auto [arc2ftid_S, arc2ftid_T] = _tf.parallel_for(0, (int)_arcs.size(), 1, [&](int idx) {
      Arc const& arc = *_idx2arc[idx];
      FOR_EACH_EL_RF_RF(el, irf, orf) {
        if(auto tv = std::get_if<TimingView>(&arc._handle); tv) { // cell arc 
          const auto t = (*tv)[el];
          if(t != nullptr) {
            if(t->is_transition_defined(irf, orf)) {
              assert(_ft.t2ftid[el][irf][orf].find(t) != _ft.t2ftid[el][irf][orf].end());
              if ((orf == Tran::RISE && t->rise_transition) 
                  || (orf == Tran::FALL && t->fall_transition)) {
                int encode_id = arc.idx() * (MAX_SPLIT_TRAN * MAX_TRAN) + el * (MAX_TRAN * MAX_TRAN)
                  + irf * MAX_TRAN + orf; 
                arc2ftid.at(encode_id) = _ft.t2ftid[el][irf][orf].at(t);
              }
            }
            else {
              assert(_ft.t2ftid[el][irf][orf].find(t) == _ft.t2ftid[el][irf][orf].end());
            }
          }
        }
      }

    }, 32);
  arc2ftid_S.succeed(prepare_timing_table);

  auto copy_arc2ftid = _tf.emplace([&](){
      _prof::stop_timer("_update_arc2ftid");
      prop_data_cpu.arc2ftid = arc2ftid.data();
      prop_data_cuda.copy_arc2ftid(arc2ftid);
    });
  copy_arc2ftid.succeed(arc2ftid_T);

  auto [init_arcs_S, init_arcs_T] = _tf.parallel_for(0, (int)_arcs.size(), 1, [&] (int idx) {
      Arc const& arc = *_idx2arc[idx];
      std::visit(Functors{
        // Case 1: Net arc
        [&] (const Net* net) {
          auto& net_arc = arc_infos[arc.idx()].net_arc;
          FOR_EACH_EL_RF(el, rf) {
            net_arc.delays[el * MAX_TRAN + rf] = net->_delay(el, rf, arc._to).value();
            net_arc.impulses[el * MAX_TRAN + rf] = net->_impulse(el, rf, arc._to).value();
            pin_loads[arc._from.idx() * MAX_SPLIT_TRAN + el * MAX_TRAN + rf] = net->_load(el, rf);
          }
        },
        // Case 2: Cell arc
        [&] (TimingView tv) {
          auto& cell_arc = arc_infos[arc.idx()].cell_arc;
          std::fill(cell_arc.delays, cell_arc.delays + MAX_SPLIT_TRAN * MAX_TRAN, std::numeric_limits<float>::max());
        }
      }, arc._handle);
    }, 32);

  auto copy_pin_loads_arc_infos = _tf.emplace([&](){
      prop_data_cpu.pin_loads = pin_loads.data();
      prop_data_cuda.copy_pin_loads(pin_loads);
      prop_data_cpu.arc_infos = arc_infos.data();
      prop_data_cuda.copy_arc_infos(arc_infos);
    });
  copy_pin_loads_arc_infos.succeed(init_arcs_T);

  auto [init_pin_slews_S, init_pin_slews_T] = _tf.parallel_for(_fprop_cands.begin(), _fprop_cands.end(), [&] (Pin *pin) {
      // PI
      if(auto pi = pin->primary_input(); pi) {
        FOR_EACH_EL_RF_IF(el, rf, pi->_slew[el][rf]) {
          auto& pin_slew = pin_slews[pin->idx() * MAX_SPLIT_TRAN + el * MAX_TRAN + rf];
          auto& pin_at = pin_ats[pin->idx() * MAX_SPLIT_TRAN + el * MAX_TRAN + rf];
          pin_slew.value = *(pi->_slew[el][rf]);
          pin_at.value = 0;
        }
      }
      else {
        FOR_EACH_EL_RF(el, rf) {
          auto& pin_slew = pin_slews[pin->idx() * MAX_SPLIT_TRAN + el * MAX_TRAN + rf];
          auto& pin_at = pin_ats[pin->idx() * MAX_SPLIT_TRAN + el * MAX_TRAN + rf];
          if (el == Split::MAX) {
            pin_slew.value = std::numeric_limits<float>::lowest(); 
            pin_at.value = std::numeric_limits<float>::lowest(); 
          } else {
            pin_slew.value = std::numeric_limits<float>::max(); 
            pin_at.value = std::numeric_limits<float>::max(); 
          }
        }
      }
    }, 32);

  auto copy_pin_slews_ats = _tf.emplace([&](){
      prop_data_cpu.pin_slews = pin_slews.data(); 
      prop_data_cpu.pin_ats = pin_ats.data(); 
      prop_data_cuda.copy_pin_slews(pin_slews);
      prop_data_cuda.copy_pin_ats(pin_ats);
    });
  copy_pin_slews_ats.succeed(init_pin_slews_T);
  
  _prof::stop_timer("build_tasks_prepare");
  
  _prof::setup_timer("prop_prepare");
  _ex.run(_tf).wait();
  _tf.clear();
  _prof::stop_timer("prop_prepare");

  _prof::setup_timer("build_tasks_exec");

  auto run_prop = _taskflow.emplace([&](){
      prop_data_cpu.num_levels = frontiers_ends.size() - 1;
      prop_data_cpu.num_pins = _pins.size(); 
      prop_data_cpu.num_arcs = _arcs.size();

      prop_cuda(prop_data_cpu, prop_data_cuda);

      _prof::setup_timer("copy_slew_at/delay");
    });
  
  // run_prop.succeed(copy_timing_table, copy_arc2ftid, copy_pin_loads_arc_infos, copy_pin_slews_ats, copy_frontiers_ends);
  // // run_prop.succeed(fanout_put_edges_T);
  
  auto [copy_slew_at_S, copy_slew_at_T]  = _taskflow.parallel_for(0, (int)_pins.size(), 1, [&](int idx) {
      Pin *pin = _idx2pin[idx];
      FOR_EACH_EL_RF(el, rf) {
        int offset = pin->idx() * MAX_SPLIT_TRAN + el * MAX_SPLIT + rf;
        PinInfoCUDA const& pin_slew = pin_slews[offset]; 
        PinInfoCUDA const& pin_at = pin_ats[offset]; 
        pin->_slew[el][rf].emplace(_idx2arc[pin_slew.from_arcidx], (Split)pin_slew.from_el, (Tran)pin_slew.from_rf, pin_slew.value);
        pin->_at[el][rf].emplace(_idx2arc[pin_at.from_arcidx], (Split)pin_at.from_el, (Tran)pin_at.from_rf, pin_at.value);
      }
    }, 32);
  copy_slew_at_S.succeed(run_prop);

  auto [copy_delay_S, copy_delay_T]  = _taskflow.parallel_for(0, (int)_arcs.size(), 1, [&](int idx) {
      Arc *arc = _idx2arc[idx];
      if (arc->is_cell_arc()) {
        auto& cell_arc = arc_infos[arc->idx()].cell_arc;
        FOR_EACH_EL_RF_RF(el, frf, trf) {
          auto value = cell_arc.delays[el * MAX_TRAN * MAX_TRAN + frf * MAX_TRAN + trf]; 
          if (value != std::numeric_limits<float>::max()) {
            arc->_delay[el][frf][trf] = value;
          }
        }
      }
      else {
        auto& net_arc = arc_infos[arc->idx()].net_arc; 
        FOR_EACH_EL_RF(el, rf) {
          arc->_delay[el][rf][rf] = net_arc.delays[el * MAX_TRAN + rf];
        }
      }
    }, 32);
  copy_delay_S.succeed(run_prop);

  auto dummy_endcopy = _taskflow.emplace([] () {
      _prof::stop_timer("copy_slew_at/delay");
    });
  dummy_endcopy.succeed(copy_slew_at_T, copy_delay_T);

  // Step 3: build remaining tasks for forward prop, and all tasks for backward prop
  tf::Task last = dummy_endcopy;

  auto dummy_task2 = _taskflow.emplace([] () {
      _prof::setup_timer("prop_forward_remaining");
    });
  dummy_task2.succeed(last);
  last = dummy_task2;

  // forward
  for(int i = (int)frontiers_ends.size() - 2; i >= 0; --i) {
    int l = frontiers_ends[i], r = frontiers_ends[i + 1];
    auto [S, T] = _taskflow.parallel_for(frontiers.begin() + l, frontiers.begin() + r, [this] (int idx) {
        Pin *pin = _idx2pin[idx];
        //_fprop_slew(*pin);
        //_fprop_delay(*pin);
        //_fprop_at(*pin);
        _fprop_test(*pin);
      }, 32);
    last.precede(S);
    last = T;
  }

  auto dummy_task3 = _taskflow.emplace([] () {
      _prof::stop_timer("prop_forward_remaining");
      _prof::setup_timer("prop_backward");
    });
  dummy_task3.succeed(last);
  last = dummy_task3;

  // backward
  for(int i = 0; i < (int)frontiers_ends.size() - 1; ++i) {
    int l = frontiers_ends[i], r = frontiers_ends[i + 1];
    auto [S, T] = _taskflow.parallel_for(frontiers.begin() + l, frontiers.begin() + r, [this] (int idx) {
        Pin *pin = _idx2pin[idx];
        _bprop_rat(*pin);
      }, 32);
    last.precede(S);
    last = T;
  }
  
  auto dummy_task4 = _taskflow.emplace([] () {
      _prof::stop_timer("prop_backward");
    });
  dummy_task4.succeed(last);
  last = dummy_task4;

  // cleanup
  auto task_cleanup = _taskflow.emplace([this] () {
      //std::vector<int> &frontiers = *_prop_frontiers;
      //std::vector<int> &frontiers_ends = *_prop_frontiers_ends;
      //for(int i = (int)frontiers_ends.size() - 2; i >= 0; --i) {
      //  int l = frontiers_ends[i], r = frontiers_ends[i + 1];
      //  for (int j = l; j < r; ++j) {
      //      auto pin = _idx2pin[frontiers[j]]; 
      //      FOR_EACH_EL_RF(el, rf) {
      //          printf("golden[%lu][%d][%d] slew %.6f, at %.6f\n", pin->idx(), el, rf, 
      //                  (pin->slew(el, rf))? pin->slew(el, rf).value() : std::numeric_limits<float>::max(), 
      //                  (pin->at(el, rf))? pin->at(el, rf).value() : std::numeric_limits<float>::max());
      //      }
      //  }
      //}

      //for (auto const& kvp : _pins) {
      //  auto const& pin = kvp.second;
      //  FOR_EACH_EL_RF(el, rf) {
      //      printf("golden[%lu][%d][%d] slew %.6f, at %.6f\n", pin.idx(), el, rf, 
      //              (pin.slew(el, rf))? pin.slew(el, rf).value() : std::numeric_limits<float>::max(), 
      //              (pin.at(el, rf))? pin.at(el, rf).value() : std::numeric_limits<float>::max());
      //  }
      //}

      _prof::setup_timer("cleanup");
      _prop_cuda_cpu.reset();
      //_prop_cuda_gpu->destroy_device();
      _prop_cuda_gpu.reset();
      _prop_fanin_arc_graph.reset();
      
      _prop_frontiers.reset();
      _prop_frontiers_ends.reset();

      _prop_arc2ftid.reset();
      _prop_net_arc_infos.reset();
      _prop_pin_loads.reset();
      _prop_pin_slews.reset();
      _prop_pin_ats.reset();
      _prof::stop_timer("cleanup");

    });
  last.precede(task_cleanup);
  
  _prof::stop_timer("build_tasks_exec");
  _prof::stop_timer("_build_prop_tasks_cuda");
}

// Procedure: _run_prop_tasks_sequential
void Timer::_run_prop_tasks_sequential() {
  _prof::setup_timer("_run_prop_tasks_sequential");
  _prof::setup_timer("_run_prop_tasks_sequential__bfsseq");
  std::vector<int> in(_pins.size(), 0);
  std::vector<Pin*> seq;
  seq.reserve(_fprop_cands.size());
  for(auto to: _fprop_cands) {
    int &tin = in[to->_idx];
    for(auto arc: to->_fanin) {
      if(!arc->_has_state(Arc::LOOP_BREAKER)
         && arc->_from._has_state(Pin::FPROP_CAND)) ++tin;
    }
    if(!tin) seq.push_back(to);
  }
  for(size_t i = 0; i < seq.size(); ++i) {
    Pin *from = seq[i];
    for(auto arc: from->_fanout) {
      if(!arc->_has_state(Arc::LOOP_BREAKER)
         && arc->_to._has_state(Pin::FPROP_CAND)) {
        if(!--in[arc->_to._idx]) seq.push_back(&(arc->_to));
      }
    }
  }
  _prof::stop_timer("_run_prop_tasks_sequential__bfsseq");
  _prof::setup_timer("_run_prop_tasks_sequential__forward");
  for(size_t i = 0; i < seq.size(); ++i) {
    Pin *pin = seq[i];
    _fprop_slew(*pin);
    _fprop_delay(*pin);
    _fprop_at(*pin);
    _fprop_test(*pin);
  }
  _prof::stop_timer("_run_prop_tasks_sequential__forward");
  _prof::setup_timer("_run_prop_tasks_sequential__backward");
  for(int i = seq.size() - 1; i >= 0; --i) {
    Pin *pin = seq[i];
    _bprop_rat(*pin);
  }
  _prof::stop_timer("_run_prop_tasks_sequential__backward");
  _prof::stop_timer("_run_prop_tasks_sequential");
}

// Procedure: _build_prop_tasks
void Timer::_build_prop_tasks() {
  _prof::setup_timer("_build_prop_tasks");
  
  // Emplace the fprop task
  // (1) propagate the rc timing
  // (2) propagate the slew 
  // (3) propagate the delay
  // (4) propagate the arrival time.
  for(auto pin : _fprop_cands) {
    assert(!pin->_ftask);
    pin->_ftask = _taskflow.emplace([this, pin] () {
      _fprop_slew(*pin);
      _fprop_delay(*pin);
      _fprop_at(*pin);
      _fprop_test(*pin);
    });
  }
  
  // Build the dependency
  for(auto to : _fprop_cands) {
    for(auto arc : to->_fanin) {
      if(arc->_has_state(Arc::LOOP_BREAKER)) {
        continue;
      }
      if(auto& from = arc->_from; from._has_state(Pin::FPROP_CAND)) {
        from._ftask->precede(to->_ftask.value());
      }
    }
  }

  // Emplace the bprop task
  // (1) propagate the required arrival time
  for(auto pin : _bprop_cands) {
    assert(!pin->_btask);
    pin->_btask = _taskflow.emplace([this, pin] () {
      _bprop_rat(*pin);
    });
  }

  // Build the task dependencies.
  for(auto to : _bprop_cands) {
    for(auto arc : to->_fanin) {
      if(arc->_has_state(Arc::LOOP_BREAKER)) {
        continue;
      }
      if(auto& from = arc->_from; from._has_state(Pin::BPROP_CAND)) {
        to->_btask->precede(from._btask.value());
      }
    } 
  }

  // Connect with ftasks
  for(auto pin : _bprop_cands) {
    if(pin->_btask->num_dependents() == 0 && pin->_ftask) {
      pin->_ftask->precede(pin->_btask.value()); 
    }
  }

  _prof::stop_timer("_build_prop_tasks");
}

// Procedure: _clear_prop_tasks
void Timer::_clear_prop_tasks() {
  
  // fprop is a subset of bprop
  for(auto pin : _bprop_cands) {
    pin->_ftask.reset();
    pin->_btask.reset();
    pin->_remove_state();
  }

  _fprop_cands.clear();
  _bprop_cands.clear();
}

// Procedure: _flatten_liberty
void Timer::_flatten_liberty() {

  // here we clear all data to regenerate the data
  _ft.num_tables = 0;
  FOR_EACH_EL_RF_RF(el, irf, orf) {
    _ft.t2ftid[el][irf][orf].clear();
  }
  _ft.slew_indices1.clear();
  _ft.slew_indices2.clear();
  _ft.slew_table.clear();
  _ft.slew_indices1_start.clear();
  _ft.slew_indices2_start.clear();
  _ft.slew_table_start.clear();
  _ft.delay_indices1.clear();
  _ft.delay_indices2.clear();
  _ft.delay_table.clear();
  _ft.delay_indices1_start.clear();
  _ft.delay_indices2_start.clear();
  _ft.delay_table_start.clear();
  std::size_t slew_indices1_size = 0; 
  std::size_t slew_indices2_size = 0; 
  std::size_t slew_table_size = 0; 
  std::size_t delay_indices1_size = 0; 
  std::size_t delay_indices2_size = 0; 
  std::size_t delay_table_size = 0; 

  OT_LOGI("flattening celllib timing arcs ...");

  // calculate slew length and delay length
  FOR_EACH_EL_IF(el, _celllib[el]) {
    
    auto& celllib = *_celllib[el];
    for(const auto& cell : celllib.cells) {
      for(const auto& cp : cell.second.cellpins) {
        for(const auto& t : cp.second.timings) {
          FOR_EACH_RF_RF_IF(irf, orf, t.is_transition_defined(irf, orf)) {
            
            assert(_ft.t2ftid[el][irf][orf].find(&t) == _ft.t2ftid[el][irf][orf].end());
            auto& t2ftid = _ft.t2ftid[el][irf][orf];
            t2ftid[&t] = _ft.num_tables++;

            const Lut* slut {nullptr};
            const Lut* dlut {nullptr};
            
            // slew and delay
            switch(orf) {
              case RISE:
                slut = t.rise_transition ? &(t.rise_transition.value()) : nullptr;
                dlut = t.cell_rise ? &(t.cell_rise.value()) : nullptr;
              break;

              case FALL:
                slut = t.fall_transition ? &(t.fall_transition.value()) : nullptr;
                dlut = t.cell_fall ? &(t.cell_fall.value()) : nullptr;
              break;
            }

            if(slut != nullptr) {
              slew_indices1_size += slut->indices1.size(); 
              slew_indices2_size += slut->indices2.size(); 
              slew_table_size += slut->table.size();
            }

            if(dlut != nullptr) {
              delay_indices1_size += dlut->indices1.size(); 
              delay_indices2_size += dlut->indices2.size(); 
              delay_table_size += dlut->table.size();
            }
          }
        }
      }
    }
  }

  // flatten the table

  // must initialize all to zero 
  _ft.slew_indices1_start.assign(_ft.num_tables+1, 0);
  _ft.slew_indices2_start.assign(_ft.num_tables+1, 0);
  _ft.slew_table_start.assign(_ft.num_tables+1, 0);
  _ft.delay_indices1_start.assign(_ft.num_tables+1, 0);
  _ft.delay_indices2_start.assign(_ft.num_tables+1, 0);
  _ft.delay_table_start.assign(_ft.num_tables+1, 0);

  _ft.slew_indices1.reserve(slew_indices1_size);  
  _ft.slew_indices2.reserve(slew_indices2_size);  
  _ft.slew_table.reserve(slew_table_size);           
  _ft.delay_indices1.reserve(delay_indices1_size);  
  _ft.delay_indices2.reserve(delay_indices2_size);  
  _ft.delay_table.reserve(delay_table_size);           

  FOR_EACH_EL_IF(el, _celllib[el]) {
    
    auto& celllib = *_celllib[el];
    for(const auto& cell : celllib.cells) {

      for(const auto& cp : cell.second.cellpins) {
        for(const auto& t : cp.second.timings) {
          FOR_EACH_RF_RF_IF(irf, orf, t.is_transition_defined(irf, orf)) {
            
            unsigned table_id = _ft.t2ftid[el][irf][orf].at(&t);

            const Lut* slut {nullptr};
            const Lut* dlut {nullptr};
            
            // slew and delay
            switch(orf) {
              case RISE:
                slut = t.rise_transition ? &(t.rise_transition.value()) : nullptr;
                dlut = t.cell_rise ? &(t.cell_rise.value()) : nullptr;
              break;

              case FALL:
                slut = t.fall_transition ? &(t.fall_transition.value()) : nullptr;
                dlut = t.cell_fall ? &(t.cell_fall.value()) : nullptr;
              break;
            }

            if(slut != nullptr) {
              _ft.slew_indices1.insert(_ft.slew_indices1.end(), slut->indices1.begin(), slut->indices1.end());
              _ft.slew_indices2.insert(_ft.slew_indices2.end(), slut->indices2.begin(), slut->indices2.end());
              _ft.slew_table.insert(_ft.slew_table.end(), slut->table.begin(), slut->table.end());
              // temporarily record the table sizes 
              _ft.slew_indices1_start [table_id + 1] = slut->indices1.size();
              _ft.slew_indices2_start [table_id + 1] = slut->indices2.size();
              _ft.slew_table_start    [table_id + 1] = slut->table.size();
            }

            if(dlut != nullptr) {
              _ft.delay_indices1.insert(_ft.delay_indices1.end(), dlut->indices1.begin(), dlut->indices1.end());
              _ft.delay_indices2.insert(_ft.delay_indices2.end(), dlut->indices2.begin(), dlut->indices2.end());
              _ft.delay_table.insert(_ft.delay_table.end(), dlut->table.begin(), dlut->table.end());
              // temporarily record the table sizes 
              _ft.delay_indices1_start[table_id + 1] = dlut->indices1.size();
              _ft.delay_indices2_start[table_id + 1] = dlut->indices2.size();
              _ft.delay_table_start   [table_id + 1] = dlut->table.size();
            }
          }
        }
      }
    }
  }
    
  // scan to compute the prefix sum, which is the offset 
  for(int i = 1; i <= _ft.num_tables; ++i) {
    _ft.slew_indices1_start[i]  += _ft.slew_indices1_start[i-1];
    _ft.slew_indices2_start[i]  += _ft.slew_indices2_start[i-1];
    _ft.slew_table_start[i]     += _ft.slew_table_start[i-1];
    _ft.delay_indices1_start[i] += _ft.delay_indices1_start[i-1];
    _ft.delay_indices2_start[i] += _ft.delay_indices2_start[i-1];
    _ft.delay_table_start[i]    += _ft.delay_table_start[i-1];
  }
  
  OT_LOGI(_ft.num_tables, " celllib timing arcs flattened");
}

//void Timer::_update_arc2ftid(std::vector<int>& arc2ftid) {
//
//  arc2ftid.assign(_arcs.size() * (MAX_SPLIT_TRAN * MAX_TRAN), std::numeric_limits<int>::max());
//  auto encode = [](int arc_id, int el, int irf, int orf) {
//      return arc_id * (MAX_SPLIT_TRAN * MAX_TRAN) + el * (MAX_TRAN * MAX_TRAN)
//          + irf * MAX_TRAN + orf; 
//  };
//
//  // map the id to each arc
//  for(auto& arc : _arcs) {
//    //std::cout << "arc " << arc._from._name << "->" << arc._to._name << " flat table mapping:\n";
//    FOR_EACH_EL_RF_RF(el, irf, orf) {
//      //std::cout << "  " << to_string(el) << ' ' << to_string(irf) << ' ' << to_string(orf) << ' ';
//      if(auto tv = std::get_if<TimingView>(&arc._handle); tv) { // cell arc 
//        const auto t = (*tv)[el];
//        if(t != nullptr) {
//          if(t->is_transition_defined(irf, orf)) {
//            assert(_ft.t2ftid[el][irf][orf].find(t) != _ft.t2ftid[el][irf][orf].end());
//            if ((orf == Tran::RISE && t->rise_transition) 
//                    || (orf == Tran::FALL && t->fall_transition)) {
//                arc2ftid.at(encode(arc.idx(), el, irf, orf)) = _ft.t2ftid[el][irf][orf].at(t);
//            }
//          }
//          else {
//            assert(_ft.t2ftid[el][irf][orf].find(t) == _ft.t2ftid[el][irf][orf].end());
//          }
//        }
//      }
//    }
//  }
//}

// Function: update_timing
// Perform comprehensive timing update: 
// (1) grpah-based timing (GBA)
// (2) path-based timing (PBA)
void Timer::update_timing() {
  std::scoped_lock lock(_mutex);
  _update_timing();
}

// Function: _update_timing
void Timer::_update_timing() {
  _prof::setup_timer("_update_timing");
  
  // Timing is update-to-date
  if(!_lineage) {
    assert(_frontiers.size() == 0);
    return;
  }

  // materialize the lineage
  _prof::setup_timer("_update_timing__taskflow_read");
  _executor.run(_taskflow).wait();
  _prof::stop_timer("_update_timing__taskflow_read");
  _taskflow.clear();
  _lineage.reset();

  // Check if full update is required
  if(_has_state(FULL_TIMING)) {
    OT_LOGI("full timing");
    _insert_full_timing_frontiers();
  }

  // explore propagation candidates
  _build_prop_cands();
  
  _prof::setup_timer("_update_timing__taskflow_frontier");
  _executor.run(_taskflow).wait();
  _prof::stop_timer("_update_timing__taskflow_frontier");
  
  _taskflow.clear();

  // build rc timing tasks.
  _build_rc_timing_tasks();
  
  _prof::setup_timer("_update_timing__taskflow_rctiming");
  _prof::init_tasktimers();
  _executor.run(_taskflow).wait();
  _prof::finalize_tasktimers();
  _prof::stop_timer("_update_timing__taskflow_rctiming");

  _taskflow.clear();

  // clear the rc timing tasks
  _clear_rc_timing_tasks();

  // debug the rc timing result
  // for(auto &[s, net] : _nets) {
  //   std::cout << "Net " << s << std::endl;
  //   Rct *r1 = std::get_if<Rct>(&net._rct);
  //   FlatRct *r2 = std::get_if<FlatRct>(&net._rct);
  //   if(r1) {
  //     for(auto const &[name, node] : r1->_nodes) {
  //       std::cout << "Rct: " << name << ' ' << node.cap((Split)0, (Tran)0) << ' ' << node._load[0][0] << ' ' << node._delay[0][0] << ' ' << node._ldelay[0][0] << ' ' << node._impulse[0][0] << std::endl;
  //     }
  //   }
  //   if(r2) {
  //     for(auto const &[name, id] : r2->name2id) {
  //       int o = (r2->arr_start + r2->bfs_reverse_order_map[id]) * 4;
  //       auto const &st = *(r2->_stor);
  //       std::cout << "FlatRct: " << name << ' ' << st.cap[o] << ' ' << st.load[o] << ' ' << st.delay[o] << ' ' << st.ldelay[o] << ' ' << st.impulse[o] << std::endl;
  //     }
  //   }
  // }

  // build propagation tasks
  if(_has_state(CUDA_ENABLED)) {
    _build_prop_tasks_cuda();
  }
  else {
    _build_prop_tasks();
  }

  // debug the graph
  //_taskflow.dump(std::cout);

  // Execute the task
  _prof::setup_timer("_update_timing__taskflow_prop");
  _executor.run(_taskflow).wait();
  _prof::stop_timer("_update_timing__taskflow_prop");
  _taskflow.clear();

  // Clear the propagation tasks.
  _clear_prop_tasks();

  // Clear frontiers
  _clear_frontiers();

  // clear the state
  _remove_state();
  _prof::stop_timer("_update_timing");
}

// Procedure: _update_area
void Timer::_update_area() {
  
  _update_timing();

  if(_has_state(AREA_UPDATED)) {
    return;
  }
  
  _area = 0.0f;

  for(const auto& kvp : _gates) {
    if(const auto& c = kvp.second._cell[MIN]; c->area) {
      _area = *_area + *c->area;
    }
    else {
      OT_LOGE("cell ", c->name, " has no area defined");
      _area.reset();
      break;
    }
  }

  _insert_state(AREA_UPDATED);
}

// Procedure: _update_power
void Timer::_update_power() {

  _update_timing();

  if(_has_state(POWER_UPDATED)) {
    return;
  }

  // Update the static leakage power
  _leakage_power = 0.0f;
  
  for(const auto& kvp : _gates) {
    if(const auto& c = kvp.second._cell[MIN]; c->leakage_power) {
      _leakage_power = *_leakage_power + *c->leakage_power;
    }
    else {
      OT_LOGE("cell ", c->name, " has no leakage_power defined");
      _leakage_power.reset();
      break;
    }
  }

  _insert_state(POWER_UPDATED);
}

// Procedure: _update_endpoints
void Timer::_update_endpoints() {

  _update_timing();

  if(_has_state(EPTS_UPDATED)) {
    return;
  }

  // reset the storage and build task
  FOR_EACH_EL_RF(el, rf) {

    _endpoints[el][rf].clear();
    
    _taskflow.emplace([this, el=el, rf=rf] () {

      // for each po
      for(auto& po : _pos) {
        if(po.second.slack(el, rf).has_value()) {
          _endpoints[el][rf].emplace_back(el, rf, po.second);
        }
      }

      // for each test
      for(auto& test : _tests) {
        if(test.slack(el, rf).has_value()) {
          _endpoints[el][rf].emplace_back(el, rf, test);
        }
      }
      
      // sort endpoints
      std::sort(_endpoints[el][rf].begin(), _endpoints[el][rf].end());

      // update the worst negative slack (wns)
      if(!_endpoints[el][rf].empty()) {
        _wns[el][rf] = _endpoints[el][rf].front().slack();
      }
      else {
        _wns[el][rf] = std::nullopt;
      }

      // update the tns, and fep
      if(!_endpoints[el][rf].empty()) {
        _tns[el][rf] = 0.0f;
        _fep[el][rf] = 0;
        for(const auto& ept : _endpoints[el][rf]) {
          if(auto slack = ept.slack(); slack < 0.0f) {
            _tns[el][rf] = *_tns[el][rf] + slack;
            _fep[el][rf] = *_fep[el][rf] + 1; 
          }
        }
      }
      else {
        _tns[el][rf] = std::nullopt;
        _fep[el][rf] = std::nullopt;
      }
    });
  }

  // run tasks
  _executor.run(_taskflow).wait();
  _taskflow.clear();

  _insert_state(EPTS_UPDATED);
}

// Function: tns
// Update the total negative slack for any transition and timing split. The procedure applies
// the parallel reduction to compute the value.
std::optional<float> Timer::report_tns(std::optional<Split> el, std::optional<Tran> rf) {

  std::scoped_lock lock(_mutex);

  _update_endpoints();

  std::optional<float> v;

  if(!el && !rf) {
    FOR_EACH_EL_RF_IF(s, t, _tns[s][t]) {
      v = !v ? _tns[s][t] : *v + *(_tns[s][t]);
    }
  }
  else if(el && !rf) {
    FOR_EACH_RF_IF(t, _tns[*el][t]) {
      v = !v ? _tns[*el][t] : *v + *(_tns[*el][t]);
    }
  }
  else if(!el && rf) {
    FOR_EACH_EL_IF(s, _tns[s][*rf]) {
      v = !v ? _tns[s][*rf] : *v + *(_tns[s][*rf]);
    }
  }
  else {
    v = _tns[*el][*rf];
  }

  return v;
}

// Function: wns
// Update the total negative slack for any transition and timing split. The procedure apply
// the parallel reduction to compute the value.
std::optional<float> Timer::report_wns(std::optional<Split> el, std::optional<Tran> rf) {

  std::scoped_lock lock(_mutex);

  _update_endpoints();

  std::optional<float> v;
  
  if(!el && !rf) {
    FOR_EACH_EL_RF_IF(s, t, _wns[s][t]) {
      v = !v ? _wns[s][t] : std::min(*v, *(_wns[s][t]));
    }
  }
  else if(el && !rf) {
    FOR_EACH_RF_IF(t, _wns[*el][t]) {
      v = !v ? _wns[*el][t] : std::min(*v, *(_wns[*el][t]));
    }
  }
  else if(!el && rf) {
    FOR_EACH_EL_IF(s, _wns[s][*rf]) {
      v = !v ? _wns[s][*rf] : std::min(*v, *(_wns[s][*rf]));
    }
  }
  else {
    v = _wns[*el][*rf];
  }

  return v;
}

// Function: fep
// Update the failing end points
std::optional<size_t> Timer::report_fep(std::optional<Split> el, std::optional<Tran> rf) {
  
  std::scoped_lock lock(_mutex);

  _update_endpoints();

  std::optional<size_t> v;

  if(!el && !rf) {
    FOR_EACH_EL_RF_IF(s, t, _fep[s][t]) {
      v = !v ? _fep[s][t] : *v + *(_fep[s][t]);
    }
  }
  else if(el && !rf) {
    FOR_EACH_RF_IF(t, _fep[*el][t]) {
      v = !v ? _fep[*el][t] : *v + *(_fep[*el][t]);
    }
  }
  else if(!el && rf) {
    FOR_EACH_EL_IF(s, _fep[s][*rf]) {
      v = !v ? _fep[s][*rf] : *v + *(_fep[s][*rf]);
    }
  }
  else {
    v = _fep[*el][*rf];
  }

  return v;
}

// Function: leakage_power
std::optional<float> Timer::report_leakage_power() {
  std::scoped_lock lock(_mutex);
  _update_power();
  return _leakage_power;
}

// Function: area
// Sum up the area of each gate in the design.
std::optional<float> Timer::report_area() {
  std::scoped_lock lock(_mutex);
  _update_area();
  return _area;
}
    
// Procedure: _enable_full_timing_update
void Timer::_enable_full_timing_update() {
  _insert_state(FULL_TIMING);
}

// Procedure: _insert_full_timing_frontiers
void Timer::_insert_full_timing_frontiers() {

  // insert all zero-fanin pins to the frontier list
  for(auto& kvp : _pins) {
    _insert_frontier(kvp.second);
  }

  // clear the rc-net update flag
  for(auto& kvp : _nets) {
    kvp.second._rc_timing_updated = false;
  }
}

// Procedure: _insert_frontier
void Timer::_insert_frontier(Pin& pin) {
  
  if(pin._frontier_satellite) {
    return;
  }

  pin._frontier_satellite = _frontiers.insert(_frontiers.end(), &pin);
  
  // reset the scc.
  if(pin._scc) {
    _remove_scc(*pin._scc);
  }
}

// Procedure: _remove_frontier
void Timer::_remove_frontier(Pin& pin) {
  if(pin._frontier_satellite) {
    _frontiers.erase(*pin._frontier_satellite);
    pin._frontier_satellite.reset();
  }
}

// Procedure: _clear_frontiers
void Timer::_clear_frontiers() {
  for(auto& ftr : _frontiers) {
    ftr->_frontier_satellite.reset();
  }
  _frontiers.clear();
}

// Procedure: _insert_scc
SCC& Timer::_insert_scc(std::vector<Pin*>& cands) {
  
  // create scc only of size at least two
  auto& scc = _sccs.emplace_front(std::move(cands));
  scc._satellite = _sccs.begin();

  return scc;
}

// Procedure: _remove_scc
void Timer::_remove_scc(SCC& scc) {
  assert(scc._satellite);
  scc._clear();
  _sccs.erase(*scc._satellite); 
}

// Function: report_at   
// Report the arrival time in picoseconds at a given pin name.
std::optional<float> Timer::report_at(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_at(name, m, t);
}

// Function: _report_at
std::optional<float> Timer::_report_at(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(name); itr != _pins.end() && itr->second._at[m][t]) {
    return itr->second._at[m][t]->numeric;
  }
  else return std::nullopt;
}

// Function: report_rat
// Report the required arrival time in picoseconds at a given pin name.
std::optional<float> Timer::report_rat(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_rat(name, m, t);
}

// Function: _report_rat
std::optional<float> Timer::_report_rat(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(name); itr != _pins.end() && itr->second._at[m][t]) {
    return itr->second._rat[m][t];
  }
  else return std::nullopt;
}

// Function: report_slew
// Report the slew in picoseconds at a given pin name.
std::optional<float> Timer::report_slew(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_slew(name, m, t);
}

// Function: _report_slew
std::optional<float> Timer::_report_slew(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(name); itr != _pins.end() && itr->second._slew[m][t]) {
    return itr->second._slew[m][t]->numeric;
  }
  else return std::nullopt;
}

// Function: report_slack
std::optional<float> Timer::report_slack(const std::string& pin, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_slack(pin, m, t);
}

// Function: _report_slack
std::optional<float> Timer::_report_slack(const std::string& pin, Split m, Tran t) {
  _update_timing();
  if(auto itr = _pins.find(pin); itr != _pins.end()) {
    return itr->second.slack(m, t);
  }
  else return std::nullopt;
}

// Function: report_load
// Report the load at a given pin name
std::optional<float> Timer::report_load(const std::string& name, Split m, Tran t) {
  std::scoped_lock lock(_mutex);
  return _report_load(name, m, t);
}

// Function: _report_load
std::optional<float> Timer::_report_load(const std::string& name, Split m, Tran t) {
  _update_timing();
  if(auto itr = _nets.find(name); itr != _nets.end()) {
    return itr->second._load(m, t);
  }
  else return std::nullopt;
}

// Function: set_at
Timer& Timer::set_at(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);

  auto task = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pis.find(name); itr != _pis.end()) {
      _set_at(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set at (PI ", name, " not found)");
    }
  });

  _add_to_lineage(task);

  return *this;
}

// Procedure: _set_at
void Timer::_set_at(PrimaryInput& pi, Split m, Tran t, std::optional<float> v) {
  pi._at[m][t] = v;
  _insert_frontier(pi._pin);
}

// Function: set_rat
Timer& Timer::set_rat(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);
  
  auto op = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pos.find(name); itr != _pos.end()) {
      _set_rat(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set rat (PO ", name, " not found)");
    }
  });

  _add_to_lineage(op);

  return *this;
}

// Procedure: _set_rat
void Timer::_set_rat(PrimaryOutput& po, Split m, Tran t, std::optional<float> v) {
  po._rat[m][t] = v;
  _insert_frontier(po._pin);
}

// Function: set_slew
Timer& Timer::set_slew(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);
  
  auto task = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pis.find(name); itr != _pis.end()) {
      _set_slew(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set slew (PI ", name, " not found)");
    }
  });

  _add_to_lineage(task);

  return *this;
}

// Procedure: _set_slew
void Timer::_set_slew(PrimaryInput& pi, Split m, Tran t, std::optional<float> v) {
  pi._slew[m][t] = v;
  _insert_frontier(pi._pin);
}

// Function: set_load
Timer& Timer::set_load(std::string name, Split m, Tran t, std::optional<float> v) {

  std::scoped_lock lock(_mutex);
  
  auto task = _taskflow.emplace([this, name=std::move(name), m, t, v] () {
    if(auto itr = _pos.find(name); itr != _pos.end()) {
      _set_load(itr->second, m, t, v);
    }
    else {
      OT_LOGE("can't set load (PO ", name, " not found)");
    }
  });

  _add_to_lineage(task);

  return *this;
}

// Procedure: _set_load
void Timer::_set_load(PrimaryOutput& po, Split m, Tran t, std::optional<float> v) {

  po._load[m][t] = v ? *v : 0.0f;

  // Update the net load
  if(auto net = po._pin._net) {
    net->_rc_timing_updated = false;
  }
  
  // Enable the timing propagation.
  for(auto arc : po._pin._fanin) {
    _insert_frontier(arc->_from);
  }
  _insert_frontier(po._pin);
}


};  // end of namespace ot. -----------------------------------------------------------------------




