#include <ot/timer/timer.hpp>
#include <ot/timer/_prof.hpp>
#include <ot/cuda/toposort.cuh>

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

          // Then the correponding net becomes dirty
          if(auto net = pin->_net; net) {
            _insert_modified_net(*net);
          }
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

  // Then this net becomes dirty
  _insert_modified_net(net);

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
  
  // Then this net becomes dirty
  _insert_modified_net(*net);
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
  Net &net = _nets.try_emplace(name, name).first->second;
  _insert_modified_net(net);
  return net;
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
  _remove_modified_net(net);

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
    if(!_flat_rct_stor) {
      _flat_rct_stor.emplace();
    }
    auto& stor = *_flat_rct_stor;

    stor.pinload.resize(_pins.size() * MAX_SPLIT_TRAN);
    stor.pindelay.resize(_pins.size() * MAX_SPLIT_TRAN);
    stor.pinimpulse.resize(_pins.size() * MAX_SPLIT_TRAN);

    stor.rct_nodes_start.clear();
    stor.rct_nodes_start.reserve(_fprop_cands.size() + 1);

    int total_num_nodes = 0;
    int total_num_edges = 0; 
    int net_id = 0;
    
    for(Net *net: _modified_nets) {
      size_t sz = net->_init_flat_rct(&stor, total_num_nodes, total_num_edges, net_id);
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
    stor.rct_pinidx2id.resize(_pins.size(), -1);
    stor.rct_pid.resize(total_num_nodes);

    // Step 2: Create task for FlatRct make
    auto updflat = _taskflow.parallel_for(_modified_nets.begin(), _modified_nets.end(), [] (Net *net) {
        net->_update_rc_timing_flat();
      }, 32);

    // Step 3: Create task for computing FlatRctStorage
    auto task_compute = _taskflow.emplace([this] () {
        _flat_rct_stor->_update_timing_cuda();
      });

    // Step 4: Persist Pin delay/impulse data
    auto persdata = _taskflow.parallel_for(_modified_nets.begin(), _modified_nets.end(), [] (Net *net) {
        net->_persist_flatrct();
      }, 32);

    updflat.second.precede(task_compute);
    task_compute.precede(persdata.first);
  }
  else {
    _taskflow.parallel_for(_modified_nets.begin(), _modified_nets.end(), [] (Net *net) {
        net->_update_rc_timing();
      }, 32);
  }
  
  _prof::stop_timer("_build_rc_timing_tasks");
}

// Procedure: _clear_prop_tasks
void Timer::_clear_rc_timing_tasks() {
  // no need to do anything if we use cuda settings

  // or...
  if(!_has_state(CUDA_ENABLED)) {
    for(auto pin: _fprop_cands) {
      pin->_ftask.reset();
    }
  }
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
  
  int n = _pins.size();
  int num_edges = std::numeric_limits<int>::max();
  std::vector<int> out(n, 0);
  std::vector<int> edgelist_start(n + 1, 0);
  std::vector<int> edgelist;
  _prop_frontiers.emplace(n, 0);
  std::vector<int> &frontiers = *_prop_frontiers;
  std::vector<int> frontiers_ends;
  int first_size = 0;

  OT_LOGI("bptc I - IV");

  // count number of edges
  auto count_edges_pair = _tf.parallel_for(_bprop_cands.begin(), _bprop_cands.end(), [&] (Pin *pin) {
      int &szi = edgelist_start[pin->_idx + 1];
      for(auto arc: pin->_fanin) {
        if(!arc->_has_state(Arc::LOOP_BREAKER)
           && arc->_from._has_state(Pin::BPROP_CAND)) ++szi;
      }
      int &outi = out[pin->_idx];
      for(auto arc: pin->_fanout) {
        if(!arc->_has_state(Arc::LOOP_BREAKER)
           && arc->_to._has_state(Pin::BPROP_CAND)) ++outi;
      }
    }, 32);

  //OT_LOGI("bptc II");
  
  // sequential partial sum
  auto prefix_sum = _tf.emplace([&](){
          for(int i = 1; i <= n; ++i) {
              edgelist_start[i] += edgelist_start[i - 1];
          }
          num_edges = edgelist_start[n];
          edgelist.assign(num_edges, 0); 
          });
  prefix_sum.succeed(count_edges_pair.second);
  
  //OT_LOGI("bptc III");

  // put edges into edgelist
  auto [put_edges_S, put_edges_T] = _tf.parallel_for(_bprop_cands.begin(), _bprop_cands.end(), [&] (Pin *pin) {
      int st = edgelist_start[pin->_idx];
      for(auto arc: pin->_fanin) {
        if(!arc->_has_state(Arc::LOOP_BREAKER)
           && arc->_from._has_state(Pin::BPROP_CAND)) {
          edgelist[st++] = arc->_from._idx;
        }
      }
    }, 32);
  put_edges_S.succeed(prefix_sum); 
  
  //OT_LOGI("bptc IV");

  // init default frontier
  auto init_frontier = _tf.emplace([&](){
              first_size = 0;
              for(auto pin: _bprop_cands) {
                if(!out[pin->_idx]) frontiers[first_size++] = pin->_idx;
              }
              frontiers_ends.push_back(0);
              frontiers_ends.push_back(first_size);
          });
  init_frontier.succeed(put_edges_T);

  _ex.run(_tf).wait();
  _tf.clear();
  
  OT_LOGI("bptc V");

  _prof::setup_timer("toposort_compute_cuda");
  
  // Step 2: call GPU function
  toposort_compute_cuda(n, num_edges, first_size,
                        edgelist_start.data(), edgelist.data(), out.data(),
                        frontiers.data(), frontiers_ends);
  
  _prof::stop_timer("toposort_compute_cuda");

  OT_LOGI("bptc VI", "  sz ", frontiers_ends.size());

  // Step 3: build_tasks
  std::optional<tf::Task> last;

  // forward
  for(int i = (int)frontiers_ends.size() - 2; i >= 0; --i) {
    int l = frontiers_ends[i], r = frontiers_ends[i + 1];
    auto [S, T] = _taskflow.parallel_for(frontiers.begin() + l, frontiers.begin() + r, [this] (int idx) {
        Pin *pin = _idx2pin[idx];
        _fprop_slew(*pin);
        _fprop_delay(*pin);
        _fprop_at(*pin);
        _fprop_test(*pin);
      }, 32);
    if(last) last->precede(S);
    last = T;
  }

  OT_LOGI("bptc VII");

  // backward
  for(int i = 0; i < (int)frontiers_ends.size() - 1; ++i) {
    int l = frontiers_ends[i], r = frontiers_ends[i + 1];
    auto [S, T] = _taskflow.parallel_for(frontiers.begin() + l, frontiers.begin() + r, [this] (int idx) {
        Pin *pin = _idx2pin[idx];
        _bprop_rat(*pin);
      }, 32);
    if(last) last->precede(S);
    last = T;
  }
  
  OT_LOGI("bptc VIII");

  // cleanup
  auto task_cleanup = _taskflow.emplace([this] () {
      _prop_frontiers.reset();
    });
  if(last) last->precede(task_cleanup);
  
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
    _prof::stop_timer("_update_timing");
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
    _insert_full_timing_frontiers();
    _insert_full_timing_modified_nets();
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

  // clear modified nets
  _clear_modified_nets();

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

  // below moved to _insert_full_timing_modified_nets
  // // clear the rc-net update flag
  // for(auto& kvp : _nets) {
  //   kvp.second._rc_timing_updated = false;
  // }
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

// Procedure: _insert_full_timing_modified_nets
void Timer::_insert_full_timing_modified_nets() {
  
  // clear the rc-net update flag
  for(auto& kvp : _nets) {
    kvp.second._rc_timing_updated = false;
    _insert_modified_net(kvp.second);
  }
}

// Procedure: _insert_modified_net
void Timer::_insert_modified_net(Net& net) {
  if(net._modified_net_satellite) return;

  net._modified_net_satellite = _modified_nets.insert(_modified_nets.end(), &net);
}

// Procedure: _remove_modified_net
void Timer::_remove_modified_net(Net &net) {
  if(net._modified_net_satellite) {
    _modified_nets.erase(*net._modified_net_satellite);
    net._modified_net_satellite.reset();
  }
}

// Procedure: _clear_modified_nets
void Timer::_clear_modified_nets() {
  for(auto& net: _modified_nets) {
    net->_modified_net_satellite.reset();
  }
  _modified_nets.clear();
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
    
    // Then this net becomes dirty. enable net update
    _insert_modified_net(*net);
  }
  
  // Enable the timing propagation.
  for(auto arc : po._pin._fanin) {
    _insert_frontier(arc->_from);
  }
  _insert_frontier(po._pin);
}


};  // end of namespace ot. -----------------------------------------------------------------------




