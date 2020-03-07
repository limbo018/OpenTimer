#include <ot/timer/net.hpp>
#include <deque>
#include <ot/cuda/rct.cuh>
#include <ot/timer/_prof.hpp>

namespace ot {

// Constructor
RctNode::RctNode(const std::string& name) : _name {name} {
}

// Procedure: _scale_capacitance
void RctNode::_scale_capacitance(float s) {
  FOR_EACH_EL_RF(el, rf) {
    _ncap[el][rf] *= s; 
  }
}

// Function: load
float RctNode::load(Split el, Tran rf) const {
  return _load[el][rf];
}

// Function: cap
float RctNode::cap(Split el, Tran rf) const {
  return _pin ? _pin->cap(el, rf) + _ncap[el][rf] : _ncap[el][rf];
}
  
// Function: slew
float RctNode::slew(Split m, Tran t, float si) const {  
  return si < 0.0f ? -std::sqrt(si*si + _impulse[m][t]) : std::sqrt(si*si + _impulse[m][t]);
}

// Function: delay
float RctNode::delay(Split m, Tran t) const {
  return _delay[m][t];
}

// ------------------------------------------------------------------------------------------------

// Constructor
RctEdge::RctEdge(RctNode& from, RctNode& to, float res) : 
  _from {from},
  _to   {to},
  _res  {res} {
}

// Procedure: _scale_resistance
void RctEdge::_scale_resistance(float s) {
  _res *= s;
}

// ------------------------------------------------------------------------------------------------

// Function: _node
RctNode* Rct::_node(const std::string& name) {
  if(auto itr = _nodes.find(name); itr != _nodes.end()) {
    return &(itr->second);
  }
  else return nullptr;
}

// Function: node
const RctNode* Rct::node(const std::string& name) const {
  if(const auto itr = _nodes.find(name); itr != _nodes.end()) {
    return &(itr->second);
  }
  else return nullptr;
}

// Procedure: insert_node
void Rct::insert_node(const std::string& name, float cap) {

  auto& node = _nodes[name];

  node._name = name;

  FOR_EACH_EL_RF(el, rf) {
    node._ncap[el][rf] = cap;
  }
}

// Procedure: insert_edge
void Rct::insert_edge(const std::string& from, const std::string& to, float res) {
  
  auto& tail = _nodes[from];
  auto& head = _nodes[to];
  auto& edge = _edges.emplace_back(tail, head, res);

  tail._fanout.push_back(&edge);
  head._fanin.push_back(&edge);
}
 
// Function: insert_segment
void Rct::insert_segment(const std::string& name1, const std::string& name2, float res) {
  insert_edge(name1, name2, res);
  insert_edge(name2, name1, res);
}

// Procedure: update_rc_timing
void Rct::update_rc_timing() {

  if(!_root) {
    OT_THROW(Error::RCT, "rctree root not found");
  }

  for(auto& kvp : _nodes) {
    FOR_EACH_EL_RF(el, rf) {
      kvp.second._ures[el][rf]    = 0.0f;
      kvp.second._beta[el][rf]    = 0.0f;
      kvp.second._load[el][rf]    = 0.0f;
      kvp.second._delay[el][rf]   = 0.0f;
      kvp.second._ldelay[el][rf]  = 0.0f;
      kvp.second._impulse[el][rf] = 0.0f;
    }
  }
  
  _update_load(nullptr, _root);
  _update_delay(nullptr, _root);   
  _update_ldelay(nullptr, _root);  
  _update_response(nullptr, _root);

}

// Procedure: _update_load
// Compute the load capacitance of each rctree node along the downstream traversal of the rctree.
void Rct::_update_load(RctNode* parent, RctNode* from) {
  // Add downstream capacitances.
  for(auto e : from->_fanout) {
    if(auto& to = e->_to; &to != parent) {
      _update_load(from, &to);
      FOR_EACH_EL_RF(el, rf) {
        from->_load[el][rf] += to._load[el][rf];
      }
    }
  }
  FOR_EACH_EL_RF(el, rf) {
    from->_load[el][rf] += from->cap(el, rf);
  }
}

// Procedure: _update_delay
// Compute the delay of each rctree node using the Elmore delay model.
void Rct::_update_delay(RctNode* parent, RctNode* from) {
  
  for(auto e : from->_fanout) {
    if(auto& to = e->_to; &to != parent) {
      FOR_EACH_EL_RF(el, rf) {
        // Update the delay.
        to._delay[el][rf] = from->_delay[el][rf] + e->_res * to._load[el][rf];
        // Update the upstream resistance.
        to._ures[el][rf] = from->_ures[el][rf] + e->_res;
      }
      _update_delay(from, &to);
    }
  }
}

// Procedure: _update_ldelay
// Compute the load delay of each rctree node along the downstream traversal of the rctree.
void Rct::_update_ldelay(RctNode* parent, RctNode* from) {

  for(auto e : from->_fanout) {
    if(auto& to = e->_to; &to != parent) {
      _update_ldelay(from, &to);
      FOR_EACH_EL_RF(el, rf) {
        from->_ldelay[el][rf] += to._ldelay[el][rf];
      }
    }
  }

  FOR_EACH_EL_RF(el, rf) {
    from->_ldelay[el][rf] += from->cap(el, rf) * from->_delay[el][rf];
  }
}

// Procedure: _update_response
// Compute the impulse and second moment of the input response for each rctree node. 
void Rct::_update_response(RctNode* parent, RctNode* from) {

  for(auto e : from->_fanout) {
    if(auto& to = e->_to; &to != parent) {
      FOR_EACH_EL_RF(el, rf) {
        to._beta[el][rf] = from->_beta[el][rf] + e->_res * to._ldelay[el][rf];
      }
      _update_response(from, &to);
    }
  }

  FOR_EACH_EL_RF(el, rf) {
    from->_impulse[el][rf] = 2.0f * from->_beta[el][rf] - std::pow(from->_delay[el][rf], 2);
  }
}

// Procedure: _scale_capacitance
void Rct::_scale_capacitance(float s) {
  for(auto& kvp : _nodes) {
    kvp.second._scale_capacitance(s);
  }
}

// Procedure: _scale_resistance
void Rct::_scale_resistance(float s) {
  for(auto& edge : _edges) {
    edge._scale_resistance(s);
  }
}

// Function: slew
float Rct::slew(const std::string& name, Split m, Tran t, float si) const {
  auto itr = _nodes.find(name);
  if(itr == _nodes.end()) {
    OT_THROW(Error::RCT, "failed to get slew (rct-node ", name, " not found)");
  }
  return itr->second.slew(m, t, si);
}

// Function: delay
float Rct::delay(const std::string& name, Split m, Tran t) const {
  auto itr = _nodes.find(name);
  if(itr == _nodes.end()) {
    OT_THROW(Error::RCT, "failed to get delay (rct-node ", name, " not found)");
  }
  return itr->second.delay(m, t);
}

// Function: total_ncap
float Rct::total_ncap() const {
  return std::accumulate(_nodes.begin(), _nodes.end(), 0.0f,
    [] (float v, const auto& pair) {
      return v + pair.second._ncap[MIN][RISE];
    }
  );
}

// ------------------------------------------------------------------------------------------------

// flat rct calculation on cpu
void FlatRctStorage::_update_timing_cuda() {
  if(!rct_nodes_start.size()) {
    OT_LOGE("rct storage update timing: not initialized");
    return;
  }

  _prof::setup_timer("_update_timing_cuda :: data preparation");
  load.resize(total_num_nodes * 4);
  delay.resize(total_num_nodes * 4);
  ldelay.resize(total_num_nodes * 4);
  impulse.resize(total_num_nodes * 4);
  // has: total_num_nodes, rct_nodes_start, rct_pid, pres, cap, load, delay, ldelay, impulse

  RctCUDA rct_cuda;
  rct_cuda.num_nets = rct_nodes_start.size() - 1;
  rct_cuda.total_num_nodes = total_num_nodes;
#define COPY_PDATA(arr) rct_cuda.arr = arr.data();
  COPY_PDATA(rct_nodes_start);
  COPY_PDATA(rct_pid);
  COPY_PDATA(pres);
  COPY_PDATA(cap);
  COPY_PDATA(load);
  COPY_PDATA(delay);
  COPY_PDATA(ldelay);
  COPY_PDATA(impulse);
#undef COPY_PDATA

  _prof::stop_timer("_update_timing_cuda :: data preparation");

  
  _prof::setup_timer("_update_timing_cuda :: rct_compute_cuda");
  rct_compute_cuda(rct_cuda);
  _prof::stop_timer("_update_timing_cuda :: rct_compute_cuda");
}

// Sequential flat rct calculation
// Just for debugging.
void FlatRctStorage::_update_timing_cpu() {
  if(!rct_nodes_start.size()) {
    OT_LOGE("rct storage update timing: not initialized");
    return;
  }

  load.resize(total_num_nodes * 4);
  delay.resize(total_num_nodes * 4);
  ldelay.resize(total_num_nodes * 4);
  impulse.resize(total_num_nodes * 4);
  // has: total_num_nodes, rct_nodes_start, rct_pid, pres, cap, load, delay, ldelay, impulse

  for(size_t net_id = 0; net_id < rct_nodes_start.size() - 1; ++net_id) {
    for(unsigned int el_rf_offset = 0; el_rf_offset < MAX_SPLIT_TRAN; ++el_rf_offset) {
      int st = rct_nodes_start[net_id], ed = rct_nodes_start[net_id + 1];
      int st4 = st * 4 + el_rf_offset, ed4 = ed * 4 + el_rf_offset;
      int rst4 = ed4 - 4, red4 = st4;   // red4 = st4, jumping over the root

      // update load from cap
      // and init array
      for(int i = st4; i < ed4; i += 4) {
        load[i] = cap[i];
        delay[i] = ldelay[i] = impulse[i] = 0;
      }

      // update load from downstream to upstream
      for(int i = rst4, j = ed - 1; i > red4; i -= 4, --j) {
        int prev = i - rct_pid[j] * 4;
        load[prev] += load[i];
      }

      // update delay from upstream to downstream
      for(int i = st4 + 4, j = st + 1; i < ed4; i += 4, ++j) {
        int prev = i - rct_pid[j] * 4;
        delay[i] += delay[prev] + load[i] * pres[j];
      }

      // update cap*delay from downstream to upstream
      for(int i = rst4, j = ed - 1; i > red4; i -= 4, --j) {
        int prev = i - rct_pid[j] * 4;
        ldelay[i] += cap[i] * delay[i];
        ldelay[prev] += ldelay[i];
      }
      ldelay[st4] += cap[st4] * delay[st4];

      // update beta from upstream to downstream
      for(int i = st4 + 4, j = st + 1; i < ed4; i += 4, ++j) {
        int prev = i - rct_pid[j] * 4;
        impulse[i] += impulse[prev] + ldelay[i] * pres[j];
      }

      // beta -> impulse
      for(int i = st4; i < ed4; i += 4) {
        float t = delay.at(i);
        impulse[i] = 2 * impulse.at(i) - t * t;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------

float FlatRct::slew(int id, Split m, Tran t, float si) const {
  float impulse = _stor->impulse[(arr_start + rct_node2bfs_order[id]) * MAX_SPLIT_TRAN + m * MAX_TRAN + t];
  return si < 0.0f ? -std::sqrt(si*si + impulse) : std::sqrt(si*si + impulse);
}

float FlatRct::delay(int id, Split m, Tran t) const {
  return _stor->delay[(arr_start + rct_node2bfs_order[id]) * MAX_SPLIT_TRAN + m * MAX_TRAN + t];
}

void FlatRct::_scale_capacitance(float s) {
  for(size_t i = 0; i < _num_nodes; ++i) {
    for(int j = 0; j < MAX_SPLIT_TRAN; ++j) {
      _stor->cap[(arr_start + i) * MAX_SPLIT_TRAN + j] *= s;
    }
  }
}

void FlatRct::_scale_resistance(float s) {
  for(size_t i = 0; i < _num_nodes; ++i) { // i start from 1 also work?
    _stor->pres[arr_start + i] *= s;
  }
}

// ------------------------------------------------------------------------------------------------

// flat rct calculation on cpu
void FlatRct2Storage::_update_timing_cuda() {
  if(!rct_nodes_start.size()) {
    OT_LOGE("rct storage update timing: not initialized");
    return;
  }

  // 1. BFS on cuda 

  RctEdgeArrayCUDA rct_edges_cuda; 
  rct_edges_cuda.num_nets = rct_nodes_start.size() - 1; 
  rct_edges_cuda.total_num_nodes = total_num_nodes; 
  rct_edges_cuda.total_num_edges = total_num_edges; 
#define COPY_PDATA(arr) rct_edges_cuda.arr = arr.data();
  COPY_PDATA(rct_edges); 
  COPY_PDATA(rct_roots); 
  COPY_PDATA(rct_nodes_start); 
  COPY_PDATA(rct_node2bfs_order); 
  COPY_PDATA(rct_pid); 
#undef COPY_PDATA

  rct_bfs_cuda(rct_edges_cuda); 

  // 2. compute timing on CUDA 
  
  _prof::setup_timer("_update_timing_cuda :: data preparation");
  load.resize(total_num_nodes * 4);
  delay.resize(total_num_nodes * 4);
  ldelay.resize(total_num_nodes * 4);
  impulse.resize(total_num_nodes * 4);
  // has: total_num_nodes, rct_nodes_start, rct_pid, pres, cap, load, delay, ldelay, impulse

  RctCUDA rct_cuda;
  rct_cuda.num_nets = rct_nodes_start.size() - 1;
  rct_cuda.total_num_nodes = total_num_nodes;
#define COPY_PDATA(arr) rct_cuda.arr = arr.data();
  COPY_PDATA(rct_nodes_start);
  COPY_PDATA(rct_pid);
  COPY_PDATA(pres);
  COPY_PDATA(cap);
  COPY_PDATA(load);
  COPY_PDATA(delay);
  COPY_PDATA(ldelay);
  COPY_PDATA(impulse);
#undef COPY_PDATA

  _prof::stop_timer("_update_timing_cuda :: data preparation");

  
  _prof::setup_timer("_update_timing_cuda :: rct_compute_cuda");
  rct_compute_cuda(rct_cuda);
  _prof::stop_timer("_update_timing_cuda :: rct_compute_cuda");
}

// ------------------------------------------------------------------------------------------------

float FlatRct2::slew(int id, Split m, Tran t, float si) const {
  float impulse = _stor->impulse[(arr_start + _stor->rct_node2bfs_order[arr_start + id]) * MAX_SPLIT_TRAN + m * MAX_TRAN + t];
  return si < 0.0f ? -std::sqrt(si*si + impulse) : std::sqrt(si*si + impulse);
}

float FlatRct2::delay(int id, Split m, Tran t) const {
  return _stor->delay[(arr_start + _stor->rct_node2bfs_order[arr_start + id]) * MAX_SPLIT_TRAN + m * MAX_TRAN + t];
}

void FlatRct2::_scale_capacitance(float s) {
  for(size_t i = 0; i < _num_nodes; ++i) {
    for(int j = 0; j < MAX_SPLIT_TRAN; ++j) {
      _stor->cap[(arr_start + i) * MAX_SPLIT_TRAN + j] *= s;
    }
  }
}

void FlatRct2::_scale_resistance(float s) {
  for(size_t i = 0; i < _num_nodes; ++i) { // i start from 1 also work?
    _stor->pres[arr_start + i] *= s;
  }
}

// ------------------------------------------------------------------------------------------------

// Constructor
Net::Net(const std::string& name) : 
  _name {name} {
}

// Procedure: _attach
void Net::_attach(spef::Net&& spef_net) {
  assert(spef_net.name == _name && _root);
  _spef_net = std::move(spef_net);
  _rc_timing_updated = false;
}

// Procedure: _make_rct
void Net::_make_rct() {
  
  if(!_spef_net) return;

  // Step 1: create a new rctree object
  auto& rct = _rct.emplace<Rct>();

  // Step 2: insert the node and capacitance (*CAP section).
  for(const auto& [node1, node2, cap] : _spef_net->caps) {
    
    // ground capacitance
    if(node2.empty()) {
      rct.insert_node(node1, cap);
    }
    // TODO: coupling capacitance
  }

  // Step 3: insert the segment (*RES section).
  for(const auto& [node1, node2, res] : _spef_net->ress) {
    rct.insert_segment(node1, node2, res);
  }
  
  _spef_net.reset();
  
  _rc_timing_updated = false;
}

// Procedure: _init_flat_rct
// returns the size of this rctree
size_t Net::_init_flat_rct(FlatRctStorage *_stor, int arr_start) {
  // The construction of a flat rctree must be split into two
  // procedures: init and make
  // init is to allocate proper space for all rctrees in a Timer
  // and make is to actually do the BFS and write to the space
  // allocated in init.
  // This ensures enough parallelization exploited at the
  // time-consuming make step.

  if(!_spef_net) return 0;

  auto &rct = _rct.emplace<FlatRct>();
  
  rct._stor = _stor;
  rct.arr_start = arr_start;
  rct._num_nodes = _spef_net->ress.size() + 1;

  return rct._num_nodes;   // assuming all are grounded caps
}

// Procedure: _init_flat_rct
// returns the size of this rctree
size_t Net::_init_flat_rct2(FlatRct2Storage *_stor, int arr_start, int edge_start, int net_id) {
  // The construction of a flat rctree must be split into two
  // procedures: init and make
  // init is to allocate proper space for all rctrees in a Timer
  // and make is to actually do the BFS and write to the space
  // allocated in init.
  // This ensures enough parallelization exploited at the
  // time-consuming make step.

  if(!_spef_net) return 0;

  auto &rct = _rct.emplace<FlatRct>();
  
  rct._stor = _stor;
  rct._arr_start = arr_start;
  rct._edge_start = edge_start; 
  rct._net_id = net_id; 
  rct._num_nodes = _spef_net->ress.size() + 1;

  return rct._num_nodes;   // assuming all are grounded caps
}

// Procedure: _test_flat_rct
// for time profiling only
// do some work and then reset the flat rctree.
void Net::_test_flat_rct() {
  // Step 1: refer to the flat rctree object created during init
  auto prct = std::get_if<FlatRct>(&_rct);
  if(!prct) return;
  auto &rct = *prct;

  rct.name2id.clear();
  //rct.bfs_order_map.resize(0);
  //rct.rct_node2bfs_order.resize(0);
}

// Procedure: _make_flat_rct
void Net::_make_flat_rct() {
  if(!_spef_net) return;

  _prof::intms st, amap_graph, abfs, acap;

  // Step 1: refer to the flat rctree object created during init
  auto prct = std::get_if<FlatRct>(&_rct);
  if(!prct) return;
  auto &rct = *prct;

  size_t num_nodes = rct._num_nodes;
  FlatRctStorage *_stor = rct._stor;

  st = _prof::timestamp();

  //rct.name2id.reserve(num_nodes);

  // Step 2: build map std::string->int
  int cnt = 0;
  for(const auto& [node1, node2, cap] : _spef_net->caps) {
    (void)cap;

    // ground capacitance
    if(node2.empty()) {
      rct.name2id[node1] = cnt;
      ++cnt;
    }
    else {
      OT_LOGE("flat rct make encounters coupling capacitance, which is not supported by flat rct");
      return;
    }
    // TODO: coupling capacitance
  }

  // Step 3: Build graph
  std::vector<std::vector<std::pair<int, float>>> edges(num_nodes);

  for(const auto& [node1, node2, res] : _spef_net->ress) {
    int a, b;
    if(auto const &it = rct.name2id.find(node1); it != rct.name2id.end()) {
      a = it->second;
    }
    else {
      rct.name2id[node1] = a = cnt++;
    }
    if(auto const &it = rct.name2id.find(node2); it != rct.name2id.end()) {
      b = it->second;
    }
    else {
      rct.name2id[node2] = b = cnt++;
    }

    edges[a].emplace_back(b, res);
    edges[b].emplace_back(a, res);
  }

  amap_graph = _prof::timestamp();

  // Step 4: BFS to compute order
  std::deque<int> q;
  //rct.bfs_order_map.resize(num_nodes);
  rct.rct_node2bfs_order.resize(num_nodes);
  std::vector<char> vis(num_nodes, false);

  int root;
  if(auto it = rct.name2id.find(_root->name()); it == rct.name2id.end()) {
    OT_LOGE("flat rct make cannot locate root in spef tree");
    return;
  }
  else root = it->second;

  vis[root] = true;
  q.push_back(root);
  //rct.bfs_order_map[0] = root;
  rct.rct_node2bfs_order[root] = 0;
  cnt = 1;

  while(!q.empty()) {
    int u = q.front(); q.pop_front();
    int uid = rct.rct_node2bfs_order[u];

    for(auto const &[v, res] : edges[u]) {
      if(!vis[v]) {
        vis[v] = true;
        //rct.bfs_order_map[cnt] = v;
        rct.rct_node2bfs_order[v] = cnt;
        _stor->rct_pid[rct.arr_start + cnt] = cnt - uid;
        _stor->pres[rct.arr_start + cnt] = res;
        ++cnt;
        q.push_back(v);
      }
    }
  }

  abfs = _prof::timestamp();

  // Step 5: set cap according to bfs order

  for(size_t i = 0; i < num_nodes * 4; ++i) {
    _stor->cap[rct.arr_start * 4 + i] = 0;
  }

  for(const auto& [node1, node2, cap] : _spef_net->caps) {
    (void)node2;
    int pos = rct.rct_node2bfs_order[rct.name2id[node1]];
    for(int i = 0; i < MAX_SPLIT_TRAN; ++i) {
      _stor->cap[(rct.arr_start + pos) * MAX_SPLIT_TRAN + i] = cap;
    }
  }

  for(auto pin : _pins) {
    if(auto it = rct.name2id.find(pin->name()); it != rct.name2id.end()) {
      if(rct.rct_node2bfs_order[it->second] == 0) continue; // ignore the root
      FOR_EACH_EL_RF(el, rf) {
        _stor->cap[(rct.arr_start + rct.rct_node2bfs_order[it->second]) * MAX_SPLIT_TRAN + el * MAX_TRAN + rf]
          += pin->cap(el, rf);
      }
    }
    else {
      OT_LOGE("pin ", pin->name(), " not found in rctree ", _name);
    }
  }

  acap = _prof::timestamp();

  // update task timers
  _prof::t_map += amap_graph - st;
  _prof::t_bfs += abfs - amap_graph;
  _prof::t_cap += acap - abfs;

  //_spef_net.reset();
  
  _rc_timing_updated = false;
}

// Procedure: _make_flat_rct2
void Net::_make_flat_rct2() {
  if(!_spef_net) return;

  _prof::intms st, amap_graph, abfs, acap;

  // Step 1: refer to the flat rctree object created during init
  auto prct = std::get_if<FlatRct2>(&_rct);
  if(!prct) return;
  auto &rct = *prct;

  size_t num_nodes = rct._num_nodes;
  FlatRct2Storage *_stor = rct._stor;

  st = _prof::timestamp();

  //rct.name2id.reserve(num_nodes);

  // Step 2: build map std::string->int
  int cnt = 0;
  for(const auto& [node1, node2, cap] : _spef_net->caps) {
    (void)cap;

    // ground capacitance
    if(node2.empty()) {
      rct.name2id[node1] = cnt;
      ++cnt;
    }
    else {
      OT_LOGE("flat rct make encounters coupling capacitance, which is not supported by flat rct");
      return;
    }
    // TODO: coupling capacitance
  }

  // Step 3: Build graph

  int edge_cnt = _rct._edge_start; 
  for(const auto& [node1, node2, res] : _spef_net->ress) {
    auto& edge = _stor->rct_edges[edge_cnt];
    _stor->rct_edges_res[edge_cnt] = res; 
    if(auto const &it = rct.name2id.find(node1); it != rct.name2id.end()) {
      edge.s = it->second;
    }
    else {
      rct.name2id[node1] = edge.s = cnt++;
    }
    if(auto const &it = rct.name2id.find(node2); it != rct.name2id.end()) {
      edge.t = it->second;
    }
    else {
      rct.name2id[node2] = edge.t = cnt++;
    }
    ++edge_cnt;
  }

  int root;
  if(auto it = rct.name2id.find(_root->name()); it == rct.name2id.end()) {
    OT_LOGE("flat rct make cannot locate root in spef tree");
    return;
  }
  else root = it->second;
  _stor->rct_roots[_rct._net_id] = root; 

  amap_graph = _prof::timestamp();

  // update task timers
  _prof::t_map += amap_graph - st;

  //_spef_net.reset();
  
  _rc_timing_updated = false;
}

// Procedure: _scale_capacitance
void Net::_scale_capacitance(float s) {

  std::visit(Functors{
    // Leave this to the next update timing
    [&] (EmptyRct& rct) {
    },
    [&] (Rct& rct) {
      rct._scale_capacitance(s);
    },
    [&] (FlatRct& rct) {
      rct._scale_capacitance(s);
    }
  }, _rct);
  
  _rc_timing_updated = false;
}

// Procedure: _scale_resistance
void Net::_scale_resistance(float s) {

  std::visit(Functors{
    // Leave this to the next update timing
    [&] (EmptyRct& rct) {
    },
    [&] (Rct& rct) {
      rct._scale_resistance(s);
    },
    [&] (FlatRct& rct) {
      rct._scale_resistance(s);
    }
  }, _rct);
  
  _rc_timing_updated = false;
}

void Net::_update_rc_timing_flat() {
  if(_rc_timing_updated) {
    return;
  }

  _make_flat_rct();
  
  std::visit(Functors{
    [&] (EmptyRct& rct) {
      FOR_EACH_EL_RF(el, rf) {
        rct.load[el][rf] = std::accumulate(_pins.begin(), _pins.end(), 0.0f, 
          [this, el=el, rf=rf] (float v, Pin* pin) {
            return pin == _root ? v : v + pin->cap(el, rf);
          }
        );
      }
    },
    [&] (Rct& rct) {
    },
    [&] (FlatRct &rct) {
    }
    }, _rct);
  
  _rc_timing_updated = true; // NOT really, for we also need to compute within the Timer-global FlatRctStorage
}

void Net::_update_rc_timing_flat2() {
  if(_rc_timing_updated) {
    return;
  }

  _make_flat_rct2();
  
  std::visit(Functors{
    [&] (EmptyRct& rct) {
      FOR_EACH_EL_RF(el, rf) {
        rct.load[el][rf] = std::accumulate(_pins.begin(), _pins.end(), 0.0f, 
          [this, el=el, rf=rf] (float v, Pin* pin) {
            return pin == _root ? v : v + pin->cap(el, rf);
          }
        );
      }
    },
    [&] (Rct& rct) {
    },
    [&] (FlatRct &rct) {
    }
    }, _rct);
  
  _rc_timing_updated = true; // NOT really, for we also need to compute within the Timer-global FlatRctStorage
}

// Procedure: _update_rc_timing
void Net::_update_rc_timing() {

  if(_rc_timing_updated) {
    return;
  }

  // Apply the spefnet if any
  _make_rct();
  
  // update the corresponding handle
  std::visit(Functors{
    [&] (EmptyRct& rct) {
      FOR_EACH_EL_RF(el, rf) {
        rct.load[el][rf] = std::accumulate(_pins.begin(), _pins.end(), 0.0f, 
          [this, el=el, rf=rf] (float v, Pin* pin) {
            return pin == _root ? v : v + pin->cap(el, rf);
          }
        );
      }
    },
    [&] (Rct& rct) {
      for(auto pin : _pins) {
        if(auto node = rct._node(pin->name()); node == nullptr) {
          OT_LOGE("pin ", pin->name(), " not found in rctree ", _name);
        }
        else {
          if(pin == _root) {
            rct._root = node;
          }
          else {
            node->_pin = pin;
          }
        }
      }
      rct.update_rc_timing();
    },
    [&] (FlatRct &rct) {
      OT_LOGE("Net::_update_timing doesn't support flat rctree");
    }
  }, _rct);

  _rc_timing_updated = true;
}

// Procedure: _remove_pin
// Remove a pin pointer from the net.
void Net::_remove_pin(Pin& pin) {

  assert(pin._net == this);

  // Reset the root pin
  if(_root == &pin) {
    _root = nullptr;
  }

  // Remove the pin from the pins
  _pins.erase(*(pin._net_satellite));
  pin._net_satellite.reset();
  pin._net = nullptr;
  
  // Enable the timing update.
  _rc_timing_updated = false;
}

// Procedure: _insert_pin
// Insert a pin pointer into the net.
void Net::_insert_pin(Pin& pin) {
  
  if(pin._net == this) {
    return;
  }

  assert(pin._net == nullptr && !pin._net_satellite);
  
  pin._net_satellite = _pins.insert(_pins.end(), &pin);
  pin._net = this;

  // NEW
  if(pin.is_rct_root()) {
    _root = &pin;
  }
  
  // Enable the timing update
  _rc_timing_updated = false;  
}

// Function: _load
// The total capacitive load is defined as the sum of the input capacitance 
// of all the other devices sharing the trace.
// Note that the capacitance of the device driving the trace is not included.
float Net::_load(Split m, Tran t) const {

  // TODO: outdated?
  assert(_rc_timing_updated);

  return std::visit(Functors{
    [&] (const EmptyRct& rct) {
      return rct.load[m][t];
    },
    [&] (const Rct& rct) {
      return rct._root->_load[m][t];
    },
    [&] (const FlatRct &rct) {
      return rct._stor->load[rct.arr_start * MAX_SPLIT_TRAN + m * MAX_TRAN + t];
    }
  }, _rct);
}

// Function: _slew
// Query the slew at the give pin through this net
std::optional<float> Net::_slew(Split m, Tran t, float si, Pin& to) const {

  assert(_rc_timing_updated && to._net == this);

  return std::visit(Functors{
    [&] (const EmptyRct&) -> std::optional<float> {
      return si;
    },
    [&] (const Rct& rct) -> std::optional<float> {
      if(auto node = rct.node(to._name); node) {
        return node->slew(m, t, si);
      }
      else return std::nullopt;
    },
    [&] (const FlatRct& rct) -> std::optional<float> {
      if(auto it = rct.name2id.find(to._name); it != rct.name2id.end()) {
        return rct.slew(it->second, m, t, si);
      }
      else return std::nullopt;
    }
  }, _rct);
}

// Function: _delay
// Query the slew at the given pin through this net.
std::optional<float> Net::_delay(Split m, Tran t, Pin& to) const {
  
  assert(_rc_timing_updated && to._net == this);

  return std::visit(Functors{
    [&] (const EmptyRct&) -> std::optional<float> {
      return 0.0f;
    },
    [&] (const Rct& rct) -> std::optional<float> {
      if(auto node = rct.node(to._name); node) {
        //if(m == 0 && t == 0) OT_LOGI("delay ", node->_name, " ", node->_delay[0][0]);
        return node->delay(m, t);
      }
      else return std::nullopt;
    },
    [&] (const FlatRct& rct) -> std::optional<float> {
      
      if(auto it = rct.name2id.find(to._name); it != rct.name2id.end()) {
        //if(m == 0 && t == 0) OT_LOGI("delay ", to._name, " ", rct._stor->delay[(rct.arr_start + rct.rct_node2bfs_order[it->second]) * MAX_SPLIT_TRAN + m * MAX_TRAN + t]);
        return rct.delay(it->second, m, t);
      }
      else return std::nullopt;
    }
  }, _rct);
}


};  // end of namespace ot. -----------------------------------------------------------------------





