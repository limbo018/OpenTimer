#include <ot/timer/timer.hpp>

namespace ot {

// Function: read_spef
Timer& Timer::read_spef(std::filesystem::path path) {

  // Create a spefnet shared pointer
  auto spef = std::make_shared<spef::Spef>(); 
  
  std::scoped_lock lock(_mutex);

  // Reader task
  auto parser = _taskflow.emplace([path=std::move(path), spef] () {
    OT_LOGI("loading spef ", path);
    if(spef->read(path); spef->error) {
      OT_LOGE("Parser-SPEF error:\n", *spef->error);
    }
    spef->expand_name();
  });
  
  // Spef update task (this has to be after parser)
  auto reader = _taskflow.emplace([this, spef] () {
    if(!(spef->error)) {
      _rebase_unit(*spef);
      _read_spef(*spef);
      OT_LOGI("added ", spef->nets.size(), " spef nets");
    }
  });
  
  // Build the task dependency.
  parser.precede(reader);
  
  _add_to_lineage(reader);

  return *this;
}

// Procedure: _read_spef
void Timer::_read_spef(spef::Spef& spef) {
  for(auto& spef_net : spef.nets) {
    if(auto itr = _nets.find(spef_net.name); itr == _nets.end()) {
      OT_LOGW("spef net ", spef_net.name, " not found");
      continue;
    }
    else {
      itr->second._attach(std::move(spef_net));
      _insert_frontier(*itr->second._root);
      _insert_modified_net(itr->second);
    }
  }
}

// Procedure dbg_net
void Timer::dbg_net(std::default_random_engine &rnd, int l, int r) {
  std::vector<Net*> nets, nets_chosen;
  nets.reserve(_nets.size());
  for(auto &it: _nets) {
    nets.push_back(&it.second);
  }
  std::shuffle(nets.begin(), nets.end(), rnd);

  int i = 0, sum = 0;
  
  for(Net *net: nets) {
    if(!net->_spef_net) continue;
    ++i;
    sum += net->_spef_net->ress.size() + 1;

    if(i > r + 5) break;
    if(i >= l) {
      
      OT_LOGW("TESTING ", i, ", sum ", sum, ", name: ", net->_name);

      std::vector<Net*> thisnet;
      thisnet.push_back(net);
      std::ofstream fout("/tmp/single.spef");
      _dump_spef(fout, thisnet);
      fout.close();
    
      std::list<Pin*> net_pins = net->_pins;
      for(Pin *pin: net_pins) {
        OT_LOGI("pin: ", pin->_name);
      }
      std::string net_name = net->_name;

      spef::Net spefnet;
      spefnet = std::move(*net->_spef_net);
    
      _remove_net(*net);
      Net &newnet = _insert_net(net_name);
      for(Pin *pin: net_pins) {
        _connect_pin(*pin, newnet);
      }
      newnet._spef_net = std::move(spefnet);
    
      _add_to_lineage(_taskflow.emplace([] () {
            OT_LOGI("launched by dbg_net");
          }));
      update_timing();
      OT_LOGW("PASS TEST ", i, ", sum ", sum, ", name: ", net_name);
    }
  }
}

// Procedure: roulette_spef
// randomly select some RC nets and reset them
// this is for profiling incremental timing
std::pair<int, int> Timer::roulette_spef(int expect_size, std::default_random_engine &rnd, std::ostream &os) {
  // std::shared_lock lock(_mutex);
  
  int count = 0, sum = 0;
  std::vector<Net*> nets, nets_chosen;
  nets.reserve(_nets.size());
  for(auto &it: _nets) {
    nets.push_back(&it.second);
  }
  std::shuffle(nets.begin(), nets.end(), rnd);
  for(Net *net: nets) {
    if(!net->_rc_timing_updated) continue;
    if(!net->_spef_net) continue;
    ++count;
    sum += net->_spef_net->ress.size() + 1;
    // net->_rc_timing_updated = false;
    // spef::Net spef = std::move(*net->_spef_net);
    // net->_attach(std::move(spef));
    // _insert_frontier(*net->_root);
    // _insert_modified_net(*net);
    nets_chosen.push_back(net);
    
    if(sum >= expect_size) break;
  }
  // if(!_lineage) {
  //   _add_to_lineage(_taskflow.emplace([] () {
  //         OT_LOGI("rouletted");
  //       }));
  // }
  // _dump_spef(os, nets_chosen);

  // choose some nets to delete and re-insert
  for(Net *net: nets_chosen) {
    
    std::list<Pin*> net_pins = net->_pins;
    std::string net_name = net->_name;
    
    _remove_net(*net);
    // these added to lineage
    insert_net(net_name);
    for(Pin *pin: net_pins) {
      connect_pin(pin->_name, net_name);
    }

  }
  
  return std::make_pair(count, sum);
}

};  // end of namespace ot. -----------------------------------------------------------------------
