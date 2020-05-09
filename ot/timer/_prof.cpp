#include <map>
#include <string>
#include <atomic>
#include <ot/static/logger.hpp>

namespace _prof {
  typedef long int intms;
  std::map<std::string, intms> timers;
  std::atomic<intms> t_map, t_bfs, t_cap;
    
  intms timestamp() {
    using namespace std::chrono;
    milliseconds ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    return ms.count();
  }

  void setup_timer(const std::string &name) {
    OT_LOGI("PROF timer ", name, " started.");
    timers[name] = timestamp();
  }

  void stop_timer(const std::string &name) {
    intms ret = timestamp() - timers[name];
    timers[name] = ret;
    OT_LOGI("PROF timer ", name, " done: ", ret, " ms.");
  }

  void init_tasktimers() {
    t_map = t_bfs = t_cap = 0;
  }

  void finalize_tasktimers() {
#define _pftt(name) OT_LOGI("PROF tasktimers ", #name, ' ', name, " ms.")
    _pftt(t_map);
    _pftt(t_bfs);
    _pftt(t_cap);
#undef _pftt
  }
}


