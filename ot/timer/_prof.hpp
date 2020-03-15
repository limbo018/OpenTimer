#include <map>
#include <string>
#include <chrono>
#include <atomic>

namespace _prof {
  typedef long int intms;
  extern std::map<std::string, intms> timers;
  extern std::atomic<intms> t_map, t_bfs, t_cap;

  intms timestamp();

  void setup_timer(const std::string &name);

  void stop_timer(const std::string &name);

  void init_tasktimers();

  void finalize_tasktimers();
}


