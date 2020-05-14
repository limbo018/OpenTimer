
#include <ot/timer/timer.hpp>
#include <ot/timer/_prof.hpp>
#include <fstream>
#include <cmath>

int main(int argc, char *argv[]) {
  std::ofstream fout("shelltest/incr-profile.out");

  // create a timer object
  ot::Timer timer;
  
  // Read design
  timer.read_celllib("benchmark/netcard/netcard_iccad_Early.lib", ot::MIN)
    .read_celllib("benchmark/netcard/netcard_iccad_Late.lib", ot::MAX)
    .read_verilog("benchmark/netcard/netcard_iccad.v")
    .read_spef("benchmark/netcard/netcard_iccad.spef")
    .read_timing("benchmark/netcard/netcard_iccad.timing");

  timer.update_timing();

  std::cout << "===============================================================" << std::endl;

  std::default_random_engine rnd(8026727);

  for(int i = 1; ; i = ceil(i * 1.08)) {
    bool end = false;
    int mxcc = (i <= 50 ? 15 : 2);
    for(int cc = 0; cc < mxcc; ++cc) {
      std::ofstream fspef("/tmp/roulette.spef");
      std::pair<int, int> ret = timer.roulette_spef(i, rnd, fspef);
      fspef.close();
      
      if(ret.second < i) {
        end = true;
        break;
      }
      fout << i << ' ' << ret.first << ' ' << ret.second << std::flush;

      timer.read_spef("/tmp/roulette.spef");
      timer.update_timing();
      fout << ' ' << _prof::timers["_build_prop_cands"]
           << ' ' << _prof::bprop_cands_size
           << ' ' << _prof::timers["_build_rc_timing_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_rctiming"]
           << ' ' << _prof::timers["_build_prop_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_prop"]
           << ' ' << _prof::timers["_update_timing"]
           << std::endl << std::flush;
    }
    if(end) break;
  }

  return 0;
}
