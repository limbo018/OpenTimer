
#include <ot/timer/timer.hpp>
#include <ot/timer/_prof.hpp>
#include <fstream>
#include <cmath>

int main(int argc, char *argv[]) {
  std::ofstream fout("shelltest/incr-profile.out");

  // create a timer object
  ot::Timer timer;
  
  // Read design
  timer.read_celllib("../../benchmark/netcard/netcard_iccad_Early.lib", ot::MIN)
    .read_celllib("../../benchmark/netcard/netcard_iccad_Late.lib", ot::MAX)
    .read_verilog("../../benchmark/netcard/netcard_iccad.v")
    .read_spef("../../benchmark/netcard/netcard_iccad.spef")
    .read_timing("../../benchmark/netcard/netcard_iccad.timing");

  timer.update_timing();

  timer.report_timing(100);

  // std::default_random_engine rnd(233);
  // timer.dbg_net(rnd, 357, 371);
  
  std::cout << "===============================================================" << std::endl;
  
  std::default_random_engine rnd(233);
  std::ofstream fnouse("/tmp/nothing.spef");
  
  std::pair<int, int> ret = timer.roulette_spef(0x3f3f3f3f, rnd, fnouse);
  std::cout << ret.first << ' ' << ret.second << std::endl;
  
  timer.read_spef("../../benchmark/netcard/netcard_iccad.spef");
  //timer.dbg_flag(true);
  timer.update_timing();
  //timer.dbg_flag(false);
  
  return 0;
}
