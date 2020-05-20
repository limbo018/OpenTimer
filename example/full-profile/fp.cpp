#include <ot/timer/timer.hpp>
#include <ot/timer/_prof.hpp>

std::ofstream fout("shelltest/full-profile.out");
std::vector<std::pair<std::string, std::string>> benchmarks = {
  {"b19_iccad_eval", "b19_iccad"},
  {"cordic_ispd2", "cordic_ispd2"},
  {"des_perf_ispd_eval", "des_perf_ispd"},
  {"edit_dist_ispd2", "edit_dist_ispd2"},
  {"leon2_iccad_eval", "leon2_iccad"},
  {"leon3mp_iccad_eval", "leon3mp_iccad"},
  {"netcard_iccad_eval", "netcard_iccad"},
  {"vga_lcd_iccad_eval", "vga_lcd_iccad"}
};

const int n = 2;   // number of tests for each benchmark, cpu or gpu

inline void init_for(std::string name1, std::string name2, ot::Timer &timer) {
  std::string start = "../benchmark/" + name1 + "/" + name2;
  timer.read_celllib(start + "_Early.lib", ot::MIN)
    .read_celllib(start + "_Late.lib", ot::MAX)
    .read_verilog(start + ".v")
    .read_spef(start + ".spef")
    .read_timing(start + ".timing");
}

inline void test_for(std::string name1, std::string name2, bool usegpu) {
  pid_t pid = fork();
  if(pid == 0) {
    fout << name1 << (usegpu ? " GPU " : " CPU ") << std::flush;
    
    ot::Timer timer;
    if(usegpu) timer.cuda(true);
    init_for(name1, name2, timer);
    timer.update_timing();
    size_t n_nets, n_nodes, n_pins;
    timer.get_sizes(n_nets, n_nodes, n_pins);

    fout << n_nets << ' ' << n_nodes << ' ' << n_pins
         << ' ' << _prof::timers["_build_prop_cands"]
         << ' ' << _prof::timers["_build_rc_timing_tasks"]
         << ' ' << _prof::timers["_update_timing__taskflow_rctiming"]
         << ' ' << _prof::timers[usegpu ? "_build_prop_tasks_cuda" : "_build_prop_tasks"]
         << ' ' << _prof::timers["_update_timing__taskflow_prop"]
         << ' ' << _prof::timers["_update_timing"]
         << std::endl << std::flush;

    exit(0);
  }
  else {
    int returnStatus;
    waitpid(pid, &returnStatus, 0);
    std::cout << "Child process finished with exit code " << returnStatus << std::endl;
  }
}

int main(int argc, char *argv[]) {
  for(auto [name1, name2] : benchmarks) {
    for(int i = 1; i <= n; ++i) {
      std::cout << "========== " << name1 << " (CPU, " << i << ")" << std::endl;
      test_for(name1, name2, false);
    }
    for(int i = 1; i <= n; ++i) {
      std::cout << "========== " << name1 << " (GPU, " << i << ")" << std::endl;
      test_for(name1, name2, true);
    }
  }
}
