#include <ot/timer/timer.hpp>
#include <ot/timer/_prof.hpp>

std::ofstream fout("full-profile.out");
std::vector<std::pair<std::string, std::string>> benchmarks = {
  //{"c432", "c432"},
  //{"c499", "c499"},
  //{"c1355", "c1355"},
  //{"c1908", "c1908"},
  //{"c2670", "c2670"},
  //{"c3540", "c3540"},
  //{"c5315", "c5315"},
  //{"c6288", "c6288"},
  //{"c7552", "c7552"},
  //{"s27", "s27"},
  //{"s344", "s344"},
  //{"s349", "s349"},
  //{"s386", "s386"},
  //{"s400", "s400"},
  //{"s510", "s510"},
  //{"s526", "s526"},
  //{"s1196", "s1196"},
  //{"s1494", "s1494"},
  //{"aes_core", "aes_core"},
  //{"ac97_ctrl", "ac97_ctrl"},
  //{"tv80", "tv80"}
  //{"vga_lcd_iccad", "vga_lcd_iccad"},
  //{"b19_iccad", "b19_iccad"},
  //{"cordic_ispd", "cordic_ispd"},
  //{"des_perf_ispd", "des_perf_ispd"},
  //{"edit_dist_ispd", "edit_dist_ispd"},
  //{"fft_ispd", "fft_ispd"},
  //{"leon2_iccad", "leon2_iccad"},
  //{"leon3mp_iccad", "leon3mp_iccad"},
  {"netcard_iccad", "netcard_iccad"},
  //{"mgc_edit_dist_iccad", "mgc_edit_dist_iccad"},
  //{"mgc_matrix_mult_iccad", "mgc_matrix_mult_iccad"},
  //{"tau2015_cordic_core", "tau2015_cordic_core"},
  //{"tau2015_crc32d16N", "tau2015_crc32d16N"},
  //{"tau2015_softusb_navre", "tau2015_softusb_navre"},
  //{"tau2015_tip_master", "tau2015_tip_master"}
};

const int n = 1;   // number of tests for each benchmark, cpu or gpu

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
    size_t n_gates, n_nets, n_nodes, n_pins, n_edges, n_pis, n_pos;
    timer.get_sizes(n_pis, n_pos, n_gates, n_nets, n_pins, n_nodes, n_edges);

    if(usegpu) {
      fout << n_pis << ' ' << n_pos << ' ' << n_gates << ' ' << n_nets << ' ' << n_pins << ' ' << n_nodes << ' ' << n_edges 
           << ' ' << _prof::timers["_build_prop_cands"]
           << ' ' << _prof::timers["_build_rc_timing_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_rctiming"]
           << ' ' << _prof::timers["_build_prop_tasks_cuda"]
           << ' ' << _prof::timers["_update_timing__taskflow_prop"]
           << ' ' << _prof::timers["_build_prop_cands"]
                   + _prof::timers["_build_rc_timing_tasks"]
                   + _prof::timers["_update_timing__taskflow_rctiming"]
                   + _prof::timers["_build_prop_tasks_cuda"]
                   + _prof::timers["_update_timing__taskflow_prop"]
           << std::endl << std::flush;
    }
    else {
      fout << n_pis << ' ' << n_pos << ' ' << n_gates << ' ' << n_nets << ' ' << n_pins << ' ' << n_nodes << ' ' << n_edges 
           << ' ' << _prof::timers["_build_prop_cands"]
           << ' ' << _prof::timers["_build_rc_timing_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_rctiming"]
           << ' ' << _prof::timers["_build_prop_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_prop"]
           //<< ' ' << _prof::timers["_update_timing"]
           << ' ' << _prof::timers["_build_prop_cands"]
                   + _prof::timers["_build_rc_timing_tasks"]
                   + _prof::timers["_update_timing__taskflow_rctiming"]
                   + _prof::timers["_build_prop_tasks"]
                   + _prof::timers["_update_timing__taskflow_prop"]
           << std::endl << std::flush;
    }

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
