#include <ot/timer/timer.hpp>
#include <ot/timer/_prof.hpp>

void i_netcard(ot::Timer &timer) {
  timer.read_celllib("./benchmark/netcard/netcard_iccad_Early.lib", ot::MIN)
    .read_celllib("./benchmark/netcard/netcard_iccad_Late.lib", ot::MAX)
    .read_verilog("./benchmark/netcard/netcard_iccad.v")
    .read_spef("./benchmark/netcard/netcard_iccad.spef")
    .read_timing("./benchmark/netcard/netcard_iccad.timing");
}

void i_c17(ot::Timer &timer) {
  timer.read_celllib("./benchmark/c17/c17_Early.lib", ot::MIN)
    .read_celllib("./benchmark/c17/c17_Late.lib", ot::MAX)
    .read_verilog("./benchmark/c17/c17.v")
    .read_spef("./benchmark/c17/c17.spef")
    .read_timing("./benchmark/c17/c17.timing");
}

void i_vga_lcd(ot::Timer &timer) {
  timer.read_celllib("./benchmark/vga_lcd/vga_lcd_Early.lib", ot::MIN)
    .read_celllib("./benchmark/vga_lcd/vga_lcd_Late.lib", ot::MAX)
    .read_verilog("./benchmark/vga_lcd/vga_lcd.v")
    .read_spef("./benchmark/vga_lcd/vga_lcd.spef")
    .read_timing("./benchmark/vga_lcd/vga_lcd.timing");
}

void i_wb_dma(ot::Timer &timer) {
  timer.read_celllib("./benchmark/wb_dma/wb_dma_Early.lib", ot::MIN)
    .read_celllib("./benchmark/wb_dma/wb_dma_Late.lib", ot::MAX)
    .read_verilog("./benchmark/wb_dma/wb_dma.v")
    .read_spef("./benchmark/wb_dma/wb_dma.spef")
    .read_timing("./benchmark/wb_dma/wb_dma.timing");
}

void i_des_perf(ot::Timer &timer) {
  timer.read_celllib("./benchmark/des_perf/des_perf_Early.lib", ot::MIN)
    .read_celllib("./benchmark/des_perf/des_perf_Late.lib", ot::MAX)
    .read_verilog("./benchmark/des_perf/des_perf.v")
    .read_spef("./benchmark/des_perf/des_perf.spef")
    .read_timing("./benchmark/des_perf/des_perf.timing");
}

int main(int argc, char *argv[]) {
  std::ofstream fout("shelltest/full-profile.out");
  std::vector<std::pair<std::string, void (*)(ot::Timer &)>> benchmarks = {
    {"netcard", i_netcard},      // need 20GiB memory
    {"c17", i_c17},
    {"vga_lcd", i_vga_lcd},
    {"wb_dma", i_wb_dma},
    {"des_perf", i_des_perf}
  };

  const int n = 2;   // number of tests for each benchmark, cpu or gpu
  
  for(auto [name, init] : benchmarks) {
    for(int i = 1; i <= n; ++i) {
      std::cout << "========== " << name << " (CPU, " << i << ")" << std::endl;
      fout << name << " CPU " << std::flush;
      
      ot::Timer timer;
      init(timer);
      timer.update_timing();
      size_t n_nets, n_nodes, n_pins;
      timer.get_sizes(n_nets, n_nodes, n_pins);

      fout << n_nets << ' ' << n_nodes << ' ' << n_pins
           << ' ' << _prof::timers["_build_prop_cands"]
           << ' ' << _prof::timers["_build_rc_timing_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_rctiming"]
           << ' ' << _prof::timers["_build_prop_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_prop"]
           << ' ' << _prof::timers["_update_timing"]
           << std::endl << std::flush;
    }
    for(int i = 1; i <= n; ++i) {
      std::cout << "========== " << name << " (GPU, " << i << ")" << std::endl;
      fout << name << " GPU " << std::flush;
      
      ot::Timer timer;
      timer.cuda(true);
      init(timer);
      timer.update_timing();
      size_t n_nets, n_nodes, n_pins;
      timer.get_sizes(n_nets, n_nodes, n_pins);

      fout << n_nets << ' ' << n_nodes << ' ' << n_pins
           << ' ' << _prof::timers["_build_prop_cands"]
           << ' ' << _prof::timers["_build_rc_timing_tasks"]
           << ' ' << _prof::timers["_update_timing__taskflow_rctiming"]
           << ' ' << _prof::timers["_build_prop_tasks_cuda"]
           << ' ' << _prof::timers["_update_timing__taskflow_prop"]
           << ' ' << _prof::timers["_update_timing"]
           << std::endl << std::flush;
    }
  }
}
