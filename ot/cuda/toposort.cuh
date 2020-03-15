#include <vector>

void toposort_compute_cuda(
  int n, int num_edges, int first_size,
  int *edgelist_start, int *edgelist, int *out, int *frontiers,
  std::vector<int> &frontiers_ends);
