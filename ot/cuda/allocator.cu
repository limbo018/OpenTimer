/*
 * allocator wrapper for cudaMallocHost that can be used in STL containers.
 * @author Zizheng Guo
 * reference: https://en.cppreference.com/w/cpp/named_req/Allocator
 */

#include <limits>
#include <memory>
#include <cassert>

void *wrapped_cudaMallocHost(std::size_t n) {
  void *ret;
  cudaError_t status = cudaMallocHost(&ret, n);
  if(status != cudaSuccess) return NULL;
  else return ret;
}

void wrapped_cudaFreeHost(void *ptr) {
  assert(cudaSuccess == cudaFreeHost(ptr));
}
