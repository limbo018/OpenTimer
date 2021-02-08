#pragma once
/*
 * allocator wrapper for cudaMallocHost that can be used in STL containers.
 * @author Zizheng Guo
 * reference: https://en.cppreference.com/w/cpp/named_req/Allocator
 */

#include <limits>
#include <memory>

void *wrapped_cudaMallocHost(std::size_t);
void wrapped_cudaFreeHost(void *);

template<typename T>
struct pinned_memory_allocator {
  typedef T value_type;
  pinned_memory_allocator() = default;
  template<class U> constexpr pinned_memory_allocator (const pinned_memory_allocator<U> &) noexcept {}
  
  [[nodiscard]] T *allocate(std::size_t n) {
    if(n > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_alloc();
    if(auto p = static_cast<T*>(wrapped_cudaMallocHost(n * sizeof(T)))) {
      return p;
    }
    else throw std::bad_alloc();
  }

  void deallocate(T *p, __attribute__((unused)) std::size_t n) noexcept {
    wrapped_cudaFreeHost(p);
  }
};

template <class T, class U>
bool operator==(const pinned_memory_allocator <T>&, const pinned_memory_allocator <U>&) { return true; }
template <class T, class U>
bool operator!=(const pinned_memory_allocator <T>&, const pinned_memory_allocator <U>&) { return false; }

// template<typename T>
// using ot_cuda_allocator = pinned_memory_allocator<T>;
template<typename T>
using ot_cuda_allocator = std::allocator<T>;
