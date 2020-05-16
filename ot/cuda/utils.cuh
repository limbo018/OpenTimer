/**
 * @file   Util.cuh
 * @author Yibo Lin
 * @date   Feb 2020
 */
#include <iostream>
#include <cassert>

#ifndef _UTIL_CUH
#define _UTIL_CUH

#define allocateCUDA(var, size, type) \
  {                                                                     \
    cudaError_t status = cudaMalloc(&(var), (size)*sizeof(type));       \
    if (status != cudaSuccess)                                          \
    {                                                                   \
      printf("cudaMalloc failed for var " #var " with size %d\n", (int)size); \
      printf("CUDA Runtime Error: %s\n",                                \
             cudaGetErrorString(status));                               \
      assert(status == cudaSuccess);                                    \
    }                                                                   \
  }

#define destroyCUDA(var) \
{ \
    cudaError_t status = cudaFree(var); \
    if (status != cudaSuccess) \
    { \
        printf("cudaFree failed for " #var "\n"); \
        assert(status == cudaSuccess);            \
    } \
}

#define checkCUDA(status) \
{\
	if (status != cudaSuccess) { \
		printf("CUDA Runtime Error: %s\n", \
			cudaGetErrorString(status)); \
		assert(status == cudaSuccess); \
	} \
}

#define allocateCopyCUDA(var, rhs, size) \
{\
    allocateCUDA(var, size, decltype(*rhs)); \
    checkCUDA(cudaMemcpy(var, rhs, sizeof(decltype(*rhs))*(size), cudaMemcpyHostToDevice)); \
}

// #define allocateCopyCUDAAsync(var, rhs, size, stream) \
// {\
//     allocateCUDA(var, size, decltype(*rhs)); \
//     checkCUDA(cudaMemcpyAsync(var, rhs, sizeof(decltype(*rhs))*(size), cudaMemcpyHostToDevice, stream)); \
// }
#define allocateCopyCUDAAsync(var, rhs, size, stream) allocateCopyCUDA(var, rhs, size)

#define memcpyDeviceHostCUDA(var, rhs, size) \
  {                                          \
    checkCUDA(cudaMemcpy(var, rhs, sizeof(decltype(*rhs)) * (size), cudaMemcpyDeviceToHost)); \
  }

#define memcpyHostDeviceCUDA(var, rhs, size) \
  {                                          \
    checkCUDA(cudaMemcpy(var, rhs, sizeof(decltype(*rhs)) * (size), cudaMemcpyHostToDevice)); \
  }

#endif
