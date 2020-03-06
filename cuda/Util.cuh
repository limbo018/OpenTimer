/**
 * @file   Util.cuh
 * @author Yibo Lin
 * @date   Feb 2020
 */

#ifndef _UTIL_CUH
#define _UTIL_CUH

#include <chrono>

typedef std::chrono::high_resolution_clock::rep hr_clock_rep;

inline hr_clock_rep get_globaltime(void) 
{
	using namespace std::chrono;
	return high_resolution_clock::now().time_since_epoch().count();
}

// Returns the period in miliseconds
inline double get_timer_period(void) 
{
	using namespace std::chrono;
	return 1000.0 * high_resolution_clock::period::num / high_resolution_clock::period::den;
}

#define allocateCUDA(var, size, type) \
{\
    cudaError_t status = cudaMalloc(&(var), (size)*sizeof(type)); \
    if (status != cudaSuccess) \
    { \
        printf("cudaMalloc failed for var##\n"); \
    } \
}

#define destroyCUDA(var) \
{ \
    cudaError_t status = cudaFree(var); \
    if (status != cudaSuccess) \
    { \
        printf("cudaFree failed for var##\n"); \
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

template <typename T>
inline __device__ T CUDADiv(T a, T b) 
{
    return a / b; 
}

template <>
inline __device__ float CUDADiv(float a, float b)
{
    return fdividef(a, b); 
}

template <typename T>
inline __device__ T CUDACeilDiv(T a, T b)
{
    return ceil(CUDADiv(a, b));
}

template <>
inline __device__ int CUDACeilDiv(int a, int b)
{
    return CUDADiv(a+b-1, b); 
}
template <>
inline __device__ unsigned int CUDACeilDiv(unsigned int a, unsigned int b)
{
    return CUDADiv(a+b-1, b); 
}

template <typename T>
inline __host__ T CPUDiv(T a, T b) 
{
    return a / b; 
}

template <typename T>
inline __host__ T CPUCeilDiv(T a, T b)
{
    return ceil(CPUDiv(a, b));
}

template <>
inline __host__ int CPUCeilDiv(int a, int b)
{
    return CPUDiv(a+b-1, b); 
}
template <>
inline __host__ unsigned int CPUCeilDiv(unsigned int a, unsigned int b)
{
    return CPUDiv(a+b-1, b); 
}



#endif
