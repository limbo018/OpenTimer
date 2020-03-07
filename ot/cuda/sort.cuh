/**
 * @file   sort.cuh
 * @author Yibo Lin
 * @date   Mar 2020
 * @brief  Counting sort algorithm in CUDA 
 */

#include <cstdlib>
#include <cassert>
#include <cub/cub.cuh>

/// @brief  Inplace prefix sum with block scan as the kernel. 
template <int BlockDim, int ItemsPerThread=4>
__device__ void block_prefix_sum(int* data, int n) {
    assert(blockDim.x == BlockDim); 
    int tid = threadIdx.x;
    // prefix sum with block scan 
    // CUB's block scan does not support arbitrary length with fixed shared memory 
    // I have to do it batch by batch in sequential  
    typedef cub::BlockScan<int, BlockDim> BlockScan; 
    __shared__ typename BlockScan::TempStorage temp_storage; 
    int thread_data[ItemsPerThread]; 
    int batch_size = blockDim.x * ItemsPerThread; 
    int batch_aggregate = 0; 
    for (int batch_bgn = 0; batch_bgn < n; batch_bgn += batch_size) {
        // copy batch of data to thread_data 
        #pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            int idx = batch_bgn + tid * ItemsPerThread + i; 
            thread_data[i] = (idx < n)? data[idx] : 0;
        }
        if (tid == 0) {
            thread_data[0] += batch_aggregate;
        }
        __syncthreads();
        // compute prefix sum for a batch 
        BlockScan(temp_storage).InclusiveSum(thread_data, thread_data, batch_aggregate); 
        __syncthreads();
        // copy thread_data to batch of data 
        #pragma unroll
        for (int i = 0; i < ItemsPerThread; ++i) {
            int idx = batch_bgn + tid * ItemsPerThread + i; 
            if (idx < n) {
                data[idx] = thread_data[i]; 
            }
        }
        __syncthreads();
    }
}

/// @brief Parallel implementation of couting sort algorithm. 
/// Assume the values of data are in range of [amin, amax] 
/// @param inputs input data array 
/// @param counts record the number of occurence for a value, length of amax - amin + 1
/// @param orders output array for orders of original array, skip if NULL; this is like the output of argsort 
/// @param outputs output array for ordered arra, skip if NULL 
/// @param n length of array 
/// @param amin minimum value in the input data 
/// @param amax maximum value in the input data 
/// @param init_counts whether initialize count array or not 
template <int BlockDim, int ItemsPerThread=4>
__device__ void block_couting_sort(const int* inputs, int* counts, int *orders, int* outputs, int n, int amin, int amax, bool init_counts) {
    assert(amin < amax); 
    int tid = threadIdx.x; 

    // compute counting array if not initialized 
    if (init_counts) { 
        for (int i = tid; i < n; i += blockDim.x) {
            atomicAdd(counts + inputs[i] - amin, 1); 
        }
    }
    __syncthreads(); 

    // prefix sum 
    block_prefix_sum<BlockDim, ItemsPerThread>(counts, n); 

    // determine order by parallel traversing the value, not the index 
    for (int i = 0; i < n; ++i) {
        int val = inputs[i];
        int val_amin = val - amin;
        if (tid == (val_amin % blockDim.x)) {
            int count = (--counts[val_amin]); 
            if (orders) {
                orders[i] = count; 
            }
            if (outputs) {
                outputs[count] = val; 
            }
        }
    }
}

