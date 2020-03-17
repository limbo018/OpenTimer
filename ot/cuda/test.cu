/**
 * @file   test.cu
 * @author Yibo Lin
 * @date   Mar 2020
 */

#include <random>
#include "sort.cuh"
#include "utils.cuh"

#define BLOCK_DIM 4

__global__ void simple_counting_sort_kernel(const int* inputs, int* counts, int* orders, int* outputs, int n, int amin, int amax) {
    block_couting_sort<BLOCK_DIM, 4>(inputs, counts, orders, outputs, n, amin, amax, true);
}

void test_simple() {
    int n = 32; 
    int amin = 1; 
    int amax = n-1; 
    std::default_random_engine generator (123);
    std::uniform_int_distribution<int> distribution(amin,amax);

    std::vector<int> h_inputs (n); 
    for (auto& v : h_inputs) {
        v = std::max(distribution(generator), amin);
    }

    std::vector<int> h_counts (n); 
    std::vector<int> h_orders (n); 
    std::vector<int> h_outputs (n); 

    int* d_inputs; 
    int* d_counts; 
    int* d_orders;
    int* d_outputs; 
    allocateCopyCUDA(d_inputs, h_inputs.data(), h_inputs.size()); 
    allocateCopyCUDA(d_counts, h_counts.data(), h_counts.size()); 
    allocateCopyCUDA(d_orders, h_orders.data(), h_orders.size()); 
    allocateCopyCUDA(d_outputs, h_outputs.data(), h_outputs.size()); 

    simple_counting_sort_kernel<<<1, BLOCK_DIM>>>(d_inputs, d_counts, d_orders, d_outputs, n, amin, amax); 
    checkCUDA(cudaDeviceSynchronize()); 

    checkCUDA(cudaMemcpy(h_counts.data(), d_counts, sizeof(int)*h_counts.size(), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(h_orders.data(), d_orders, sizeof(int)*h_orders.size(), cudaMemcpyDeviceToHost));
    checkCUDA(cudaMemcpy(h_outputs.data(), d_outputs, sizeof(int)*h_outputs.size(), cudaMemcpyDeviceToHost));

    printf("inputs[%d]: ", n); 
    for (int i = 0; i < n; ++i) {
        printf("%d, ", h_inputs[i]); 
    }
    printf("\n");

    printf("counts[%d]: ", n); 
    for (int i = 0; i < n; ++i) {
        printf("%d, ", h_counts[i]); 
    }
    printf("\n");

    printf("orders[%d]: ", n); 
    for (int i = 0; i < n; ++i) {
        printf("%d, ", h_orders[i]); 
    }
    printf("\n");

    printf("outputs[%d]: ", n); 
    for (int i = 0; i < n; ++i) {
        printf("%d, ", h_outputs[i]); 
    }
    printf("\n");

    for (int i = 0; i < n; ++i) {
        if (h_inputs[i] != h_outputs[h_orders[i]]) {
            printf("i = %d\n", i);
        }
        assert(h_inputs[i] == h_outputs[h_orders[i]]); 
    }

    destroyCUDA(d_inputs); 
    destroyCUDA(d_counts); 
    destroyCUDA(d_orders); 
    destroyCUDA(d_outputs); 
}

int main() {

    test_simple(); 
    return 0; 
}
