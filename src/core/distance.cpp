#include "vamana/core/distance.h"
#include <cstdlib>  // for std::aligned_alloc and std::free, general purpose cpp library for utility functions like memory allocation, random numbers, etc
#include <cmath>    // for std::sqrt, standard math library for mathematical functions like sqrt, sin, cos, etc
#include <immintrin.h>  // for AVX2 SIMD intrinsics

// Global flag to enable/disable SIMD (default: enabled)
bool use_simd = true;

void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    // using C++17 standard aligned allocation
    return std::aligned_alloc(alignment, size);
}

void aligned_free_wrapper(void* ptr){
    std::free(ptr);
}

distance_t l2_distance(const float* a, const float* b, size_t dim){
    distance_t dist = 0.0f;
    for(size_t i=0; i<dim; i++){
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

distance_t simd_l2_distance(const float* a, const float*b, size_t dim){
    __m256 sum = _mm256_setzero_ps();  // Initialize sum vector to zero
    
    size_t simd_dim = (dim / 8) * 8;  // Process 8 floats at a time with AVX2
    
    // SIMD loop - process 8 elements at a time
    for (size_t i = 0; i < simd_dim; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);    // Load 8 floats from a
        __m256 vb = _mm256_loadu_ps(&b[i]);    // Load 8 floats from b
        __m256 diff = _mm256_sub_ps(va, vb);   // diff = a - b
        sum = _mm256_fmadd_ps(diff, diff, sum); // sum += diff * diff (fused multiply-add)
    }
    
    // Horizontal sum of the 8 values in sum vector
    float result[8];
    _mm256_storeu_ps(result, sum);
    float sum_scalar = result[0] + result[1] + result[2] + result[3] + 
                      result[4] + result[5] + result[6] + result[7];
    
    // Handle remaining elements with scalar code
    for (size_t i = simd_dim; i < dim; i++) {
        float diff = a[i] - b[i];
        sum_scalar += diff * diff;
    }
    
    return std::sqrt(sum_scalar);
}

distance_t adaptive_l2_distance(const float* a, const float* b, size_t dim) {
    if (use_simd) {
        return simd_l2_distance(a, b, dim);
    } else {
        return l2_distance(a, b, dim);
    }
}