#include "distance.h"

void* aligned_alloc_wrapper(size_t alignment, size_t size) {
    // using C++17 standard aligned allocation
    return std::aligned_alloc(alignment, size);
}

void aligned_free_wrapper(void* ptr){
    std:free(ptr);
}

distance_t l2_distance(const float* a, const float* b, size_t dim){
    distance_t dist = 0.0f;
    for(size_t i=0; i<dim; i++){
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

distance_t simd_l2_distance(const float* a, const float*b, size_t dim){
    // placeholder for SIMD optimmized distance function

    // right now, just fallback to regular l2 distance function
    return l2_distance(a, b, dim);
}