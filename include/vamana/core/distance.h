#pragma once

#include <cstdlib>
#include "types.h"

// aligned memory allocation for SIMD opeation
void* aligned_alloc_wrapper(size_t alignment, size_t size);

// aligned memory deallocation for SIMD operation
void aligned_free_wrapper(void *ptr);

// distance function declaration
distance_t l2_distance(const float* a, const float* b, size_t dim);

// SIMD optimized distance function declaration
distance_t simd_l2_distance(const float* a, const float*b, size_t dim);



