#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <chrono>
#include "../include/vamana/core/distance.h"

void test_l2_distance() {
    std::cout << "Testing L2 distance..." << std::endl;
    
    // Test 2D vectors
    float a[] = {1.0f, 2.0f};
    float b[] = {4.0f, 6.0f};
    
    float dist = l2_distance(a, b, 2);
    float expected = 5.0f; // sqrt((4-1)² + (6-2)²) = sqrt(9 + 16) = sqrt(25) = 5
    assert(std::abs(dist - expected) < 1e-6f);
    
    // Test identical vectors
    dist = l2_distance(a, a, 2);
    assert(std::abs(dist) < 1e-6f);
    
    std::cout << "✓ L2 distance works" << std::endl;
}

void test_simd_distance() {
    std::cout << "Testing SIMD distance..." << std::endl;
    
    // Test with small vector first
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    float regular_dist = l2_distance(a, b, 4);
    float simd_dist = simd_l2_distance(a, b, 4);
    
    // SIMD should give same result as regular
    assert(std::abs(simd_dist - regular_dist) < 1e-6f);
    
    std::cout << "✓ SIMD distance correctness (4D)" << std::endl;
}

void test_aligned_memory() {
    std::cout << "Testing aligned memory allocation..." << std::endl;
    
    void* ptr = aligned_alloc_wrapper(32, 128);
    assert(ptr != nullptr);
    
    // Check alignment
    assert(reinterpret_cast<uintptr_t>(ptr) % 32 == 0);
    
    aligned_free_wrapper(ptr);
    
    std::cout << "✓ Aligned memory works" << std::endl;
}

void test_simd_performance() {
    std::cout << "Testing SIMD performance..." << std::endl;
    
    // Test with 128D vectors like SIFT
    const size_t dim = 128;
    const size_t num_tests = 100000;
    
    // Allocate aligned memory for SIMD
    float* a = (float*)aligned_alloc_wrapper(32, dim * sizeof(float));
    float* b = (float*)aligned_alloc_wrapper(32, dim * sizeof(float));
    
    // Initialize with random values
    for (size_t i = 0; i < dim; i++) {
        a[i] = (float)(i % 100) / 10.0f;
        b[i] = (float)((i * 7) % 100) / 10.0f;
    }
    
    // Test correctness first
    float regular_dist = l2_distance(a, b, dim);
    float simd_dist = simd_l2_distance(a, b, dim);
    float diff = std::abs(simd_dist - regular_dist);
    
    std::cout << "128D Regular distance: " << regular_dist << std::endl;
    std::cout << "128D SIMD distance: " << simd_dist << std::endl;
    std::cout << "Difference: " << diff << std::endl;
    
    assert(diff < 1e-4f); // Allow slightly more tolerance for 128D
    std::cout << "✓ SIMD distance correctness (128D)" << std::endl;
    
    // Performance comparison
    auto start = std::chrono::high_resolution_clock::now();
    volatile float sum_regular = 0; // volatile to prevent optimization
    for (size_t i = 0; i < num_tests; i++) {
        sum_regular += l2_distance(a, b, dim);
    }
    auto end_regular = std::chrono::high_resolution_clock::now();
    
    start = std::chrono::high_resolution_clock::now();
    volatile float sum_simd = 0;
    for (size_t i = 0; i < num_tests; i++) {
        sum_simd += simd_l2_distance(a, b, dim);
    }
    auto end_simd = std::chrono::high_resolution_clock::now();
    
    auto regular_time = std::chrono::duration_cast<std::chrono::microseconds>(end_regular - start).count();
    auto simd_time = std::chrono::duration_cast<std::chrono::microseconds>(end_simd - start).count();
    
    double speedup = (double)regular_time / (double)simd_time;
    
    std::cout << "Regular L2 (" << num_tests << " calls): " << regular_time << " μs" << std::endl;
    std::cout << "SIMD L2 (" << num_tests << " calls): " << simd_time << " μs" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // Expect at least 1.5x speedup for 128D vectors
    if (speedup >= 1.5) {
        std::cout << "✓ SIMD performance gain achieved" << std::endl;
    } else {
        std::cout << "⚠ SIMD speedup lower than expected (got " << speedup << "x, expected >1.5x)" << std::endl;
    }
    
    aligned_free_wrapper(a);
    aligned_free_wrapper(b);
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        DISTANCE UNIT TESTS            " << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_l2_distance();
        test_simd_distance();
        test_aligned_memory();
        test_simd_performance();
        
        std::cout << "\n🎉 ALL DISTANCE TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}