#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
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
    
    float a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[] = {5.0f, 6.0f, 7.0f, 8.0f};
    
    float regular_dist = l2_distance(a, b, 4);
    float simd_dist = simd_l2_distance(a, b, 4);
    
    // SIMD should give same result as regular
    assert(std::abs(simd_dist - regular_dist) < 1e-6f);
    
    std::cout << "✓ SIMD distance works" << std::endl;
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

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        DISTANCE UNIT TESTS            " << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_l2_distance();
        test_simd_distance();
        test_aligned_memory();
        
        std::cout << "\n🎉 ALL DISTANCE TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}