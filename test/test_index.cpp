#include <iostream>
#include <cassert>
#include <vector>
#include <random>
#include "../include/vamana/core/index.h"

void test_index_construction() {
    std::cout << "Testing index construction..." << std::endl;
    
    // Create index with 2D vectors
    VamanaIndex index(2, 32, 64, 1.2f, 500);
    
    // Basic checks - constructor should work without errors
    std::cout << "✓ Index construction works" << std::endl;
}

void test_index_build() {
    std::cout << "Testing index build..." << std::endl;
    
    // Create small dataset
    const size_t num_points = 20;
    const size_t dim = 2;
    float* data = new float[num_points * dim];
    
    // Fill with simple test data
    for (size_t i = 0; i < num_points; ++i) {
        data[i * dim + 0] = static_cast<float>(i);
        data[i * dim + 1] = static_cast<float>(i % 5);
    }
    
    VamanaIndex index(dim, 32, 64, 1.2f, 500);
    index.build(data, num_points);
    
    // Check basic properties
    assert(index.get_num_points() == num_points);
    assert(index.get_dimension() == dim);
    
    delete[] data;
    std::cout << "✓ Index build works" << std::endl;
}

void test_index_search() {
    std::cout << "Testing index search..." << std::endl;
    
    // Create simple dataset
    const size_t num_points = 5;
    const size_t dim = 2;
    float data[] = {
        0.0f, 0.0f,  // Point 0
        1.0f, 0.0f,  // Point 1
        0.0f, 1.0f,  // Point 2
        1.0f, 1.0f,  // Point 3
        2.0f, 2.0f   // Point 4
    };
    
    VamanaIndex index(dim, 32, 64, 1.2f, 500);
    index.build(data, num_points);
    
    // Search for point close to (0,0)
    float query[] = {0.1f, 0.1f};
    auto results = index.search(query, 3);
    
    // Should return some results
    assert(!results.empty());
    assert(results.size() <= 3);
    
    std::cout << "✓ Index search works" << std::endl;
}

void test_index_properties() {
    std::cout << "Testing index properties..." << std::endl;
    
    const size_t dim = 3;
    VamanaIndex index(dim, 16, 32, 1.1f, 200);
    
    // Check properties before build
    assert(index.get_dimension() == dim);
    assert(index.get_num_points() == 0); // Not built yet
    
    std::cout << "✓ Index properties work" << std::endl;
}

void test_index_edge_cases() {
    std::cout << "Testing edge cases..." << std::endl;
    
    // Single point dataset
    const size_t dim = 2;
    float data[] = {1.0f, 2.0f};
    
    VamanaIndex index(dim, 32, 64, 1.2f, 500);
    index.build(data, 1);
    
    float query[] = {1.5f, 2.5f};
    auto results = index.search(query, 1);
    assert(results.size() == 1);
    assert(results[0].id == 0);
    
    std::cout << "✓ Edge cases work" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "         INDEX UNIT TESTS              " << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_index_construction();
        test_index_build();
        test_index_search();
        test_index_properties();
        test_index_edge_cases();
        
        std::cout << "\n🎉 ALL INDEX TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}