#include "vamana/core/index.h"
#include <iostream>
#include <cassert>
#include <random>

using namespace std;

vector<float> generate_test_data(size_t num_points, size_t dim) {
    vector<float> data(num_points * dim);
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (size_t i = 0; i < num_points * dim; i++) {
        data[i] = dis(gen);
    }
    
    return data;
}

void test_index_build_and_search() {
    cout << "Testing index build and search..." << endl;
    
    const size_t num_points = 100;
    const size_t dim = 16;
    
    auto data = generate_test_data(num_points, dim);
    
    VamanaIndex index(dim, 8, 16);  // Small parameters
    index.build(data.data(), num_points);
    
    // Verify index built
    assert(index.get_num_points() == num_points);
    assert(index.get_dimension() == dim);
    assert(index.get_medoid() < num_points);
    
    // Test search
    float* query = data.data();
    auto results = index.search(query, 5);
    
    // Should get at least 1 result
    assert(results.size() > 0);
    
    // All results should be valid
    for (const auto& result : results) {
        assert(result.id < num_points);
        assert(result.distance >= 0);
    }
    
    cout << "✓ Index builds and searches successfully" << endl;
    cout << "  Built index with " << num_points << " points" << endl;
    cout << "  Search returned " << results.size() << " results" << endl;
}

void test_multiple_searches() {
    cout << "Testing multiple searches..." << endl;
    
    const size_t num_points = 50;
    const size_t dim = 8;
    
    auto data = generate_test_data(num_points, dim);
    
    VamanaIndex index(dim);
    index.build(data.data(), num_points);
    
    // Test 10 different queries
    for (size_t i = 0; i < 10; i++) {
        float* query = data.data() + (i * dim);
        auto results = index.search(query, 3);
        
        assert(results.size() > 0);
        
        // Results should be sorted by distance
        for (size_t j = 1; j < results.size(); j++) {
            assert(results[j-1].distance <= results[j].distance);
        }
    }
    
    cout << "✓ Multiple searches work correctly" << endl;
}

int main() {
    cout << "========================================" << endl;
    cout << "        INTEGRATION TESTS               " << endl;
    cout << "========================================" << endl;
    
    try {
        test_index_build_and_search();
        test_multiple_searches();
        
        cout << "\n🎉 ALL INTEGRATION TESTS PASSED!" << endl;
        cout << "Your Vamana implementation works end-to-end!" << endl;
        return 0;
    } catch (...) {
        cout << "❌ Integration test failed" << endl;
        return 1;
    }
}