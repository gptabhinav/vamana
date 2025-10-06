#include <iostream>
#include <cassert>
#include "../include/vamana/core/neighbor.h"

void test_neighbor_construction() {
    std::cout << "Testing neighbor construction..." << std::endl;
    
    // Test default construction
    Neighbor n1;
    assert(n1.id == 0);
    assert(n1.distance == 0.0f);
    
    // Test parameterized construction
    Neighbor n2(42, 3.14f);
    assert(n2.id == 42);
    assert(n2.distance == 3.14f);
    
    std::cout << "✓ Neighbor construction works" << std::endl;
}

void test_neighbor_comparison() {
    std::cout << "Testing neighbor comparison..." << std::endl;
    
    Neighbor n1(1, 2.0f);
    Neighbor n2(2, 1.5f);
    Neighbor n3(3, 2.0f);
    
    // Test distance-based comparison (reverse order for priority queue)
    assert(n1 < n2);  // larger distance is "less than" for priority queue
    assert(!(n2 < n1)); // smaller distance is not "less than"
    
    // Test equality
    assert(n1 == Neighbor(1, 3.0f)); // equality based on ID only
    
    std::cout << "✓ Neighbor comparison works" << std::endl;
}

void test_neighbor_operations() {
    std::cout << "Testing neighbor operations..." << std::endl;
    
    Neighbor n1(10, 5.0f);
    Neighbor n2;
    
    // Test assignment
    n2 = n1;
    assert(n2.id == 10);
    assert(n2.distance == 5.0f);
    
    // Test modification
    n2.id = 20;
    n2.distance = 10.0f;
    assert(n2.id == 20);
    assert(n2.distance == 10.0f);
    
    std::cout << "✓ Neighbor operations work" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        NEIGHBOR UNIT TESTS            " << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_neighbor_construction();
        test_neighbor_comparison();
        test_neighbor_operations();
        
        std::cout << "\n🎉 ALL NEIGHBOR TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}