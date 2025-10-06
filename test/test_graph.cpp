#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include "../include/vamana/core/graph.h"

void test_graph_construction() {
    std::cout << "Testing graph construction..." << std::endl;
    
    Graph graph(10);
    
    // Check initial state
    assert(graph.size() == 10);
    for (size_t i = 0; i < 10; ++i) {
        assert(graph.get_neighbors(i).empty());
    }
    
    std::cout << "✓ Graph construction works" << std::endl;
}

void test_add_edge() {
    std::cout << "Testing add edge..." << std::endl;
    
    Graph graph(5);
    
    // Add some edges
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(1, 3);
    
    // Check edges were added
    auto neighbors_0 = graph.get_neighbors(0);
    assert(neighbors_0.size() == 2);
    assert(std::find(neighbors_0.begin(), neighbors_0.end(), 1) != neighbors_0.end());
    assert(std::find(neighbors_0.begin(), neighbors_0.end(), 2) != neighbors_0.end());
    
    auto neighbors_1 = graph.get_neighbors(1);
    assert(neighbors_1.size() == 1);
    assert(std::find(neighbors_1.begin(), neighbors_1.end(), 3) != neighbors_1.end());
    
    std::cout << "✓ Add edge works" << std::endl;
}

void test_graph_properties() {
    std::cout << "Testing graph properties..." << std::endl;
    
    Graph graph(5);
    
    // Add some edges
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(0, 3);
    
    // Check degree
    assert(graph.degree(0) == 3);
    assert(graph.degree(1) == 0);
    
    std::cout << "✓ Graph properties work" << std::endl;
}

void test_clear() {
    std::cout << "Testing clear..." << std::endl;
    
    Graph graph(5);
    
    // Add some edges
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    assert(graph.get_neighbors(0).size() == 2);
    
    // Clear all edges but keep nodes
    graph.clear();
    assert(graph.size() == 5); // Same number of nodes
    assert(graph.get_neighbors(0).size() == 0); // No edges
    
    std::cout << "✓ Clear works" << std::endl;
}

void test_set_neighbors() {
    std::cout << "Testing set neighbors..." << std::endl;
    
    Graph graph(5);
    
    // Set neighbors directly
    std::vector<location_t> new_neighbors = {1, 3, 4};
    graph.set_neighbors(0, new_neighbors);
    
    auto neighbors = graph.get_neighbors(0);
    assert(neighbors.size() == 3);
    assert(std::find(neighbors.begin(), neighbors.end(), 1) != neighbors.end());
    assert(std::find(neighbors.begin(), neighbors.end(), 3) != neighbors.end());
    assert(std::find(neighbors.begin(), neighbors.end(), 4) != neighbors.end());
    
    std::cout << "✓ Set neighbors works" << std::endl;
}

void test_graph_operations() {
    std::cout << "Testing graph operations..." << std::endl;
    
    Graph graph(4);
    
    // Build a small graph: 0 -> 1, 2; 1 -> 3; 2 -> 3
    graph.add_edge(0, 1);
    graph.add_edge(0, 2);
    graph.add_edge(1, 3);
    graph.add_edge(2, 3);
    
    // Verify structure
    assert(graph.get_neighbors(0).size() == 2);
    assert(graph.get_neighbors(1).size() == 1);
    assert(graph.get_neighbors(2).size() == 1);
    assert(graph.get_neighbors(3).size() == 0);
    
    std::cout << "✓ Graph operations work" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "         GRAPH UNIT TESTS              " << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_graph_construction();
        test_add_edge();
        test_graph_properties();
        test_clear();
        test_set_neighbors();
        test_graph_operations();
        
        std::cout << "\n🎉 ALL GRAPH TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}