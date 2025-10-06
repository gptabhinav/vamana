#include "vamana/core/types.h"
#include "vamana/core/neighbor.h"
#include "vamana/core/distance.h"
#include "vamana/core/scratch.h"
#include "vamana/core/graph.h"
#include "vamana/core/index.h"
#include <iostream>
#include <vector>

using namespace std;

int main(){

    std::vector<Neighbor> neighbors = {
        Neighbor(1, 0.5f),
        Neighbor(2, 0.3f),
        Neighbor(3, 0.8f),
        Neighbor(4, 0.1f),
        Neighbor(2, 0.2f), // duplicate id
        Neighbor(5, 0.4f)
    };


    cout << "Neighbors:" << neighbors.size() << endl;

    // testing priority queue
    NeighborPriorityQueue pq;
    for(const auto& neighbor: neighbors){
        pq.push(neighbor);
    };

    sort_neighbors_by_distance(neighbors);

    cout << "Sorted Neighbors by distance:" << endl;
    for(const auto& neighbor: neighbors){
        cout << "ID: " << neighbor.id << ", Distance: " << neighbor.distance << endl;
    }   

    remove_duplicate_neighbors(neighbors);

    cout<< "After removing duplicates:" << endl;
    for(const auto& neighbor: neighbors){
        cout << "ID: " << neighbor.id << ", Distance: " << neighbor.distance << endl;
    }

    // testing priority queue
    cout << "Neighbors (Priority Queue):" << endl;
    while (!pq.empty()) {
        const auto& neighbor = pq.top();
        cout << "ID: " << neighbor.id << ", Distance: " << neighbor.distance << endl;
        pq.pop();
    }

    cout << "=== Testing Distance Functions ===" << endl;

    // testing distance functions
    const size_t dimension = 4;

    float* a = (float*)aligned_alloc_wrapper(SIMD_ALIGNMENT, dimension * sizeof(float));
    float* b = (float*)aligned_alloc_wrapper(SIMD_ALIGNMENT, dimension * sizeof(float));

    // initialize vectors
    for(size_t i=0; i<dimension; i++){
        a[i] = (float)(i);
        b[i] = (float)(i*2);
    }

    distance_t dist1 = l2_distance(a, b, dimension);
    distance_t dist2 = simd_l2_distance(a, b, dimension);

    cout << "L2 Distance: " << dist1 << endl;
    cout << "SIMD L2 Distance: " << dist2 << endl;

    // clean up
    aligned_free_wrapper(a);
    aligned_free_wrapper(b);

    // test scratch space
    cout << "=== Testing Scratch Space ===" << endl;
    ScratchSpace scratch;
    cout << "Initial Scratch Space Capacity: " << scratch.candidates.capacity() << endl;
    scratch.resize(200);
    cout << "Resized Scratch Space Capacity: " << scratch.candidates.capacity() << endl;

    scratch.candidates.emplace_back(1, 0.5f);
    scratch.candidates.emplace_back(2, 0.3f);
    scratch.result_buffer.push_back(10);
    scratch.result_buffer.push_back(20);

    cout << "Scratch Candidates Size: " << scratch.candidates.size() << endl;
    cout << "Scratch Result Buffer Size: " << scratch.result_buffer.size() << endl;

    scratch.clear();

    cout << "After Clear - Scratch Candidates Size: " << scratch.candidates.size() << endl;
    cout << "After Clear - Scratch Result Buffer Size: " << scratch.result_buffer.size() << endl;

    cout<<"scratch test complete"<<endl;

    // Test graph operations
    cout << "=== Testing Graph Operations ===" << endl;
    Graph test_graph(5);
    test_graph.add_edge(0, 1);
    test_graph.add_edge(0, 2);
    test_graph.add_edge(1, 3);
    test_graph.add_edge(2, 4);
    
    cout << "Graph size: " << test_graph.size() << endl;
    cout << "Node 0 degree: " << test_graph.degree(0) << endl;
    cout << "Node 0 neighbors: ";
    for (location_t neighbor : test_graph.get_neighbors(0)) {
        cout << neighbor << " ";
    }
    cout << endl;

    cout << "All component tests completed successfully!" << endl;

    return 0;
}