#include <iostream>
#include <cassert>
#include <vector>
#include "../include/vamana/core/scratch.h"

void test_scratch_construction() {
    std::cout << "Testing scratch space construction..." << std::endl;
    
    ScratchSpace scratch;
    
    // Check initial state
    assert(scratch.candidates.empty());
    assert(scratch.visited.empty());
    assert(scratch.result_buffer.empty());
    
    std::cout << "✓ Scratch space construction works" << std::endl;
}

void test_resize_operations() {
    std::cout << "Testing resize operations..." << std::endl;
    
    ScratchSpace scratch;
    
    // Resize and check capacity
    scratch.resize(100);
    
    // Capacity should be at least 100 for all vectors
    assert(scratch.candidates.capacity() >= 100);
    assert(scratch.visited.capacity() >= 100);
    assert(scratch.result_buffer.capacity() >= 100);
    
    std::cout << "✓ Resize operations work" << std::endl;
}

void test_clear_operations() {
    std::cout << "Testing clear operations..." << std::endl;
    
    ScratchSpace scratch;
    scratch.resize(50);
    
    // Add some data
    scratch.candidates.push_back(Neighbor(1, 2.0f));
    scratch.visited.push_back(5);
    scratch.result_buffer.push_back(10);
    
    // Verify data exists
    assert(!scratch.candidates.empty());
    assert(!scratch.visited.empty());
    assert(!scratch.result_buffer.empty());
    
    // Clear and verify empty but capacity preserved
    size_t old_capacity = scratch.candidates.capacity();
    scratch.clear();
    assert(scratch.candidates.empty());
    assert(scratch.visited.empty());
    assert(scratch.result_buffer.empty());
    assert(scratch.candidates.capacity() == old_capacity); // Capacity preserved
    
    std::cout << "✓ Clear operations work" << std::endl;
}

void test_reset_visited() {
    std::cout << "Testing reset visited..." << std::endl;
    
    ScratchSpace scratch;
    scratch.resize(50);
    
    // Add data to all vectors
    scratch.candidates.push_back(Neighbor(1, 2.0f));
    scratch.visited.push_back(5);
    scratch.result_buffer.push_back(10);
    
    // Reset only visited
    scratch.reset_visited();
    
    // Only visited should be empty
    assert(!scratch.candidates.empty());
    assert(scratch.visited.empty());
    assert(!scratch.result_buffer.empty());
    
    std::cout << "✓ Reset visited works" << std::endl;
}

void test_clear_work_vectors() {
    std::cout << "Testing clear work vectors..." << std::endl;
    
    ScratchSpace scratch;
    scratch.resize(50);
    
    // Add data to all vectors
    scratch.candidates.push_back(Neighbor(1, 2.0f));
    scratch.visited.push_back(5);
    scratch.result_buffer.push_back(10);
    
    // Clear work vectors
    scratch.clear_work_vectors();
    
    // Candidates and result_buffer should be empty, visited preserved
    assert(scratch.candidates.empty());
    assert(!scratch.visited.empty());
    assert(scratch.result_buffer.empty());
    
    std::cout << "✓ Clear work vectors works" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        SCRATCH UNIT TESTS             " << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        test_scratch_construction();
        test_resize_operations();
        test_clear_operations();
        test_reset_visited();
        test_clear_work_vectors();
        
        std::cout << "\n🎉 ALL SCRATCH TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed: " << e.what() << std::endl;
        return 1;
    }
}