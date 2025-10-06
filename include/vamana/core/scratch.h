#pragma once

#include "vamana/core/types.h"
#include "vamana/core/neighbor.h"

#include <vector>

struct ScratchSpace{
    //  this is a temporary storage for search operations
    //  we want to avoid frequent allocations and deallocations during search
    //  so we use a pre-allocated scratch space that can be reused across searches
    std::vector<Neighbor> candidates;      // candidate list during search/build
    std::vector<location_t> visited;       // visited nodes tracking
    std::vector<location_t> result_buffer; // temporary result storage
    std::vector<float> occlude_factors;    // occlusion factors for pruning
    std::vector<Neighbor> neighbor_pool;   // temporary neighbor storage
    
    // constructor
    ScratchSpace();

    // resize all vectors to accommodate max_candidates
    void resize(size_t max_candidates);

    // clear all vectors but keep the capacity
    void clear();

    // reset visited tracking only
    void reset_visited();
    
    // clear candidates and result buffers (used between operations)
    void clear_work_vectors();

};

