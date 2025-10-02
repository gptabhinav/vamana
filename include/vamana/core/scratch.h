#pragma once

#include "vamana/core/types.h"
#include "vamana/core/neighbor.h"

#include <vector>

struct ScratchSpace{
    //  this is a temporary storage for search operations
    //  we want to avoid frequent allocations and deallocations during search
    //  so we use a pre-allocated scratch space that can be reused across searches
    std::vector<Neighbor> candidates;   // our candidate list during search
    std::vector<location_t> visited;    // list of visited nodes during search
    std::vector<distance_t> distances;  // precomputed distances to avoid recomputation
    std::vector<float> occlude_factors; // occlusion factors for each candidate
    
    // constructor
    ScratchSpace();

    // resize all vectors to accommondate max_candidates
    void resize(size_t max_candidates);

    // clear all vectors but keep the capacity
    void clear();

    // reset visited tracking
    // this needs to be separate from clear() because visited has a different lifecycle
    // visited is reset after every search, while others are reset multiple times during a search
    void reset_visited();

};

