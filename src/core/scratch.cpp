#include "vamana/core/scratch.h"

ScratchSpace::ScratchSpace() {
    resize(DEFAULT_L);
}

void ScratchSpace::resize(size_t max_candidates) {
    candidates.reserve(max_candidates);
    visited.reserve(max_candidates);
    result_buffer.reserve(max_candidates);
    occlude_factors.reserve(max_candidates);
    neighbor_pool.reserve(max_candidates);
}

void ScratchSpace::clear(){
    candidates.clear();
    visited.clear();
    result_buffer.clear();
    occlude_factors.clear();
    neighbor_pool.clear();
}

void ScratchSpace::reset_visited(){
    visited.clear();
}

void ScratchSpace::clear_work_vectors(){
    candidates.clear();
    result_buffer.clear();
    neighbor_pool.clear();
}