#include "vamana/core/scratch.h"

ScratchSpace::ScratchSpace() {
    resize(DEFAULT_L);
}

void ScratchSpace::resize(size_t max_candidates) {
    candidates.reserve(max_candidates);
    visited.reserve(max_candidates);
    distances.reserve(max_candidates);
    occlude_factors.reserve(max_candidates);
}

void ScratchSpace::clear(){
    candidates.clear();
    distances.clear();
    occlude_factors.clear();
}

void ScratchSpace::reset_visited(){
    visited.clear();
}