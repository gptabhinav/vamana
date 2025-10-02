#include "vamana/core/neighbor.h"
#include <algorithm>

void sort_neighbors_by_distance(std::vector<Neighbor>& neighbors){
    std::sort(
        neighbors.begin(), 
        neighbors.end(), 
        [](const Neighbor& a, const Neighbor& b){
            return a.distance < b.distance;
        });
};

void remove_duplicate_neighbors(std::vector<Neighbor>& neighbors){
    // sort th neighbors by id
    // this takes O(n log n)
    std::sort(
        neighbors.begin(),
        neighbors.end(),
        [](const Neighbor& a, const Neighbor&b){
            return a.id < b.id;
        }
    );

    // use std::unique to get iterator to new end of unique range
    // this takes O(n)
    auto last = std::unique(
        neighbors.begin(),
        neighbors.end(),
        [](const Neighbor& a, const Neighbor&b){
            return a.id == b.id;
        }
    );

    // erase the non-unique elements
    // this takes O(1)
    neighbors.erase(last, neighbors.end());
}
