#pragma once

#include "types.h"
#include <queue>
#include <vector>

struct Neighbor{
    location_t id;
    distance_t distance;

    // default constructor
    Neighbor() : id(0), distance(0.0f) {}
    
    // constructor
    Neighbor(location_t id_, distance_t distance_): id(id_), distance(distance_){}

    // operator overloads for priority queue
    bool operator<(const Neighbor& other) const{
        return distance > other.distance;   // reverse order for min-heap, basically we want the smallest distance to have the highest priority
    }

    bool operator>(const Neighbor& other) const{
        return distance < other.distance;   // smaller the distance, higher the priority in the min heap based priority queue
    }

    bool operator==(const Neighbor& other) const{
        return id == other.id;               // equality based on id only. we could also use distance too. basically if we want to avoid duplicates in our neighbor lists
    }

};

typedef std::priority_queue<Neighbor> NeighborPriorityQueue; // min-heap based priority queue for neighbors

void sort_neighbors_by_distance(std::vector<Neighbor>& neighbors);
void remove_duplicate_neighbors(std::vector<Neighbor>& neighbors);