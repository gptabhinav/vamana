#include "vamana/core/index.h"
#include "vamana/core/io.h"
#include <omp.h>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <limits>
#include <cmath>
#include <iostream>
#include <fstream>

VamanaIndex::VamanaIndex(size_t dim, size_t R, size_t L, float alpha, size_t maxc, size_t num_threads) 
    : data(nullptr), num_points(0), dimension(dim), medoid(0),
      R(R), L(L), alpha(alpha), maxc(maxc) {

    // right now keeping this as a single scratch space, to make it backward compatible
    scratch = std::make_unique<ScratchSpace>();
    
    // if number of threads arent specified. 
    // use maximum available threads using OpenMP configuration
    // or if OpenMP is not available or configured
    // just use one thread
    if(this->num_threads==0){
        #ifdef _OPENMP
            this->num_threads = omp_get_max_threads();
        #else
            this->num_threads = 1;
        #endif
    }

    // initialize thread pool
    initialize_thread_pool();
}

void VamanaIndex::initialize_thread_pool(){
    // we'll probably need to initialize our thread pool multiple times
    // and we would want to get rid of the old scratch entries
    thread_scratch.clear();
    thread_scratch.reserve(this->num_threads);

    for(size_t i = 0; i <this->num_threads; i++){
        thread_scratch.push_back(std::make_unique<ScratchSpace>());
    }

    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
}

VamanaIndex::~VamanaIndex() {
    // data is owned by caller, don't delete it
}

void VamanaIndex::build(float* data_ptr, size_t num_pts) {
    data = data_ptr;
    num_points = num_pts;
    
    // Initialize graph
    graph.resize(num_points);
    
    // Create initial random graph
    initialize_random_graph();
    
    // Find medoid
    medoid = find_medoid();
    
    // Iterative improvement
    for (size_t i = 0; i < num_points; i++) {
        search_and_prune(i);
        
        // Progress indicator
        if (i % 1000 == 0) {
            std::cout << "Processed " << i << "/" << num_points << " nodes" << std::endl;
        }
    }
}

std::vector<Neighbor> VamanaIndex::search(const float* query, size_t k, size_t search_L) {
    if (search_L == 0) search_L = L;
    
    auto candidates = greedy_search(query, search_L, medoid);
    
    // Return top k
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    
    return candidates;
}

// CRITICAL: The occlude_list algorithm - heart of Vamana
void VamanaIndex::occlude_list(location_t location, std::vector<Neighbor>& pool, 
                              std::vector<location_t>& result) {
    if (pool.empty()) return;
    
    // CRITICAL: Must be sorted by distance
    std::sort(pool.begin(), pool.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.distance < b.distance;
    });
    
    result.clear();
    
    // Limit pool size to maxc
    if (pool.size() > maxc) {
        pool.resize(maxc);
    }
    
    auto& occlude_factors = scratch->occlude_factors;
    occlude_factors.clear();
    occlude_factors.resize(pool.size(), 0.0f);
    
    float cur_alpha = 1.0f;  // START AT 1.0, NOT 1.2! This is critical
    
    while (cur_alpha <= alpha && result.size() < R) {
        for (size_t i = 0; i < pool.size() && result.size() < R; i++) {
            if (occlude_factors[i] > cur_alpha) continue;
            
            // Mark as selected
            occlude_factors[i] = std::numeric_limits<float>::max();
            
            // Add to result (avoid self-loops)
            if (pool[i].id != location) {
                result.push_back(pool[i].id);
            }
            
            // Update occlusion factors for remaining candidates
            for (size_t j = i + 1; j < pool.size(); j++) {
                if (occlude_factors[j] > alpha) continue;
                
                const float* point_i = data + pool[i].id * dimension;
                const float* point_j = data + pool[j].id * dimension;
                float djk = adaptive_l2_distance(point_j, point_i, dimension);
                
                if (djk == 0.0f) {
                    occlude_factors[j] = std::numeric_limits<float>::max();
                } else {
                    occlude_factors[j] = std::max(occlude_factors[j], pool[j].distance / djk);
                }
            }
        }
        cur_alpha *= 1.2f;  // Increase alpha for next iteration
    }
}

void VamanaIndex::search_and_prune(location_t location) {
    // Search for candidates
    const float* query = data + location * dimension;
    auto candidates = greedy_search(query, L, medoid);
    
    // Use scratch space for result
    auto& pruned = scratch->result_buffer;
    pruned.clear();
    occlude_list(location, candidates, pruned);
    
    // Update graph
    graph.set_neighbors(location, pruned);
    
    // Reverse link insertion (inter_insert)
    for (location_t neighbor : pruned) {
        auto neighbor_list = graph.get_neighbors(neighbor);
        std::vector<location_t> updated_neighbors(neighbor_list.begin(), neighbor_list.end());
        updated_neighbors.push_back(location);
        
        if (updated_neighbors.size() > R) {
            // Re-prune neighbor's list using scratch space
            auto& neighbor_candidates = scratch->neighbor_pool;
            neighbor_candidates.clear();
            const float* neighbor_data = data + neighbor * dimension;
            
            for (location_t n : updated_neighbors) {
                const float* n_data = data + n * dimension;
                float d = adaptive_l2_distance(neighbor_data, n_data, dimension);
                neighbor_candidates.emplace_back(n, d);
            }
            
            // Reuse result_buffer for pruned neighbors
            auto& pruned_neighbors = scratch->result_buffer;
            pruned_neighbors.clear();
            occlude_list(neighbor, neighbor_candidates, pruned_neighbors);
            graph.set_neighbors(neighbor, pruned_neighbors);
        } else {
            graph.set_neighbors(neighbor, updated_neighbors);
        }
    }
}

std::vector<Neighbor> VamanaIndex::greedy_search(const float* query, size_t search_L, location_t start_node) {
    // Use scratch space to avoid allocations
    scratch->reset_visited();
    auto& visited_set = scratch->visited;
    auto& candidates = scratch->candidates;
    candidates.clear();
    
    // Priority queue for unvisited candidates (min-heap by distance)
    NeighborPriorityQueue unvisited;
    
    // Set for O(1) visited lookup - use vector as set for small sizes
    std::unordered_set<location_t> visited;
    
    // Initialize with start node
    const float* start_data = data + start_node * dimension;
    float dist = adaptive_l2_distance(query, start_data, dimension);
    unvisited.push(Neighbor(start_node, dist));
    
    size_t iterations = 0;
    const size_t MAX_ITERATIONS = search_L * 3; // Allow more exploration
    
    while (iterations < MAX_ITERATIONS && !unvisited.empty()) {
        // Get closest unvisited node
        Neighbor curr = unvisited.top();
        unvisited.pop();
        
        // Skip if already visited
        if (visited.count(curr.id)) continue;
        
        // Mark as visited and add to candidates
        visited.insert(curr.id);
        candidates.push_back(curr);
        
        // Explore neighbors
        for (location_t neighbor : graph.get_neighbors(curr.id)) {
            if (!visited.count(neighbor)) {
                const float* neighbor_data = data + neighbor * dimension;
                float d = adaptive_l2_distance(query, neighbor_data, dimension);
                unvisited.push(Neighbor(neighbor, d));
            }
        }
        
        // Limit queue size to prevent explosion
        if (unvisited.size() > search_L * 4) {
            // Convert to vector, sort, and rebuild queue with best candidates
            std::vector<Neighbor> temp_candidates;
            while (!unvisited.empty()) {
                temp_candidates.push_back(unvisited.top());
                unvisited.pop();
            }
            std::sort(temp_candidates.begin(), temp_candidates.end(), 
                     [](const Neighbor& a, const Neighbor& b) {
                         return a.distance < b.distance;
                     });
            
            size_t keep = std::min(search_L * 2, temp_candidates.size());
            for (size_t i = 0; i < keep; i++) {
                unvisited.push(temp_candidates[i]);
            }
        }
        
        iterations++;
    }
    
    // Sort final candidates by distance
    std::sort(candidates.begin(), candidates.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.distance < b.distance;
    });
    
    // Return top search_L candidates
    size_t result_size = std::min(search_L, candidates.size());
    return std::vector<Neighbor>(candidates.begin(), candidates.begin() + result_size);
}

location_t VamanaIndex::find_medoid() {
    float min_avg_dist = std::numeric_limits<float>::max();
    location_t medoid_candidate = 0;
    
    // Sample points for efficiency
    size_t sample_size = std::min(1000u, (unsigned)num_points);
    
    for (size_t i = 0; i < sample_size; i++) {
        float avg_dist = 0;
        const float* point_i = data + i * dimension;
        
        for (size_t j = 0; j < sample_size; j++) {
            if (i != j) {
                const float* point_j = data + j * dimension;
                avg_dist += adaptive_l2_distance(point_i, point_j, dimension);
            }
        }
        avg_dist /= (sample_size - 1);
        
        if (avg_dist < min_avg_dist) {
            min_avg_dist = avg_dist;
            medoid_candidate = i;
        }
    }
    
    return medoid_candidate;
}

void VamanaIndex::initialize_random_graph() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (size_t i = 0; i < num_points; i++) {
        std::uniform_int_distribution<> dis(0, num_points - 1);
        std::vector<location_t> neighbors;
        
        while (neighbors.size() < std::min(R, num_points - 1)) {
            location_t neighbor = dis(gen);
            if (neighbor != i && 
                std::find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end()) {
                neighbors.push_back(neighbor);
            }
        }
        
        graph.set_neighbors(i, neighbors);
    }
}

void VamanaIndex::save_index(const std::string& filename) const {
    // Resolve path to datasets/indexes directory
    std::string resolved_path = vamana::io::resolve_dataset_path(filename);
    
    graph.save(resolved_path + ".graph");
    
    // Save metadata
    std::ofstream meta(resolved_path + ".meta", std::ios::binary);
    meta.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
    meta.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    meta.write(reinterpret_cast<const char*>(&medoid), sizeof(medoid));
    meta.close();
}

void VamanaIndex::load_index(const std::string& filename) {
    // Resolve path to datasets/indexes directory
    std::string resolved_path = vamana::io::resolve_dataset_path(filename);
    
    graph.load(resolved_path + ".graph");
    
    // Load metadata
    std::ifstream meta(resolved_path + ".meta", std::ios::binary);
    meta.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    meta.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    meta.read(reinterpret_cast<char*>(&medoid), sizeof(medoid));
    meta.close();
}

void VamanaIndex::set_data(float* data_ptr, size_t points) {
    data = data_ptr;
    num_points = points;
}
