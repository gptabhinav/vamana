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

VamanaIndex::VamanaIndex(size_t dim, size_t R, size_t L, float alpha, size_t maxc) 
    : data(nullptr), num_points(0), dimension(dim), medoid(0),
      R(R), L(L), alpha(alpha), maxc(maxc),
      is_building(false), build_threads(0), search_threads(0) {

}

void VamanaIndex::initialize_build_scratch(size_t num_threads) {
    build_scratch.clear();
    build_scratch.reserve(num_threads);

    std::cout << "Initializing build scratch spaces for " << num_threads << " threads..." << std::endl;

    for(size_t i = 0; i < num_threads; i++){
        build_scratch.push_back(std::make_unique<ScratchSpace>());
    }
    
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
}

void VamanaIndex::initialize_search_scratch(size_t num_threads) {
    search_scratch.clear();
    search_scratch.reserve(num_threads);

    std::cout << "Initializing search scratch spaces for " << num_threads << " threads..." << std::endl;

    for(size_t i = 0; i < num_threads; i++){
        search_scratch.push_back(std::make_unique<ScratchSpace>());
    }
}

VamanaIndex::~VamanaIndex() {
    // data is owned by caller, don't delete it
}

void VamanaIndex::build(float* data_ptr, size_t num_pts, size_t num_threads) {
    data = data_ptr;
    num_points = num_pts;

    // Set build mode flag
    is_building = true;

    // Determine build thread count
    if(num_threads == 0){
        #ifdef _OPENMP
            num_threads = omp_get_max_threads();
        #else
            num_threads = 1;
        #endif
    }

    build_threads = num_threads;
    initialize_build_scratch(build_threads);
    
    // Initialize graph
    graph.resize(num_points);

    // Initialize per-node locks for thread-safe updates
    node_locks.clear();
    node_locks.reserve(num_points);
    for (size_t i = 0; i < num_points; i++) {
        node_locks.push_back(std::make_unique<std::mutex>());
    }
    
    std::cout << "Initialized " << node_locks.size() << " node locks" << std::endl;
    
    // Create initial random graph
    std::cout << "Initializing random graph..." << std::endl;
    initialize_random_graph();
    
    std::cout << "Finding medoid..." << std::endl;
    // Find medoid
    medoid = find_medoid();
    std::cout << "Medoid: " << medoid << std::endl;
    
    // Iterative improvement -- PARALLEL
    // 2048 is just what is being used in DiskANN, it is probably good middleground for performance
    // try this out later, and see what works best based on the size of dataset
    // should this be made configurable (for later) 
    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < num_points; i++) {
        search_and_prune(i);
        
        // simple progress indication based on DiskANN
        // can show percentges out of order. only roughly tells what percentage of data is processed
        // this prints every 100000 nodes. so for a 10k dataset, it prints 10 times
        if (i % 100000 == 0) {

            // progress output is the name of the critical section
            // same named critical section occupy the same lock, and different named ones occupy different locks
            #pragma omp critical(progress_output)
            {
                std::cout << "\r " << (100 * i) / num_points << "% index" << std::endl;
            }
        }
    }

    std::cout << std::endl;
    
    // Build complete - turn off build mode
    is_building = false;
}

std::vector<Neighbor> VamanaIndex::search(const float* query, size_t k, size_t search_L) {
    // Single-threaded per-query search
    // For batch parallelism, call this from multiple threads
    
    if (search_L == 0) search_L = L;
    
    // Ensure search scratch is initialized (should be done during load_index)
    if (search_scratch.empty()) {
        std::cerr << "Error: Search scratch not initialized. Call load_index() with num_threads parameter!" << std::endl;
        throw std::runtime_error("Search scratch not initialized");
    }
    
    // Get thread-local scratch space (for parallel batch search)
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();
    #else
    int thread_id = 0;
    #endif
    
    // Safety check: ensure thread_id is valid
    if (thread_id >= (int)search_scratch.size()) {
        std::cerr << "Warning: thread_id " << thread_id << " >= search_scratch.size() " << search_scratch.size() << std::endl;
        thread_id = 0;
    }
    
    auto& local_scratch = search_scratch[thread_id];
    auto candidates = greedy_search(query, search_L, medoid, local_scratch.get());
    
    // Return top k
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    
    return candidates;
}

// CRITICAL: The occlude_list algorithm - heart of Vamana
void VamanaIndex::occlude_list(location_t location, std::vector<Neighbor>& pool, 
                              std::vector<location_t>& result, ScratchSpace* scratch) {
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

    // Get thread-local build scratch space
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();
    #else
    int thread_id = 0;
    #endif

    // Safety check - ensure thread_id is within bounds
    if (thread_id >= (int)build_scratch.size()) {
        thread_id = 0;  // Fallback to thread 0
    }

    auto& local_scratch = build_scratch[thread_id];

    // Search for candidates
    const float* query = data + location * dimension;
    auto candidates = greedy_search(query, L, medoid, local_scratch.get());
    
    // Use local scratch space for result
    auto& pruned = local_scratch->result_buffer;
    pruned.clear();
    
    occlude_list(location, candidates, pruned, local_scratch.get());
    
    // Update SOURCE node's neighbors (with lock)
    {
        std::lock_guard<std::mutex> guard(*node_locks[location]);
        graph.set_neighbors(location, pruned);
    }

    // Reverse link insertion (inter_insert) - lock each TARGET neighbor
    for (location_t neighbor : pruned) {
        std::vector<location_t> copy_of_neighbors;
        bool prune_needed = false;
        
        // Phase 1: Check and decide (with lock on TARGET)
        {
            std::lock_guard<std::mutex> guard(*node_locks[neighbor]);
            
            auto neighbor_list = graph.get_neighbors(neighbor);
            
            // Check if reverse link already exists
            if (std::find(neighbor_list.begin(), neighbor_list.end(), location) 
                == neighbor_list.end()) {
                
                if (neighbor_list.size() < R * 1.5) {  // SLACK factor
                    // Room to add without pruning
                    std::vector<location_t> updated(neighbor_list.begin(), 
                                                   neighbor_list.end());
                    updated.push_back(location);
                    graph.set_neighbors(neighbor, updated);
                } else {
                    // Need to prune - copy data for processing outside lock
                    copy_of_neighbors.assign(neighbor_list.begin(), 
                                            neighbor_list.end());
                    copy_of_neighbors.push_back(location);
                    prune_needed = true;
                }
            }
        } // Release lock on TARGET
        
        // Phase 2: Expensive pruning (NO lock - happens in parallel)
        if (prune_needed) {
            auto& neighbor_candidates = local_scratch->neighbor_pool;
            neighbor_candidates.clear();
            const float* neighbor_data = data + neighbor * dimension;
            
            for (location_t n : copy_of_neighbors) {
                const float* n_data = data + n * dimension;
                float d = adaptive_l2_distance(neighbor_data, n_data, dimension);
                neighbor_candidates.emplace_back(n, d);
            }
            
            // Use a temporary vector for pruned neighbors (don't reuse result_buffer!)
            std::vector<location_t> pruned_neighbors;
            occlude_list(neighbor, neighbor_candidates, pruned_neighbors, 
                        local_scratch.get());
            
            // Phase 3: Update with pruned list (lock TARGET again)
            {
                std::lock_guard<std::mutex> guard(*node_locks[neighbor]);
                graph.set_neighbors(neighbor, pruned_neighbors);
            }
        }
    }
}

std::vector<Neighbor> VamanaIndex::greedy_search(const float* query, size_t search_L,
    location_t start_node, ScratchSpace* scratch) {
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
        
        // Explore neighbors - copy under lock during build to avoid race
        std::vector<location_t> neighbors_copy;
        if (is_building) {
            std::lock_guard<std::mutex> guard(*node_locks[curr.id]);
            neighbors_copy = graph.get_neighbors(curr.id);
        } else {
            // During search, graph is read-only so no lock needed
            neighbors_copy = graph.get_neighbors(curr.id);
        }
        
        for (location_t neighbor : neighbors_copy) {
            // Bounds check to prevent accessing invalid memory
            if (neighbor >= num_points) {
                std::cerr << "ERROR: Invalid neighbor ID " << neighbor 
                         << " (>= num_points " << num_points << ") from node " << curr.id << std::endl;
                continue;
            }
            
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

void VamanaIndex::load_index(const std::string& filename, size_t num_threads) {
    // Resolve path to datasets/indexes directory
    std::string resolved_path = vamana::io::resolve_dataset_path(filename);
    
    graph.load(resolved_path + ".graph");
    
    // Load metadata
    std::ifstream meta(resolved_path + ".meta", std::ios::binary);
    meta.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    meta.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    meta.read(reinterpret_cast<char*>(&medoid), sizeof(medoid));
    meta.close();
    
    // Initialize search scratch spaces for parallel batch search
    if (num_threads == 0) {
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #else
        num_threads = 1;
        #endif
    }
    
    search_threads = num_threads;
    initialize_search_scratch(search_threads);
}

void VamanaIndex::set_data(float* data_ptr, size_t points) {
    data = data_ptr;
    num_points = points;
}
