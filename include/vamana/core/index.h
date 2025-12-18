#pragma once

#include "vamana/core/types.h"
#include "vamana/core/neighbor.h"
#include "vamana/core/graph.h"
#include "vamana/core/scratch.h"
#include "vamana/core/distance.h"
#include <vector>
#include <memory>
#include <mutex>

class VamanaIndex
{
private:
    Graph graph;
    float *data;
    size_t num_points;
    size_t dimension;
    location_t medoid;

    // Parameters
    size_t R;    // max degree
    size_t L;    // candidate list size
    float alpha; // diversity parameter
    size_t maxc; // max candidates for pruning

    // Per-node locks for thread-safe graph updates during parallel build
    std::vector<std::unique_ptr<std::mutex>> node_locks;
    bool is_building;  // Flag to indicate if we're in build mode (locks needed)

    // Scratch space for operations
    std::unique_ptr<ScratchSpace> scratch;

    // Build-time threading
    size_t build_threads;
    std::vector<std::unique_ptr<ScratchSpace>> build_scratch;

    // Search-time threading  
    size_t search_threads;
    std::vector<std::unique_ptr<ScratchSpace>> search_scratch;

public:
    // Constructor/Destructor
    /**
     * Constructor - Initialize Vamana index with given parameters
     * @param dim Vector dimension of the dataset
     * @param R Maximum degree (number of neighbors) per node in the graph
     * @param L Candidate list size during search operations
     * @param alpha Diversity parameter for pruning (controls exploration vs exploitation)
     * @param maxc Maximum candidates to consider during pruning operations
     */
    VamanaIndex(size_t dim, size_t R = DEFAULT_R, size_t L = DEFAULT_L,
                float alpha = DEFAULT_ALPHA, size_t maxc = DEFAULT_MAXC);

    /**
     * Destructor - Clean up resources (data is owned by caller, so not deleted)
     */
    ~VamanaIndex();

    // Main operations
    /**
     * Build the Vamana index on the given dataset
     * This is the core index construction algorithm that:
     * 1. Initializes a random graph
     * 2. Finds the medoid (center point)
     * 3. Iteratively improves connections using search_and_prune
     * @param data Pointer to the dataset (num_points * dimension floats)
     * @param num_points Number of vectors in the dataset
     * @param num_threads Number of threads to use for parallel construction (0 = select maximum available threads)
     */
    void build(float *data, size_t num_points, size_t num_threads = 0);
    
    /**
     * Search for k nearest neighbors to the query vector
     * Single-threaded per-query search. For batch parallelism, call this
     * from multiple threads (e.g., with OpenMP parallel for)
     * @param query Query vector (dimension floats)
     * @param k Number of nearest neighbors to return
     * @param search_L Candidate list size for this search (0 = use default L)
     * @return Vector of k nearest neighbors sorted by distance
     */
    std::vector<Neighbor> search(const float *query, size_t k, size_t search_L = 0);

    // I/O operations
    /**
     * Save the constructed index to disk
     * Saves both the graph structure and metadata (medoid, dimensions, etc.)
     * @param filename Base filename (will create .graph and .meta files)
     */
    void save_index(const std::string &filename) const;
    
    /**
     * Load a previously saved index from disk
     * Loads both graph structure and metadata
     * Initializes search scratch spaces for parallel batch search
     * @param filename Base filename to load from
     * @param num_threads Number of threads for parallel batch search (0 = use all available)
     */
    void load_index(const std::string &filename, size_t num_threads = 0);

    // Utility functions
    location_t get_medoid() const { return medoid; }
    
    /**
     * Set data pointer after loading index (for use after load_index)
     * @param data_ptr Pointer to the dataset
     * @param points Number of points in dataset
     */
    void set_data(float* data_ptr, size_t points);
    size_t get_num_points() const { return num_points; }
    size_t get_dimension() const { return dimension; }

private:
    
    // Thread pool initialization helpers
    void initialize_build_scratch(size_t num_threads);
    void initialize_search_scratch(size_t num_threads);

    // Core algorithms
    /**
     * CRITICAL: The occlude_list algorithm - heart of Vamana
     * Implements diversity-aware pruning to select diverse neighbors
     * Ensures the selected neighbors are not occluded by each other
     * This is what makes Vamana different from simple kNN graphs
     * @param location The node we're selecting neighbors for
     * @param pool Candidate neighbors sorted by distance
     * @param result Output vector of selected diverse neighbors
     * @param scratch Scratch space for temporary storage during pruning
     */
    void occlude_list(location_t location, std::vector<Neighbor> &pool,
                      std::vector<location_t> &result, ScratchSpace* scratch);
    
    /**
     * Search and prune operation for a single node during index construction
     * 1. Searches for candidate neighbors using greedy search
     * 2. Prunes candidates using occlude_list for diversity
     * 3. Updates the graph with new connections
     * 4. Performs reverse link insertion (bidirectional connections)
     * @param location The node to update connections for
     */
    void search_and_prune(location_t location);
    
    /**
     * Find the medoid (center point) of the dataset
     * The medoid is used as the starting point for all searches
     * Uses sampling for efficiency on large datasets
     * @return ID of the point closest to the center of the dataset
     */
    location_t find_medoid();
    
    /**
     * Initialize the graph with random connections
     * Creates initial connectivity before the iterative improvement phase
     * Each node gets R random neighbors (excluding self-loops)
     * This provides a starting point for the search_and_prune iterations
     */
    void initialize_random_graph();

    // Helper functions
    /**
     * Greedy graph traversal search algorithm
     * Starting from start_node, explores the graph by always moving to
     * the closest unvisited neighbor until search_L candidates are found
     * @param query Query vector to search for
     * @param search_L Maximum number of candidates to explore
     * @param start_node Node to start the search from (usually medoid)
     * @param scratch Scratch space for temporary storage during search
     * @return Candidate neighbors found during traversal
     */
    std::vector<Neighbor> greedy_search(const float *query, size_t search_L, 
        location_t start_node, ScratchSpace* scratch);
};
