# Code Analysis: Vamana vs DiskANN

## Overview
This document provides a comprehensive architectural comparison between our Vamana implementation and Microsoft's DiskANN library. The analysis covers both high-level design patterns and low-level implementation details.

---

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Threading and Scratch Space Management](#threading-and-scratch-space-management)
3. [Core Data Structures](#core-data-structures)
4. [Graph Management](#graph-management)
5. [Search Algorithms](#search-algorithms)
6. [Index Construction (Build)](#index-construction-build)
7. [Distance Computation](#distance-computation)
8. [I/O and Serialization](#io-and-serialization)
9. [Configurability and Extensibility](#configurability-and-extensibility)
10. [Memory Management](#memory-management)
11. [Recommendations](#recommendations)

---

## High-Level Architecture

### DiskANN Architecture

**Design Pattern:** Factory + Strategy + Abstract Interfaces

```cpp
// Abstract base class with type erasure
class AbstractIndex {
    virtual void _build(const DataType& data, size_t num_points, TagVector& tags) = 0;
    virtual pair<uint32_t, uint32_t> _search(const DataType& query, size_t K, uint32_t L, 
                                             any& indices, float* distances) = 0;
    // ... other virtual methods
};

// Concrete implementation (template)
template<typename T, typename TagT, typename LabelT>
class Index : public AbstractIndex {
    shared_ptr<AbstractDataStore<T>> _data_store;      // Strategy: data storage
    unique_ptr<AbstractGraphStore> _graph_store;        // Strategy: graph storage
    shared_ptr<AbstractDataStore<T>> _pq_data_store;   // Strategy: PQ storage
    // ...
};

// Configuration builder pattern
IndexConfig config = IndexConfigBuilder()
    .with_metric(L2)
    .with_dimension(dim)
    .with_max_points(max_pts)
    .with_index_write_params(params)
    .with_data_load_store_strategy(DataStoreStrategy::MEMORY)
    .with_graph_load_store_strategy(GraphStoreStrategy::MEMORY)
    .is_dynamic_index(false)
    .is_pq_dist_build(false)
    .build();

// Factory creates index from config
IndexFactory factory(config);
auto index = factory.create_instance();
```

**Key Design Elements:**
- **Abstraction Layers:** `AbstractIndex`, `AbstractDataStore<T>`, `AbstractGraphStore`
- **Strategy Pattern:** Pluggable storage strategies (MEMORY, DISK)
- **Type Erasure:** `std::any` for type-agnostic API surface
- **Builder Pattern:** `IndexConfigBuilder` for complex configuration
- **Factory Pattern:** `IndexFactory` creates instances from config
- **Separation of Concerns:** Data, graph, and PQ storage are separate abstractions

### Vamana Architecture

**Design Pattern:** Monolithic + Direct Composition

```cpp
// Single monolithic class
class VamanaIndex {
    float* data;                                // Raw pointer (externally owned)
    size_t num_points;
    size_t dimension;
    location_t medoid;
    
    Graph graph;                                // Direct composition
    
    size_t R, L;                                // Build parameters
    float alpha;
    size_t maxc;
    
    size_t build_threads, search_threads;
    vector<unique_ptr<ScratchSpace>> build_scratch;
    vector<unique_ptr<ScratchSpace>> search_scratch;
    
public:
    VamanaIndex(size_t dim, size_t R=32, size_t L=100, float alpha=1.2, size_t maxc=750);
    void build(float* data, size_t num_pts, size_t num_threads);
    vector<Neighbor> search(const float* query, size_t k, size_t search_L);
    // ...
};

// Simple types (no abstraction)
class Graph {
    vector<vector<location_t>> adj_list;  // Direct adjacency list
    // ...
};

struct ScratchSpace {
    vector<Neighbor> candidates;
    vector<bool> visited;
    vector<location_t> result_buffer;
    vector<float> occlude_factors;
    vector<Neighbor> neighbor_pool;
};
```

**Key Design Elements:**
- **Monolithic Class:** Single `VamanaIndex` class with all functionality
- **Direct Composition:** `Graph` and `ScratchSpace` are concrete types
- **No Abstraction:** No interfaces, virtual methods, or inheritance
- **Simple Configuration:** Constructor parameters (not builder pattern)
- **Procedural Style:** Functions operate directly on data structures
- **Minimal Encapsulation:** Data members are implementation details (private)

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| **Architecture Style** | Layered with abstractions | Flat/monolithic |
| **Extensibility** | High (strategy pattern) | Low (modify VamanaIndex directly) |
| **Complexity** | High (many abstractions) | Low (simple structure) |
| **Type System** | Template + type erasure | Template only |
| **Configuration** | Builder pattern (15+ options) | Constructor parameters (5 params) |
| **Factory** | Yes (`IndexFactory`) | No (direct construction) |
| **Storage Strategies** | Pluggable (memory/disk) | Fixed (memory only) |
| **Data Ownership** | Owned by `AbstractDataStore` | External (raw pointer) |
| **LOC Estimate** | ~15k lines (complex) | ~2k lines (simple) |

**Trade-offs:**

**DiskANN Advantages:**
- Supports multiple storage backends (memory, disk)
- Easy to add new distance metrics
- Configurable for many use cases
- Better for production library

**Vamana Advantages:**
- Simpler to understand and maintain
- Faster compilation (less template instantiation)
- Lower overhead (no virtual dispatch, no indirection)
- Better for research/prototyping

---

## Threading and Scratch Space Management

### DiskANN Approach

**Thread-safe scratch pool** with RAII pattern:

```cpp
// In diskann::Index class
ConcurrentQueue<InMemQueryScratch<T>*> _query_scratch;

// Scratch manager with RAII pattern
template <typename T>
class ScratchStoreManager {
    InMemQueryScratch<T>* _scratch_space;
    ConcurrentQueue<InMemQueryScratch<T>*>* _query_scratch;
public:
    ScratchStoreManager(ConcurrentQueue<InMemQueryScratch<T>*>& query_scratch) {
        _query_scratch = &query_scratch;
        _query_scratch->pop(_scratch_space);  // Blocks if queue empty
    }
    
    ~ScratchStoreManager() {
        _query_scratch->push(_scratch_space);  // Auto-return on scope exit
    }
    
    InMemQueryScratch<T>* scratch_space() { return _scratch_space; }
};

// Search method (NO num_threads parameter!)
pair<uint32_t, uint32_t> Index::search(const T* query, size_t K, uint32_t L, 
                                       uint32_t* indices, float* distances) {
    ScratchStoreManager<T> manager(_query_scratch);  // Acquire scratch
    auto scratch = manager.scratch_space();
    // ... use scratch for search
}  // Scratch auto-returned here

// Scratch pool initialization (in load_index)
void Index::initialize_query_scratch(uint32_t num_threads, uint32_t search_l,
                                    uint32_t indexing_l, uint32_t r, uint32_t maxc, size_t dim) {
    // Create N scratch objects sized for num_threads + indexing_threads
    for (uint32_t i = 0; i < num_threads; i++) {
        _query_scratch.emplace(new InMemQueryScratch<T>(search_l, indexing_l, r, maxc, dim));
    }
}
```

### Vamana Approach

**Direct vector indexing** by thread ID:

```cpp
// In VamanaIndex class
vector<unique_ptr<ScratchSpace>> build_scratch;
vector<unique_ptr<ScratchSpace>> search_scratch;
size_t build_threads, search_threads;

// Search method (single-query)
vector<Neighbor> VamanaIndex::search(const float* query, size_t k, size_t search_L) {
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();
    #else
    int thread_id = 0;
    #endif
    
    // Safety check
    if (thread_id >= (int)search_scratch.size()) {
        thread_id = 0;
    }
    
    auto& local_scratch = search_scratch[thread_id];
    // ... use local_scratch for search
}

// load_index initializes search_scratch
void VamanaIndex::load_index(const string& filename, size_t num_threads) {
    // ... load graph and metadata
    search_threads = num_threads;
    initialize_search_scratch(search_threads);
}
```

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Scratch storage | `ConcurrentQueue<scratch*>` | `vector<unique_ptr<ScratchSpace>>` |
| Thread safety | Queue with blocking | Bounds-checked indexing |
| Memory management | RAII (ScratchStoreManager) | Manual (no RAII) |
| Overflow handling | Blocks/waits for available scratch | Falls back to thread 0 |
| Search signature | `search(query, K, L, indices, distances)` | `search(query, k, search_L)` |
| Batch parallelism | External in apps (OMP) | External in apps (OMP) |
| Initialization | `initialize_query_scratch(...)` in `load()` | `initialize_search_scratch(...)` in `load_index()` |
| Sizing | `num_threads + indexing_threads` | `num_threads` |

---

## Core Data Structures

### DiskANN Data Structures

```cpp
// Abstract data store (strategy pattern)
template<typename data_t>
class AbstractDataStore {
    virtual data_t* get_vector(location_t i) = 0;
    virtual void set_vector(location_t i, const data_t* vec) = 0;
    virtual size_t save(const string& filename, location_t num_points) = 0;
    virtual size_t load(const string& filename) = 0;
    // ...
};

// In-memory implementation
template<typename T>
class InMemDataStore : public AbstractDataStore<T> {
    T* _data;  // Aligned allocation
    size_t _capacity;
    size_t _dimension;
    // ...
};

// Abstract graph store
class AbstractGraphStore {
    virtual const vector<location_t>& get_neighbours(location_t i) const = 0;
    virtual void add_neighbour(location_t i, location_t neighbour_id) = 0;
    virtual void set_neighbours(location_t i, vector<location_t>& neighbours) = 0;
    // ...
};

// In-memory graph implementation
class InMemGraphStore : public AbstractGraphStore {
    vector<vector<location_t>> _graph;
    vector<non_recursive_mutex> _locks;  // Per-node locks!
    // ...
};
```

**Key Features:**
- **Per-node locks** in graph store for fine-grained concurrency
- **Aligned memory** allocation for SIMD efficiency
- **Abstraction layers** for different storage backends
- **Capacity management** (separate from active point count)

### Vamana Data Structures

```cpp
// Simple adjacency list graph
class Graph {
    vector<vector<location_t>> adj_list;
    
public:
    void add_edge(location_t from, location_t to);
    void set_neighbors(location_t node, const vector<location_t>& neighbors);
    const vector<location_t>& get_neighbors(location_t node) const;
    size_t degree(location_t node) const;
    void save(const string& filename) const;
    void load(const string& filename);
    // ...
};

// Scratch space for work vectors
struct ScratchSpace {
    vector<Neighbor> candidates;
    vector<bool> visited;
    vector<location_t> result_buffer;
    vector<float> occlude_factors;
    vector<Neighbor> neighbor_pool;
    
    void reset_visited() {
        fill(visited.begin(), visited.end(), false);
    }
};

// Basic types
using location_t = uint32_t;
using distance_t = float;

struct Neighbor {
    location_t id;
    distance_t distance;
    
    Neighbor(location_t i, distance_t d) : id(i), distance(d) {}
    
    bool operator<(const Neighbor& other) const {
        return distance > other.distance;  // Max-heap for priority queue
    }
};
```

**Key Features:**
- **No locking** (assumes external synchronization)
- **Stack allocation** for work vectors (no alignment)
- **No abstraction** (concrete types only)
- **Simpler memory layout** (vector-of-vectors)

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Graph representation | `AbstractGraphStore` (pluggable) | `Graph` (vector-of-vectors) |
| Per-node locking | Yes (`non_recursive_mutex`) | No (external sync) |
| Data alignment | Yes (SIMD-aligned `_data`) | No (raw pointer, external) |
| Abstraction level | High (abstract interfaces) | None (concrete types) |
| Storage flexibility | Memory or disk | Memory only |
| Memory ownership | DataStore owns data | External (raw pointer) |

---

## Graph Management

### DiskANN Graph Operations

```cpp
// Graph store with fine-grained locking
class InMemGraphStore {
    vector<vector<location_t>> _graph;
    vector<non_recursive_mutex> _locks;  // One lock per node!
    
    void add_neighbour(location_t i, location_t neighbour_id) {
        LockGuard guard(_locks[i]);  // RAII lock
        _graph[i].push_back(neighbour_id);
    }
    
    void set_neighbours(location_t i, vector<location_t>& neighbours) {
        LockGuard guard(_locks[i]);
        _graph[i] = neighbours;
    }
    
    // Degree bounds checking
    size_t _reserve_graph_degree;  // Max degree * slack factor
    size_t _max_observed_degree = 0;
};

// Graph slack factor for over-allocation
constexpr float GRAPH_SLACK_FACTOR = 1.3f;
```

**Key Features:**
- **Per-node locks** for concurrent insertions (dynamic index)
- **Degree bounds** with slack factor for growth
- **RAII lock guards** for exception safety
- **Max observed degree tracking**

### Vamana Graph Operations

```cpp
class Graph {
    vector<vector<location_t>> adj_list;
    
public:
    void add_edge(location_t from, location_t to) {
        if (from >= adj_list.size()) {
            resize(from + 1);
        }
        
        // Avoid duplicates
        auto& neighbors = adj_list[from];
        if (find(neighbors.begin(), neighbors.end(), to) == neighbors.end()) {
            neighbors.push_back(to);
        }
    }
    
    void set_neighbors(location_t node, const vector<location_t>& neighbors) {
        if (node >= adj_list.size()) {
            resize(node + 1);
        }
        adj_list[node] = neighbors;
    }
    
    void clear() {
        for (auto& neighbors : adj_list) {
            neighbors.clear();
        }
    }
};
```

**Key Features:**
- **No locking** (single-threaded or external sync)
- **Automatic resizing** on demand
- **Duplicate checking** in `add_edge`
- **No degree bounds** (grows as needed)

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Concurrency | Per-node locks | No locks (external sync) |
| Degree bounds | Yes (with slack factor) | No (dynamic vectors) |
| RAII locking | Yes (`LockGuard`) | N/A |
| Dynamic growth | Pre-allocated with slack | On-demand resize |
| Duplicate prevention | No (assumes correct input) | Yes (in `add_edge`) |
| Use case | Dynamic index (concurrent inserts) | Static index (batch build) |

---

## Search Algorithms

### DiskANN Search (Greedy Best-First)

```cpp
// Core search algorithm - beam search with dual queues
template<typename T>
vector<Neighbor> Index<T>::iterate_to_fixed_point(
    const T* query, uint32_t L_value, const vector<uint32_t>& init_ids,
    InMemQueryScratch<T>* scratch, bool use_filter) {
    
    // Min-heap for best candidates (closest points)
    priority_queue<Neighbor> best_candidates;
    
    // Max-heap for candidates to explore (farthest in beam)
    priority_queue<Neighbor, vector<Neighbor>, greater<Neighbor>> candidate_queue;
    
    // Initialize with entry points
    for (auto id : init_ids) {
        float dist = distance_function(query, data[id]);
        best_candidates.emplace(id, dist);
        candidate_queue.emplace(id, dist);
    }
    
    while (!candidate_queue.empty()) {
        auto curr = candidate_queue.top();
        candidate_queue.pop();
        
        // Early termination: curr farther than L-th best
        if (best_candidates.size() >= L && curr.distance > best_candidates.top().distance) {
            break;
        }
        
        // Explore neighbors
        for (auto neighbor : graph.get_neighbors(curr.id)) {
            if (!visited.count(neighbor)) {
                visited.insert(neighbor);
                float dist = distance_function(query, data[neighbor]);
                
                if (best_candidates.size() < L || dist < best_candidates.top().distance) {
                    best_candidates.emplace(neighbor, dist);
                    candidate_queue.emplace(neighbor, dist);
                    
                    // Prune best_candidates to size L
                    if (best_candidates.size() > L) {
                        best_candidates.pop();
                    }
                }
            }
        }
    }
    
    return convert_to_vector(best_candidates);
}
```

**Algorithm Properties:**
- **Beam search** with beam width L
- **Early termination** when current candidate farther than L-th best
- **Dual priority queues**: best candidates (min-heap), exploration frontier (max-heap)
- **Visited set** to avoid revisits
- **Initialization** from frozen points or medoid

### Vamana Search

```cpp
vector<Neighbor> VamanaIndex::greedy_search(
    const float* query, size_t search_L, location_t start_node, ScratchSpace* scratch) {
    
    scratch->reset_visited();
    auto& candidates = scratch->candidates;
    candidates.clear();
    
    // Priority queue for unvisited candidates (min-heap by distance)
    NeighborPriorityQueue unvisited;  // typedef priority_queue<Neighbor>
    
    unordered_set<location_t> visited;
    
    // Initialize with start node
    float dist = adaptive_l2_distance(query, data[start_node * dimension], dimension);
    unvisited.push(Neighbor(start_node, dist));
    
    size_t iterations = 0;
    const size_t MAX_ITERATIONS = search_L * 3;
    
    while (iterations < MAX_ITERATIONS && !unvisited.empty()) {
        // Get closest unvisited node
        Neighbor curr = unvisited.top();
        unvisited.pop();
        
        if (visited.count(curr.id)) continue;
        
        visited.insert(curr.id);
        candidates.push_back(curr);
        
        // Explore neighbors
        for (location_t neighbor : graph.get_neighbors(curr.id)) {
            if (!visited.count(neighbor)) {
                float d = adaptive_l2_distance(query, data[neighbor * dimension], dimension);
                unvisited.push(Neighbor(neighbor, d));
            }
        }
        
        // Limit queue size to prevent explosion
        if (unvisited.size() > search_L * 4) {
            vector<Neighbor> temp;
            while (!unvisited.empty()) {
                temp.push_back(unvisited.top());
                unvisited.pop();
            }
            sort(temp.begin(), temp.end(), 
                 [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
            
            size_t keep = min(search_L * 2, temp.size());
            for (size_t i = 0; i < keep; i++) {
                unvisited.push(temp[i]);
            }
        }
        
        iterations++;
    }
    
    // Sort and return top search_L
    sort(candidates.begin(), candidates.end(), 
         [](const Neighbor& a, const Neighbor& b) { return a.distance < b.distance; });
    
    size_t result_size = min(search_L, candidates.size());
    return vector<Neighbor>(candidates.begin(), candidates.begin() + result_size);
}
```

**Algorithm Properties:**
- **Greedy expansion** without dual queues
- **No early termination** criterion
- **MAX_ITERATIONS** guard (search_L * 3)
- **Queue pruning** when size exceeds search_L * 4
- **Single priority queue** for unvisited candidates

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Algorithm style | Beam search (dual queues) | Greedy expansion (single queue) |
| Early termination | Yes (beam width criterion) | No (MAX_ITERATIONS guard) |
| Priority queues | 2 (best, frontier) | 1 (unvisited) |
| Queue pruning | Automatic (size L) | Manual (if > search_L * 4) |
| Iteration limit | Implicit (termination) | Explicit (search_L * 3) |
| Memory efficiency | Better (fixed L size) | Worse (queue can grow) |
| Correctness | Proven convergence | Heuristic bounds |

**Recommendation:** Adopt DiskANN's dual-queue beam search algorithm for better convergence and memory efficiency.

---

## Index Construction (Build)

### DiskANN Build Algorithm

```cpp
template<typename T, typename TagT, typename LabelT>
void Index<T, TagT, LabelT>::build(const string& data_file, size_t num_points,
                                   IndexFilterParams& params) {
    // 1. Load data into data store
    _data_store->populate_data(data_file, 0U);
    
    // 2. Generate frozen point(s) if dynamic
    if (_dynamic_index) {
        generate_frozen_point();
    }
    
    // 3. Calculate entry point (medoid)
    _start = calculate_entry_point();
    
    // 4. Link all points (batch parallel)
    #pragma omp parallel for schedule(dynamic,1)
    for (size_t i = 0; i < _nd; i++) {
        link_point(i, ...);  // Search + prune + inter_insert
    }
    
    // 5. Saturate graph (optional refinement)
    if (_saturate_graph) {
        prune_all_neighbors(R);
    }
}

// Robust pruning with occlusion (RobustPrune in paper)
template<typename T>
void Index<T>::prune_neighbors(location_t location, vector<Neighbor>& pool,
                               vector<location_t>& pruned_list) {
    
    // Ensure pool is sorted
    sort(pool.begin(), pool.end());
    
    pruned_list.clear();
    pruned_list.reserve(R);
    
    float cur_alpha = 1.0f;
    vector<float> occlude_factors(pool.size(), 0.0f);
    
    while (cur_alpha <= alpha && pruned_list.size() < R) {
        for (size_t i = 0; i < pool.size() && pruned_list.size() < R; i++) {
            if (occlude_factors[i] > cur_alpha) continue;
            
            // Select candidate
            occlude_factors[i] = numeric_limits<float>::max();
            pruned_list.push_back(pool[i].id);
            
            // Update occlusion factors for remaining
            for (size_t j = i + 1; j < pool.size(); j++) {
                if (occlude_factors[j] > alpha) continue;
                
                float djk = distance(pool[j].id, pool[i].id);
                occlude_factors[j] = max(occlude_factors[j], pool[j].distance / djk);
            }
        }
        cur_alpha *= 1.2f;
    }
}
```

**Key Features:**
- **Frozen points** for dynamic index navigation
- **Medoid calculation** for entry point
- **Batch parallelism** with `schedule(dynamic,1)`
- **RobustPrune algorithm** (occlusion-based)
- **Alpha progression** (1.0 → 1.2 → 1.44 → ...)
- **Inter-insertion** (reverse links)

### Vamana Build Algorithm

```cpp
void VamanaIndex::build(float* data_ptr, size_t num_pts, size_t num_threads) {
    data = data_ptr;
    num_points = num_pts;
    
    build_threads = num_threads;
    initialize_build_scratch(build_threads);
    
    graph.resize(num_points);
    
    // 1. Initialize random graph
    initialize_random_graph();
    
    // 2. Find medoid
    medoid = find_medoid();
    
    // 3. Iterative improvement (parallel)
    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < num_points; i++) {
        search_and_prune(i);
        
        if (i % 100000 == 0) {
            #pragma omp critical(progress_output)
            {
                cout << "\r " << (100 * i) / num_points << "% indexed" << endl;
            }
        }
    }
}

void VamanaIndex::occlude_list(location_t location, vector<Neighbor>& pool,
                               vector<location_t>& result, ScratchSpace* scratch) {
    if (pool.empty()) return;
    
    sort(pool.begin(), pool.end());
    result.clear();
    
    if (pool.size() > maxc) {
        pool.resize(maxc);
    }
    
    auto& occlude_factors = scratch->occlude_factors;
    occlude_factors.clear();
    occlude_factors.resize(pool.size(), 0.0f);
    
    float cur_alpha = 1.0f;  // START AT 1.0, NOT 1.2!
    
    while (cur_alpha <= alpha && result.size() < R) {
        for (size_t i = 0; i < pool.size() && result.size() < R; i++) {
            if (occlude_factors[i] > cur_alpha) continue;
            
            occlude_factors[i] = numeric_limits<float>::max();
            
            if (pool[i].id != location) {
                result.push_back(pool[i].id);
            }
            
            for (size_t j = i + 1; j < pool.size(); j++) {
                if (occlude_factors[j] > alpha) continue;
                
                const float* point_i = data + pool[i].id * dimension;
                const float* point_j = data + pool[j].id * dimension;
                float djk = adaptive_l2_distance(point_j, point_i, dimension);
                
                if (djk == 0.0f) {
                    occlude_factors[j] = numeric_limits<float>::max();
                } else {
                    occlude_factors[j] = max(occlude_factors[j], pool[j].distance / djk);
                }
            }
        }
        cur_alpha *= 1.2f;
    }
}
```

**Key Features:**
- **Random graph initialization** (not incremental)
- **Medoid calculation** (samples 1000 points)
- **Batch parallelism** with `schedule(dynamic, 2048)`
- **Occlusion-based pruning** (RobustPrune variant)
- **Alpha progression** (matches DiskANN)
- **Inter-insertion** with re-pruning

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Initialization | Incremental (link one at a time) | Random graph |
| Entry point | Medoid (all points) | Medoid (sample 1000) |
| Frozen points | Yes (for dynamic) | No |
| Pruning algorithm | RobustPrune | RobustPrune (same core logic) |
| Alpha start | 1.0 | 1.0 (CRITICAL: not 1.2) |
| Inter-insertion | Yes (with locks) | Yes (no locks) |
| Schedule | `dynamic,1` | `dynamic,2048` |
| Progress output | No | Yes (every 100k) |

---

## Distance Computation

### DiskANN Distance

```cpp
// Abstract distance interface
template<typename T>
class Distance {
public:
    virtual float compare(const T* a, const T* b, size_t dim) const = 0;
    virtual ~Distance() = default;
};

// L2 distance with AVX optimizations
template<typename T>
class DistanceL2 : public Distance<T> {
    float compare(const T* a, const T* b, size_t dim) const override {
        #ifdef USE_AVX2
        return avx2_l2_distance(a, b, dim);
        #else
        return scalar_l2_distance(a, b, dim);
        #endif
    }
};

// AVX2 implementation
float avx2_l2_distance(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum of 8 floats
    float result[8];
    _mm256_storeu_ps(result, sum);
    float total = 0;
    for (int j = 0; j < 8; j++) total += result[j];
    
    // Scalar tail
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    
    return sqrt(total);
}
```

**Key Features:**
- **Abstract interface** (`Distance<T>`)
- **Compile-time selection** (`#ifdef USE_AVX2`)
- **FMA instructions** (`_mm256_fmadd_ps`)
- **Aligned loads** (assumes aligned data)
- **Multiple metrics** (L2, cosine, MIPS)

### Vamana Distance

```cpp
// Global flag for runtime selection
bool use_simd = true;

// Scalar L2 distance
distance_t l2_distance(const float* a, const float* b, size_t dim) {
    distance_t sum = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// SIMD L2 distance (AVX2)
#ifdef __AVX2__
distance_t simd_l2_distance(const float* a, const float* b, size_t dim) {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    // Horizontal sum
    __m256 hsum = _mm256_hadd_ps(sum, sum);
    hsum = _mm256_hadd_ps(hsum, hsum);
    float result[8];
    _mm256_storeu_ps(result, hsum);
    distance_t total = result[0] + result[4];
    
    // Scalar tail
    for (; i < dim; i++) {
        float diff = a[i] - b[i];
        total += diff * diff;
    }
    
    return std::sqrt(total);
}
#endif

// Adaptive dispatcher
distance_t adaptive_l2_distance(const float* a, const float* b, size_t dim) {
    #ifdef __AVX2__
    if (use_simd) {
        return simd_l2_distance(a, b, dim);
    }
    #endif
    return l2_distance(a, b, dim);
}
```

**Key Features:**
- **Runtime selection** (global `use_simd` flag)
- **Unaligned loads** (`_mm256_loadu_ps`)
- **Horizontal add** (`_mm256_hadd_ps`)
- **Single metric** (L2 only)
- **No abstraction** (direct functions)

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Abstraction | Abstract `Distance<T>` interface | Direct functions |
| Selection | Compile-time (`#ifdef`) | Runtime (global flag) |
| Memory alignment | Aligned loads (stricter) | Unaligned loads (flexible) |
| Horizontal sum | Manual array sum | `_mm256_hadd_ps` |
| Metrics | L2, cosine, MIPS | L2 only |
| Configurability | Per-index (strategy) | Global flag |

---

## I/O and Serialization

### DiskANN I/O

```cpp
// Data store serialization
template<typename T>
size_t InMemDataStore<T>::save(const string& filename, location_t num_points) {
    ofstream writer(filename, ios::binary);
    
    // Write header
    uint32_t num_pts = (uint32_t)num_points;
    uint32_t dim = (uint32_t)_dimension;
    writer.write((char*)&num_pts, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    
    // Write data (contiguous)
    writer.write((char*)_data, num_points * _dimension * sizeof(T));
    
    writer.close();
    return num_points * _dimension * sizeof(T) + 2 * sizeof(uint32_t);
}

// Graph store serialization
size_t InMemGraphStore::store(const string& filename, size_t num_points,
                              size_t num_frozen_pts, uint32_t start) {
    ofstream writer(filename, ios::binary);
    
    // Write metadata
    uint32_t file_version = 1;
    uint64_t file_size = calculate_file_size();
    uint32_t max_degree = get_max_observed_degree();
    uint32_t start_node = start;
    uint32_t num_frozen = (uint32_t)num_frozen_pts;
    
    writer.write((char*)&file_size, sizeof(uint64_t));
    writer.write((char*)&max_degree, sizeof(uint32_t));
    writer.write((char*)&start_node, sizeof(uint32_t));
    writer.write((char*)&num_frozen, sizeof(uint32_t));
    
    // Write adjacency list
    for (size_t i = 0; i < num_points; i++) {
        uint32_t degree = (uint32_t)_graph[i].size();
        writer.write((char*)&degree, sizeof(uint32_t));
        writer.write((char*)_graph[i].data(), degree * sizeof(location_t));
    }
    
    writer.close();
    return file_size;
}

// Master save method
void Index::save(const char* filename, bool compact_before_save) {
    string prefix(filename);
    
    // Save data
    save_data(prefix + "_data.bin");
    
    // Save graph
    save_graph(prefix + "_graph.bin");
    
    // Save tags (if enabled)
    if (_enable_tags) {
        save_tags(prefix + "_tags.bin");
    }
    
    // Save metadata (JSON or binary)
    save_metadata(prefix + "_metadata.json");
}
```

**Key Features:**
- **Separate files** (data, graph, tags, metadata)
- **Rich metadata** (version, max_degree, start, num_frozen)
- **Compact option** (remove deleted points)
- **Abstraction** (data/graph stores handle their own I/O)

### Vamana I/O

```cpp
// Graph serialization
void Graph::save(const string& filename) const {
    ofstream file(filename, ios::binary);
    
    // Save number of nodes
    size_t num_nodes = adj_list.size();
    file.write((const char*)&num_nodes, sizeof(num_nodes));
    
    // Save adjacency list
    for (const auto& neighbors : adj_list) {
        size_t degree = neighbors.size();
        file.write((const char*)&degree, sizeof(degree));
        file.write((const char*)neighbors.data(), degree * sizeof(location_t));
    }
    
    file.close();
}

// Index serialization
void VamanaIndex::save_index(const string& filename) const {
    string resolved = resolve_dataset_path(filename);
    
    graph.save(resolved + ".graph");
    
    // Save metadata
    ofstream meta(resolved + ".meta", ios::binary);
    meta.write((const char*)&num_points, sizeof(num_points));
    meta.write((const char*)&dimension, sizeof(dimension));
    meta.write((const char*)&medoid, sizeof(medoid));
    meta.close();
}
```

**Key Features:**
- **Two files** (.graph, .meta)
- **Minimal metadata** (num_points, dimension, medoid)
- **No data serialization** (data external)
- **No versioning** or file format checking
- **Path resolution** (`resolve_dataset_path`)

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| File structure | Multiple (data, graph, tags, metadata) | Two (.graph, .meta) |
| Metadata | Rich (version, max_degree, start, frozen) | Minimal (num_points, dim, medoid) |
| Data ownership | DataStore saves data | Data external (not saved) |
| Versioning | Yes (file_version field) | No |
| Compaction | Optional (remove deleted points) | N/A |
| Abstraction | Stores handle their own I/O | Monolithic methods |

---

## Configurability and Extensibility

### DiskANN Configuration

```cpp
// Builder pattern for complex configuration
IndexConfig config = IndexConfigBuilder()
    .with_metric(Metric::L2)                              // Distance metric
    .with_dimension(128)                                  // Vector dimension
    .with_max_points(1000000)                             // Capacity
    .with_data_load_store_strategy(DataStoreStrategy::MEMORY)
    .with_graph_load_store_strategy(GraphStoreStrategy::MEMORY)
    .with_data_type("float")                              // Data type
    .with_tag_type("uint32")                              // Tag type
    .with_label_type("uint32")                            // Label type
    .is_dynamic_index(false)                              // Static/dynamic
    .is_enable_tags(false)                                // Tag support
    .is_pq_dist_build(false)                              // PQ compression
    .with_num_pq_chunks(0)                                // PQ chunks
    .is_use_opq(false)                                    // Optimized PQ
    .is_filtered(false)                                   // Filtered search
    .with_num_frozen_pts(1)                               // Frozen points
    .is_concurrent_consolidate(false)                     // Concurrent ops
    .with_index_write_params(build_params)                // Build params
    .with_index_search_params(search_params)              // Search params
    .build();
```

**Configuration Options (15+):**
- Metric (L2, cosine, MIPS)
- Data/graph storage strategy
- Data/tag/label types
- Dynamic vs static
- Tag support
- PQ compression
- Frozen points
- Concurrent operations
- Build parameters (R, L, alpha, threads)
- Search parameters

### Vamana Configuration

```cpp
// Constructor parameters (5 options)
VamanaIndex index(
    size_t dim,                                           // Vector dimension
    size_t R = DEFAULT_R,                                 // Max degree (32)
    size_t L = DEFAULT_L,                                 // Search list size (100)
    float alpha = DEFAULT_ALPHA,                          // Alpha (1.2)
    size_t maxc = DEFAULT_MAXC                            // Max candidates (750)
);

// Runtime configuration
index.build(data, num_points, num_threads);               // Build threads
index.load_index(filename, num_threads);                  // Search threads
auto results = index.search(query, k, search_L);          // Per-query L

// Global distance flag
extern bool use_simd;  // Set before build/search
```

**Configuration Options (5):**
- Dimension (constructor)
- R (constructor, default 32)
- L (constructor, default 100)
- Alpha (constructor, default 1.2)
- Maxc (constructor, default 750)

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Configuration style | Builder pattern | Constructor parameters |
| Options count | 15+ | 5 |
| Type system | String-based types | Fixed (float, uint32_t) |
| Distance metrics | Multiple (L2, cosine, MIPS) | L2 only |
| Storage backends | Pluggable (memory, disk) | Memory only |
| PQ compression | Yes (optional) | No |
| Dynamic index | Yes | No |
| Tag/label support | Yes | No |
| Extensibility | High (strategy pattern) | Low (modify source) |

---

## Memory Management

### DiskANN Memory

```cpp
// Aligned allocation for SIMD
template<typename T>
class InMemDataStore {
    T* _data;  // SIMD-aligned
    
    InMemDataStore(size_t capacity, size_t dimension) {
        _data = (T*)aligned_alloc(SIMD_ALIGNMENT, capacity * dimension * sizeof(T));
    }
    
    ~InMemDataStore() {
        if (_data) aligned_free(_data);
    }
};

// Graph with over-allocation
class InMemGraphStore {
    vector<vector<location_t>> _graph;
    size_t _reserve_graph_degree;  // Max degree * slack factor (1.3)
    
    InMemGraphStore(size_t capacity, size_t reserve_degree) 
        : _reserve_graph_degree(reserve_degree) {
        _graph.reserve(capacity);
        for (size_t i = 0; i < capacity; i++) {
            _graph.emplace_back();
            _graph[i].reserve(reserve_degree);  // Pre-allocate
        }
    }
};

// Scratch pool (heap-allocated, reused)
ConcurrentQueue<InMemQueryScratch<T>*> _query_scratch;

~Index() {
    // Clean up scratch pool
    InMemQueryScratch<T>* scratch;
    while (_query_scratch.try_pop(scratch)) {
        delete scratch;
    }
}
```

**Key Features:**
- **Aligned allocation** for SIMD (`aligned_alloc`)
- **Pre-allocation** with slack factor
- **Scratch pooling** (heap objects, reused)
- **Explicit cleanup** in destructor

### Vamana Memory

```cpp
// External data (not owned)
class VamanaIndex {
    float* data;  // Raw pointer (externally owned)
    
    void set_data(float* data_ptr, size_t points) {
        data = data_ptr;
        num_points = points;
    }
    
    ~VamanaIndex() {
        // data is owned by caller, don't delete it
    }
};

// Graph (dynamic vectors)
class Graph {
    vector<vector<location_t>> adj_list;  // No pre-allocation
    
    void add_edge(location_t from, location_t to) {
        if (from >= adj_list.size()) {
            resize(from + 1);  // Grow on demand
        }
        adj_list[from].push_back(to);  // Dynamic growth
    }
};

// Scratch spaces (unique_ptr, thread-local)
vector<unique_ptr<ScratchSpace>> build_scratch;
vector<unique_ptr<ScratchSpace>> search_scratch;

void initialize_search_scratch(size_t num_threads) {
    search_scratch.clear();
    for (size_t i = 0; i < num_threads; i++) {
        search_scratch.push_back(make_unique<ScratchSpace>());
    }
}

~VamanaIndex() {
    // unique_ptr auto-cleanup
}
```

**Key Features:**
- **No alignment** (standard allocation)
- **On-demand growth** (no slack factor)
- **External data ownership** (raw pointer)
- **Automatic cleanup** (`unique_ptr`)

### Comparison

| Aspect | DiskANN | Vamana |
|--------|---------|--------|
| Data alignment | SIMD-aligned (`aligned_alloc`) | No alignment (standard) |
| Data ownership | DataStore owns | External (raw pointer) |
| Graph allocation | Pre-allocated with slack | On-demand growth |
| Scratch management | Pool (ConcurrentQueue) | Vector (`unique_ptr`) |
| Cleanup | Explicit (destructor) | Automatic (`unique_ptr`) |
| Memory overhead | Higher (slack, alignment) | Lower (minimal overhead) |

---

## Recommendations

### Immediate (Low Risk)

1. ✅ **Keep Vamana's threading model:** Direct vector indexing is simpler and sufficient for static index
2. ✅ **Keep separate build/search scratch:** Clearer separation of concerns  
3. ⚠️ **Adopt DiskANN's search algorithm:** Dual-queue beam search for better convergence
4. ⚠️ **Add search termination criterion:** Implement early stopping like DiskANN
5. ⚠️ **Fix schedule parameter:** Change from `dynamic,2048` to `dynamic,1` to match DiskANN

### Medium Priority (Medium Risk)

6. **Add per-node locking:** If planning dynamic index support
7. **Implement RAII for scratch:** For exception safety (low priority for C++)
8. **Add file format versioning:** For backwards compatibility
9. **Expand metadata:** Save R, L, alpha, maxc in .meta file
10. **Add input validation:** Check dimensions, bounds, null pointers

### Long Term (High Risk / Major Refactoring)

11. **Add abstraction layers:** If need multiple storage backends (memory/disk)
12. **Implement builder pattern:** For complex configuration
13. **Support multiple metrics:** Cosine, MIPS (requires distance abstraction)
14. **Add PQ compression:** For memory efficiency (large datasets)
15. **Dynamic index support:** Requires concurrent data structures

### Not Recommended

- ❌ **Don't switch to ConcurrentQueue:** Adds complexity without clear benefit for static index
- ❌ **Don't force SIMD alignment:** Flexible memory layout is fine for current use
- ❌ **Don't adopt full DiskANN architecture:** Over-engineering for research prototype

---

## Summary

**Vamana's simplicity is its strength.** It provides a clear, understandable implementation of the Vamana/DiskANN algorithm without unnecessary abstraction. The architecture is well-suited for:

- Research and prototyping
- Educational purposes
- Static index use cases
- Single-machine deployments

**DiskANN's complexity enables production features** like dynamic indexing, multiple storage backends, and extensive configurability. These are valuable for:

- Production systems
- Large-scale deployments
- Multiple use cases with one codebase
- Concurrent operations

**Key takeaway:** Adopt specific algorithmic improvements (search, pruning) from DiskANN, but maintain Vamana's architectural simplicity unless specific features (dynamic index, disk-based storage) are required.
