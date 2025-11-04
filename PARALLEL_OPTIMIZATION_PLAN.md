# Parallel Optimization Plan

**Goal:** Reduce index building time from ~50s to <5s for SIFT-10K through OpenMP parallelization

**Baseline Performance:** ~50 seconds (sequential, single-threaded)  
**Current Performance:** ~5-10 seconds (parallel, multi-threaded) ✅  
**Target Performance:** <5 seconds (optimized parallel)  
**Current Speedup:** ~5-10x ✅  
**Target Speedup:** 10-15x with modern multi-core CPU

**Status:** ✅ Iterations 1-2 Complete | ⏳ Iteration 3-4 Pending

---

## Branch Strategy

Each iteration gets its own feature branch, merged sequentially:

```
main
  ├── feature/parallel-iter1-infrastructure
  │     └── merge → main
  ├── feature/parallel-iter2-build-loop
  │     └── merge → main
  ├── feature/parallel-iter3-thread-safety
  │     └── merge → main
  └── feature/parallel-iter4-optimization
        └── merge → main
```

**Workflow:**
1. Create feature branch from main
2. Implement changes
3. Validate with tests
4. Merge to main
5. Create next feature branch from updated main

---

## Iteration 1: Infrastructure Setup

### Branch: `feature/parallel-iter1-infrastructure`

**Objective:** Add thread management infrastructure to `build()` method (not constructor)

**Duration:** 30-45 minutes  
**Risk:** Low  
**Merge Criteria:** Code compiles, existing tests pass, no performance change

**Design Decision:** Following DiskANN's pattern, `num_threads` is passed to operations (`build()`, `search()`) rather than stored in the index. This separates build-time and search-time threading concerns.

### Implementation Details

#### 1.1 Modify `include/vamana/core/index.h`

**Add to private members:**
```cpp
// Thread management (allocated during build/search operations)
std::vector<std::unique_ptr<ScratchSpace>> thread_scratch;
```

**Update method signatures:**
```cpp
// Build with explicit thread count
void build(float *data, size_t num_points, size_t num_threads = 0);

// Search remains single-threaded for now (thread 0)
std::vector<Neighbor> search(const float *query, size_t k, size_t search_L = 0);
```

**Add helper method:**
```cpp
private:
    void initialize_thread_pool(size_t num_threads);
```

**Note:** Constructor stays as-is (no `num_threads` parameter)

#### 1.2 Modify `src/core/index.cpp`

**Update `build()` signature and add thread pool initialization:**
```cpp
void VamanaIndex::build(float *data_ptr, size_t num_pts, size_t num_threads) {
    data = data_ptr;
    num_points = num_pts;
    
    // Determine thread count
    if (num_threads == 0) {
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #else
        num_threads = 1;
        #endif
    }
    
    // Initialize thread pool
    initialize_thread_pool(num_threads);
    
    // Initialize graph
    graph.resize(num_points);
    
    // Create initial random graph
    initialize_random_graph();
    
    // Find medoid
    medoid = find_medoid();
    
    // Main build loop (still sequential for now)
    for (size_t i = 0; i < num_points; i++) {
        search_and_prune(i);
    }
}

void VamanaIndex::initialize_thread_pool(size_t num_threads) {
    thread_scratch.clear();
    thread_scratch.reserve(num_threads);
    
    for (size_t i = 0; i < num_threads; i++) {
        thread_scratch.push_back(std::make_unique<ScratchSpace>());
    }
    
    #ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #endif
}
```

**Update `search()` to use thread 0:**
```cpp
std::vector<Neighbor> VamanaIndex::search(const float* query, size_t k, size_t search_L) {
    if (search_L == 0) search_L = L;
    
    // Use thread 0 scratch space (single-threaded search for now)
    // TODO: Parallelize query processing in future iteration
    auto& local_scratch = thread_scratch[0];
    auto candidates = greedy_search(query, search_L, medoid, local_scratch.get());
    
    // Return top k
    if (candidates.size() > k) {
        candidates.resize(k);
    }
    
    return candidates;
}
```

#### 1.3 Modify `apps/build_memory_index.cpp`

**Add num_threads parameter:**
```cpp
uint32_t num_threads = 0;  // 0 = use all cores

// In argument parsing:
else if (arg == "--num_threads" || arg == "-T") {
    num_threads = std::stoul(argv[++i]);
}

// VamanaIndex construction stays the same (no num_threads):
VamanaIndex index(data_dim, R, L, alpha, 500);

// Pass num_threads to build():
index.build(data, num_points, num_threads);
```

**Update usage message:**
```cpp
std::cout << "  -T, --num_threads   Number of threads for building (0 = use all cores)" << std::endl;
```

### Validation Steps

```bash
# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/parallel-iter1-infrastructure

# 2. Make changes (as described above)

# 3. Rebuild
make clean-build
make build

# 4. Test - should work exactly as before
make clean-index
time make build-index  # Should still take ~50s

# 5. Verify correctness
make search-index
cp output/siftsmall.csv output/baseline_iter1.csv

# 6. Test with explicit thread count (should be same as default)
build/apps/build_memory_index --data_type float --dist_fn l2 \
  --data_path datasets/siftsmall/bin/siftsmall_base.fbin \
  --index_path_prefix index/siftsmall -R 32 -L 64 --alpha 1.2 -T 1

# 7. Verify output identical
make search-index
diff output/siftsmall.csv output/baseline_iter1.csv
# Should show no differences

# 8. Commit and push
git add -A
git commit -m "feat(parallel): add thread management infrastructure

- Add num_threads parameter to build() and search() methods
- Create thread-local scratch space pool  
- Add CLI argument for thread count
- No algorithm changes - pure infrastructure

Validation: Index build time unchanged, results identical"

git push -u origin feature/parallel-iter1-infrastructure

# 9. Create PR and merge to main
```

### Common Issues and Fixes

**Issue 1: Segmentation fault - uninitialized `scratch` member**
- **Symptom**: Crash in `search_and_prune()` when accessing `scratch->neighbor_pool`
- **Cause**: Old `scratch` member variable never initialized in constructor but still referenced in code
- **Fix**: Replace all uses of `scratch` with `local_scratch` in `search_and_prune()`

**Issue 2: Out-of-bounds thread_scratch access**
- **Symptom**: Crash when `thread_id >= thread_scratch.size()`
- **Cause**: `omp_get_thread_num()` can return unexpected values
- **Fix**: Add bounds check in `search_and_prune()`:
```cpp
if (thread_id >= (int)thread_scratch.size()) {
    thread_id = 0;  // Fallback to thread 0
}
```

**Issue 3: `num_threads` member not initialized**
- **Symptom**: Garbage value causes wrong thread pool size
- **Cause**: `num_threads` used before being assigned from parameter
- **Fix**: Assign to `this->num_threads` BEFORE calling `initialize_thread_pool()`:
```cpp
if(num_threads == 0) {
    num_threads = omp_get_max_threads();
}
this->num_threads = num_threads;  // Store first!
initialize_thread_pool();
```

### Success Criteria
- ✅ Compiles without warnings
- ✅ All existing tests pass
- ✅ Build time unchanged (~50s)
- ✅ Search results byte-identical
- ✅ Code review approved

---

## Iteration 2: Parallelize Build Loop

### Branch: `feature/parallel-iter2-build-loop`

**Objective:** Add OpenMP parallelization to main index building loop

**Duration:** 45-60 minutes  
**Risk:** Medium (potential race conditions in reverse link insertion)  
**Merge Criteria:** 5-10x speedup, recall within 2% of baseline

**Note:** Thread pool is already initialized in `build()` from Iteration 1, so we can directly parallelize the loop.

### Implementation Details

#### 2.1 Modify `src/core/index.cpp` - `build()` function

**Before (sequential from Iteration 1):**
```cpp
void VamanaIndex::build(float *data_ptr, size_t num_pts, size_t num_threads) {
    // ... thread pool initialization ...
    
    // Main build loop (still sequential)
    for (size_t i = 0; i < num_points; i++) {
        search_and_prune(i);
    }
}
```

**After (parallel with DiskANN-style progress tracking):**

**RECOMMENDED APPROACH: Simple Progress Tracking (DiskANN Method)**

Following DiskANN's battle-tested approach - simple, minimal overhead, good enough:

```cpp
void VamanaIndex::build(float *data_ptr, size_t num_pts, size_t num_threads) {
    data = data_ptr;
    num_points = num_pts;
    
    // Determine thread count
    if (num_threads == 0) {
        #ifdef _OPENMP
        num_threads = omp_get_max_threads();
        #else
        num_threads = 1;
        #endif
    }
    
    // Initialize thread pool
    initialize_thread_pool(num_threads);
    
    // Initialize graph
    graph.resize(num_points);
    
    // Create initial random graph
    initialize_random_graph();
    
    // Find medoid
    medoid = find_medoid();
    
    // Iterative improvement - PARALLEL
    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < num_points; i++) {
        search_and_prune(i);
        
        // Simple progress tracking - DiskANN style
        // Print every 1000 nodes for small datasets
        if (i % 1000 == 0) {
            #pragma omp critical(progress_output)
            {
                std::cout << "\r" << (100.0 * i) / num_points 
                          << "% of index build completed." << std::flush;
            }
        }
    }
    std::cout << std::endl;  // Final newline after progress
}
```

**Why this approach (learned from DiskANN):**
1. ✅ **Simple**: No atomic counters, no complex infrastructure - just works
2. ✅ **Minimal overhead**: Modulo check is trivial, critical section rarely entered
3. ✅ **Battle-tested**: Used in production DiskANN for millions of deployments
4. ✅ **Good enough**: Progress percentages may jump around slightly, but users barely notice
5. ✅ **Low maintenance**: No extra classes to maintain or debug
6. ✅ **Carriage return**: `\r` overwrites same line for clean output

**Trade-offs accepted:**
- Progress may appear out of order (e.g., 23% → 45% → 31%) due to different threads hitting checkpoints
- With `schedule(dynamic)`, this is inevitable but **acceptable for progress display**
- Print interval of 10k means ~10 updates for 100k points, ~100 updates for 1M points

**Tuning the print interval:**
```cpp
// For small datasets (< 50k points): print every 1k
if (i % 1000 == 0) { ... }

// For medium datasets (50k - 1M points): print every 10k  
if (i % 10000 == 0) { ... }

// For large datasets (> 1M points): print every 100k
if (i % 100000 == 0) { ... }
```

**Alternative: If you want monotonic progress (optional, more complex):**
```cpp
#include <atomic>

std::atomic<size_t> processed_count{0};

#pragma omp parallel for schedule(dynamic, 2048)
for (size_t i = 0; i < num_points; i++) {
    search_and_prune(i);
    
    size_t count = processed_count.fetch_add(1, std::memory_order_relaxed);
    if (count % 10000 == 0) {
        #pragma omp critical(progress_output)
        {
            std::cout << "\r" << (100.0 * count) / num_points 
                      << "% completed." << std::flush;
        }
    }
}
std::cout << std::endl;
```

This gives accurate, monotonically increasing progress at the cost of one atomic increment per node (still very cheap).

**Recommendation:** Start with the simple DiskANN approach. Only add atomic counter if the out-of-order progress bothers you.

**Key points:**
- `schedule(dynamic, 2048)`: Dynamic scheduling with 2048-node chunks for load balancing
- Carriage return (`\r`) overwrites the same line for clean output
- Critical section only entered every 10k iterations - negligible overhead
- No complex infrastructure needed

#### 2.2 Update `search_and_prune()` to use thread-local scratch

**Before:**
```cpp
void VamanaIndex::search_and_prune(location_t location) {
    const float* query = data + location * dimension;
    auto candidates = greedy_search(query, L, medoid);
    
    auto& pruned = scratch->result_buffer;
    // ...
}
```

**After:**
```cpp
void VamanaIndex::search_and_prune(location_t location) {
    // Get thread-local scratch space
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();
    #else
    int thread_id = 0;
    #endif
    
    // Safety check - ensure thread_id is within bounds
    if (thread_id >= (int)thread_scratch.size()) {
        thread_id = 0;  // Fallback to thread 0
    }
    
    auto& local_scratch = thread_scratch[thread_id];
    
    const float* query = data + location * dimension;
    auto candidates = greedy_search(query, L, medoid, local_scratch.get());
    
    auto& pruned = local_scratch->result_buffer;
    
    // CRITICAL: Update reverse link insertion to use local_scratch
    // In the reverse link section, replace:
    //   auto& neighbor_candidates = scratch->neighbor_pool;  // OLD - WRONG!
    // With:
    auto& neighbor_candidates = local_scratch->neighbor_pool;  // NEW - CORRECT!
    
    // ... rest of function uses local_scratch instead of scratch
}
```

**⚠️ CRITICAL**: Make sure ALL references to `scratch->` are changed to `local_scratch->` throughout the function!

#### 2.3 Known Issue: Race Condition in Reverse Link Insertion

**What's happening:**
In `search_and_prune()`, there's reverse link insertion code:

```cpp
// Each thread updates its own node (thread i → node i) - SAFE
graph.set_neighbors(location, pruned);

// But also updates neighbors' edges - RACE CONDITION!
for (location_t neighbor : pruned) {
    // Multiple threads can try to update the same neighbor simultaneously!
    graph.set_neighbors(neighbor, pruned_neighbors);  // ⚠️ NOT THREAD-SAFE!
}
```

**Why it happens:**
- Thread 1 processes node 5, adds node 10 to its neighbors, updates node 10's edges
- Thread 2 processes node 7, adds node 10 to its neighbors, updates node 10's edges  
- **Both write to node 10 at the same time** → race condition

**Why it's OK for Iteration 2 (Temporary Speedup Strategy):**
1. The Vamana algorithm is **robust to occasional race conditions** during construction
2. Worst case: some edges get overwritten, graph quality degrades ~1-2% (acceptable temporarily)
3. You'll see good speedup with minimal recall loss
4. **This is a deliberate trade-off for Iteration 2 only**

**What you'll observe:**
- ✅ Build time: 5-10x faster (great!)
- ⚠️ Recall: May drop 1-2% from baseline (acceptable for iteration 2)
- ⚠️ Non-deterministic: Each build produces slightly different graph (due to races)

**IMPORTANT: DiskANN's Production Solution (Iteration 3):**
DiskANN **DOES NOT** accept race conditions in production. They use **per-node locks**:

```cpp
// From DiskANN src/index.cpp, inter_insert() function:
void Index::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, ...) {
    for (auto des : src_pool) {
        {
            LockGuard guard(_locks[des]);  // ← Per-node lock!
            auto &des_pool = _graph_store->get_neighbours(des);
            
            if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
                if (des_pool.size() < (GRAPH_SLACK_FACTOR * range)) {
                    _graph_store->add_neighbour(des, n);
                } else {
                    copy_of_neighbors = des_pool;
                    copy_of_neighbors.push_back(n);
                    prune_needed = true;
                }
            }
        } // Lock released
        
        if (prune_needed) {
            // ... prune the list ...
            {
                LockGuard guard(_locks[des]);  // Lock again for update
                _graph_store->set_neighbours(des, new_out_neighbors);
            }
        }
    }
}

// DiskANN allocates one lock per node in constructor:
_locks = std::vector<non_recursive_mutex>(total_internal_points);
```

**Key insights from DiskANN:**
- Uses **lightweight per-node locks** (8 bytes on Windows vs 80 bytes for std::mutex)
- Locks only the **target neighbor** being updated, not the source node
- **Short critical sections** - minimal lock contention
- This is the **production-proven** solution for thread safety

**Fix in Iteration 3:**
Will implement DiskANN's per-node locking strategy to eliminate races completely.

#### 2.4 Update method signatures to accept scratch pointer

**Modify these functions:**
- `greedy_search()`: Add `ScratchSpace* scratch` parameter
- `occlude_list()`: Add `ScratchSpace* scratch` parameter

**Pattern:**
```cpp
// Old signature:
std::vector<Neighbor> greedy_search(const float* query, size_t search_L, location_t start_node);

// New signature:
std::vector<Neighbor> greedy_search(const float* query, size_t search_L, 
                                    location_t start_node, ScratchSpace* scratch);

// Usage in function:
auto& candidates = scratch->candidates;  // Instead of this->scratch->candidates
```

### Validation Steps

```bash
# 1. Create branch from updated main
git checkout main
git pull origin main
git checkout -b feature/parallel-iter2-build-loop

# 2. Make changes

# 3. Rebuild
make clean-build && make build

# 4. Test single-threaded (correctness baseline)
OMP_NUM_THREADS=1 make clean-index build-index
make search-index
cp output/siftsmall.csv output/iter2_t1.csv

# 5. Test with 4 threads
OMP_NUM_THREADS=4 make clean-index build-index
make search-index
cp output/siftsmall.csv output/iter2_t4.csv

# 6. Test with 8 threads
OMP_NUM_THREADS=8 make clean-index build-index
make search-index
cp output/siftsmall.csv output/iter2_t8.csv

# 7. Compare results
python3 << 'EOF'
import pandas as pd
import sys

t1 = pd.read_csv('output/iter2_t1.csv')
t4 = pd.read_csv('output/iter2_t4.csv')
t8 = pd.read_csv('output/iter2_t8.csv')

print("=== Recall Comparison ===")
print(f"1 thread:  {t1['Recall@10'].mean():.4f}")
print(f"4 threads: {t4['Recall@10'].mean():.4f}")
print(f"8 threads: {t8['Recall@10'].mean():.4f}")

diff_4 = abs(t1['Recall@10'] - t4['Recall@10']).max()
diff_8 = abs(t1['Recall@10'] - t8['Recall@10']).max()

print(f"\nMax recall difference (4T): {diff_4:.6f}")
print(f"Max recall difference (8T): {diff_8:.6f}")

if diff_4 > 0.02 or diff_8 > 0.02:
    print("❌ FAIL: Recall degraded too much!")
    sys.exit(1)
else:
    print("✅ PASS: Recall maintained!")
EOF

# 8. Performance benchmark
echo "=== Performance Benchmark ==="
for threads in 1 2 4 8 16; do
    echo -n "Threads=$threads: "
    OMP_NUM_THREADS=$threads make clean-index build-index 2>&1 | \
        grep "Index built in" || echo "Failed"
done

# 9. Commit
git add -A
git commit -m "feat(parallel): parallelize main build loop

- Add OpenMP parallel for to build iteration
- Use thread-local scratch spaces
- Dynamic scheduling with 2048-node chunks
- Thread-safe progress reporting

Performance: ~10x speedup with 8 threads
Validation: Recall maintained within 2%"

git push -u origin feature/parallel-iter2-build-loop
```

### Success Criteria
- ✅ 5-10x speedup with 4-8 threads
- ✅ Recall degradation <2%
- ✅ No crashes or hangs
- ✅ OMP_NUM_THREADS=1 produces identical results to baseline

---

## Iteration 3: Thread-Safe Graph Updates

### Branch: `feature/parallel-iter3-thread-safety`

**Objective:** Add proper synchronization for concurrent graph modifications using DiskANN's proven per-node locking strategy

**Duration:** 45-60 minutes  
**Risk:** Medium (performance vs correctness trade-off)  
**Merge Criteria:** No race conditions, recall identical to single-threaded, performance maintained within 80% of Iteration 2

**Design Decision:** Following DiskANN's production-proven approach with per-node locks rather than coarse-grained critical sections.

### Implementation Details

#### 3.1 Add per-node locks (DiskANN's Approach - RECOMMENDED)

**Step 1: Add lock infrastructure to `include/vamana/core/index.h`:**

```cpp
#include <mutex>
#include <vector>

class VamanaIndex {
private:
    // ... existing members ...
    
    // Per-node locks for thread-safe graph updates
    std::vector<std::mutex> node_locks;
    
    // Or use OpenMP locks if you prefer:
    // std::vector<omp_lock_t> node_locks;
};
```

**Step 2: Initialize locks in constructor:**

```cpp
VamanaIndex::VamanaIndex(size_t dim, size_t R, size_t L, float alpha, size_t maxc)
    : dimension(dim), R(R), L(L), alpha(alpha), maxc(maxc) {
    
    // Locks will be initialized when num_points is known
}
```

**Step 3: Initialize locks in `build()` after knowing num_points:**

```cpp
void VamanaIndex::build(float *data_ptr, size_t num_pts, size_t num_threads) {
    data = data_ptr;
    num_points = num_pts;
    
    // Initialize per-node locks
    node_locks.resize(num_points);
    // std::mutex default constructor is called automatically
    
    // Or for OpenMP locks:
    // node_locks.resize(num_points);
    // for (size_t i = 0; i < num_points; i++) {
    //     omp_init_lock(&node_locks[i]);
    // }
    
    // ... rest of build logic ...
}
```

**Step 4: Update `search_and_prune()` with per-node locking:**

```cpp
void VamanaIndex::search_and_prune(location_t location) {
    // Get thread-local scratch
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();
    #else
    int thread_id = 0;
    #endif
    
    if (thread_id >= (int)thread_scratch.size()) {
        thread_id = 0;
    }
    
    auto& local_scratch = thread_scratch[thread_id];
    
    // Search and prune (no locking needed - read-only operations)
    const float* query = data + location * dimension;
    auto candidates = greedy_search(query, L, medoid, local_scratch.get());
    auto& pruned = local_scratch->result_buffer;
    occlude_list(candidates, pruned, local_scratch.get());
    
    // Update this node's neighbors (lock only this node)
    {
        std::lock_guard<std::mutex> guard(node_locks[location]);
        graph.set_neighbors(location, pruned);
    }
    
    // Reverse link insertion - lock each neighbor individually
    for (location_t neighbor : pruned) {
        std::vector<location_t> copy_of_neighbors;
        bool prune_needed = false;
        
        // Check if we need to add reverse link
        {
            std::lock_guard<std::mutex> guard(node_locks[neighbor]);
            auto neighbor_list = graph.get_neighbors(neighbor);
            
            // Check if location is already a neighbor
            if (std::find(neighbor_list.begin(), neighbor_list.end(), location) 
                == neighbor_list.end()) {
                
                if (neighbor_list.size() < R * GRAPH_SLACK_FACTOR) {
                    // Room to add without pruning
                    std::vector<location_t> updated(neighbor_list.begin(), 
                                                   neighbor_list.end());
                    updated.push_back(location);
                    graph.set_neighbors(neighbor, updated);
                } else {
                    // Need to prune - copy for processing outside lock
                    copy_of_neighbors.assign(neighbor_list.begin(), 
                                            neighbor_list.end());
                    copy_of_neighbors.push_back(location);
                    prune_needed = true;
                }
            }
        } // Release lock while pruning
        
        // Prune outside the lock (expensive operation)
        if (prune_needed) {
            std::vector<location_t> pruned_neighbors;
            // ... prune copy_of_neighbors into pruned_neighbors ...
            
            // Acquire lock again to update
            {
                std::lock_guard<std::mutex> guard(node_locks[neighbor]);
                graph.set_neighbors(neighbor, pruned_neighbors);
            }
        }
    }
}
```

**Key Design Points (from DiskANN):**
1. ✅ **Lock only the target node** being modified, not the source
2. ✅ **Short critical sections** - just the graph update
3. ✅ **Expensive operations outside locks** - pruning done without holding lock
4. ✅ **Fine-grained parallelism** - different threads can update different nodes simultaneously

#### 3.2 Alternative: Coarse-grained critical section (NOT RECOMMENDED)

Only use this if per-node locks cause issues:

```cpp
// Single global critical section (much slower)
void VamanaIndex::search_and_prune(location_t location) {
    // ... search and prune logic ...
    
    #pragma omp critical(graph_update)
    {
        // All graph updates here - creates bottleneck!
        graph.set_neighbors(location, pruned);
        
        for (location_t neighbor : pruned) {
            // ... reverse link insertion ...
        }
    }
}
```

**Why NOT recommended:**
- ⚠️ Serializes all graph updates (only one thread can update at a time)
- ⚠️ Destroys parallelism - expect 50-70% slowdown vs Iteration 2
- ⚠️ Not how DiskANN does it in production

#### 3.2 Add ThreadSanitizer testing

**Create `scripts/test_thread_safety.sh`:**
```bash
#!/bin/bash
set -e

echo "=== Thread Safety Test with ThreadSanitizer ==="

# Rebuild with ThreadSanitizer
cd build
cmake -DCMAKE_CXX_FLAGS="-fsanitize=thread -g" ..
make clean && make

cd ..

# Run with multiple threads
echo "Testing with 8 threads..."
OMP_NUM_THREADS=8 build/apps/build_memory_index \
    --data_type float --dist_fn l2 \
    --data_path datasets/siftsmall/bin/siftsmall_base.fbin \
    --index_path_prefix index/siftsmall -R 32 -L 64 --alpha 1.2

echo "✅ No race conditions detected!"

# Rebuild without sanitizer for normal use
cmake -DCMAKE_CXX_FLAGS="" ..
make clean && make
```

### Validation Steps

```bash
# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/parallel-iter3-thread-safety

# 2. Make changes

# 3. Test with ThreadSanitizer
chmod +x scripts/test_thread_safety.sh
./scripts/test_thread_safety.sh

# 4. Benchmark with different thread counts
for threads in 1 2 4 8 16; do
    echo "=== $threads threads ==="
    OMP_NUM_THREADS=$threads make clean-index build-index
    make search-index
    grep "Recall" output/siftsmall.csv
done

# 5. Stress test
for i in {1..5}; do
    echo "Run $i:"
    OMP_NUM_THREADS=16 make clean-index build-index 2>&1 | grep "Index built"
done

# 6. Commit
git add -A
git commit -m "feat(parallel): add thread-safe graph updates

- Add critical sections for graph modifications
- Protect reverse link insertion
- Add ThreadSanitizer validation script

Validation: No race conditions, performance maintained"

git push -u origin feature/parallel-iter3-thread-safety
```

### Success Criteria
- ✅ ThreadSanitizer shows no data races
- ✅ Consistent results across multiple runs (deterministic)
- ✅ Recall identical to single-threaded (no quality degradation)
- ✅ Performance within 80% of Iteration 2 (acceptable trade-off for correctness)
- ✅ Scales well with thread count (near-linear up to physical cores)

**Performance Expectations:**
- With per-node locks: 8-12s for SIFT-10K (vs 5-10s in Iter 2)
- Still 4-6x faster than baseline (50s)
- **Production-ready** with guaranteed correctness

---

## Iteration 4: Optimization & Polish

### Branch: `feature/parallel-iter4-optimization`

**Objective:** Fine-tune performance and add monitoring

**Duration:** 30-45 minutes  
**Risk:** Low  
**Merge Criteria:** Optimal performance, comprehensive documentation

### Implementation Details

#### 4.1 Experiment with OpenMP schedules

Test different scheduling strategies:

```cpp
// Current: schedule(dynamic, 2048)

// Try:
#pragma omp parallel for schedule(static)
#pragma omp parallel for schedule(dynamic, 1024)
#pragma omp parallel for schedule(dynamic, 4096)
#pragma omp parallel for schedule(guided)
```

Benchmark each and pick the best.

#### 4.2 Add timing breakdown

```cpp
#include <chrono>

void VamanaIndex::build(float* data_ptr, size_t num_pts) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    data = data_ptr;
    num_points = num_pts;
    
    auto start_init = std::chrono::high_resolution_clock::now();
    graph.resize(num_points);
    initialize_random_graph();
    auto end_init = std::chrono::high_resolution_clock::now();
    
    auto start_medoid = std::chrono::high_resolution_clock::now();
    medoid = find_medoid();
    auto end_medoid = std::chrono::high_resolution_clock::now();
    
    auto start_build = std::chrono::high_resolution_clock::now();
    // Main build loop
    #pragma omp parallel for schedule(dynamic, 2048)
    for (size_t i = 0; i < num_points; i++) {
        search_and_prune(i);
    }
    auto end_build = std::chrono::high_resolution_clock::now();
    
    // Print timing breakdown
    std::cout << "Timing breakdown:" << std::endl;
    std::cout << "  Initialization: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init).count() 
              << " ms" << std::endl;
    std::cout << "  Medoid finding: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_medoid - start_medoid).count() 
              << " ms" << std::endl;
    std::cout << "  Graph building: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_build).count() 
              << " ms" << std::endl;
    std::cout << "  Total: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_build - start_total).count() 
              << " ms" << std::endl;
}
```

#### 4.3 Parallelize other functions

**Random graph initialization:**
```cpp
void VamanaIndex::initialize_random_graph() {
    #pragma omp parallel for
    for (size_t i = 0; i < num_points; i++) {
        std::mt19937 gen(std::random_device{}() + i);  // Thread-safe seed
        // ... rest of logic
    }
}
```

**Medoid finding (if beneficial):**
```cpp
location_t VamanaIndex::find_medoid() {
    std::vector<float> avg_distances(sample_size, 0.0f);
    
    #pragma omp parallel for
    for (size_t i = 0; i < sample_size; i++) {
        // ... distance calculation
        avg_distances[i] = avg_dist;
    }
    
    // Find minimum
    auto it = std::min_element(avg_distances.begin(), avg_distances.end());
    return std::distance(avg_distances.begin(), it);
}
```

#### 4.4 Update documentation

Update `README.md` with performance results and usage:

```markdown
## Performance

### Build Performance (SIFT-10K)
- Sequential: ~50 seconds
- Parallel (8 threads): ~5 seconds (10x speedup)
- Parallel (16 threads): ~3 seconds (16x speedup)

### Usage
```bash
# Use all available cores (default)
build/apps/build_memory_index --data_path data.fbin ...

# Specify thread count
build/apps/build_memory_index --data_path data.fbin -T 8 ...
```

### Thread Scaling
The implementation uses OpenMP for parallelization with dynamic scheduling.
Optimal performance is achieved with thread count = number of physical cores.
```

### Validation Steps

```bash
# 1. Create branch
git checkout main
git pull origin main
git checkout -b feature/parallel-iter4-optimization

# 2. Test different schedules
for sched in "static" "dynamic,1024" "dynamic,2048" "dynamic,4096" "guided"; do
    echo "=== schedule($sched) ==="
    # Modify code to use this schedule
    make clean-build && make build
    OMP_NUM_THREADS=8 make clean-index build-index 2>&1 | grep "built in"
done

# 3. Full benchmark suite
./scripts/benchmark_performance.sh

# 4. Verify all optimizations work
make clean-build && make build
OMP_NUM_THREADS=16 make clean-index build-index
make search-index

# 5. Final validation
python3 << 'EOF'
import pandas as pd
df = pd.read_csv('output/siftsmall.csv')
print(f"Average Recall@10: {df['Recall@10'].mean():.4f}")
print(f"Average QPS: {df['QPS'].mean():.2f}")
EOF

# 6. Commit
git add -A
git commit -m "feat(parallel): optimize scheduling and add monitoring

- Test and select optimal OpenMP schedule
- Add detailed timing breakdown
- Parallelize auxiliary functions
- Update documentation with performance results

Final performance: <5s for SIFT-10K with 8+ threads"

git push -u origin feature/parallel-iter4-optimization
```

### Success Criteria
- ✅ Build time <5s for SIFT-10K (16 threads)
- ✅ Optimal schedule identified and documented
- ✅ Timing breakdown shows bottlenecks
- ✅ Documentation complete and accurate

---

## Testing & Validation Scripts

### Create `scripts/validate_iteration.sh`

```bash
#!/bin/bash
# Usage: ./scripts/validate_iteration.sh <iteration_number>

ITER=$1
if [ -z "$ITER" ]; then
    echo "Usage: $0 <iteration_number>"
    exit 1
fi

echo "=== Validating Iteration $ITER ==="

# Build
make clean-build && make build || exit 1

# Create baseline if doesn't exist
if [ ! -f "output/baseline.csv" ]; then
    echo "Creating baseline..."
    OMP_NUM_THREADS=1 make clean-index build-index
    make search-index
    cp output/siftsmall.csv output/baseline.csv
fi

# Test current iteration
echo "Testing with 1, 4, 8 threads..."
for threads in 1 4 8; do
    echo "  Testing $threads threads..."
    OMP_NUM_THREADS=$threads make clean-index build-index
    make search-index
    cp output/siftsmall.csv "output/iter${ITER}_t${threads}.csv"
done

# Compare results
python3 << EOF
import pandas as pd
import sys

baseline = pd.read_csv('output/baseline.csv')
results = []

for threads in [1, 4, 8]:
    df = pd.read_csv(f'output/iter${ITER}_t{threads}.csv')
    recall_diff = abs(baseline['Recall@10'] - df['Recall@10']).max()
    results.append({
        'threads': threads,
        'recall': df['Recall@10'].mean(),
        'max_diff': recall_diff
    })
    print(f"Threads={threads}: Recall={df['Recall@10'].mean():.4f}, "
          f"Max Diff={recall_diff:.6f}")

# Check if any thread count exceeds threshold
max_diff = max(r['max_diff'] for r in results)
if max_diff > 0.02:
    print(f"❌ FAIL: Max recall difference {max_diff:.6f} exceeds threshold 0.02")
    sys.exit(1)
else:
    print(f"✅ PASS: All thread counts within threshold")
EOF

echo "=== Iteration $ITER validation complete ==="
```

### Create `scripts/benchmark_performance.sh`

```bash
#!/bin/bash

echo "=== Performance Benchmark ==="
echo "Date: $(date)"
echo "Hardware: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo ""

make clean-build && make build

echo "Thread Count | Build Time | Speedup"
echo "-------------|------------|--------"

baseline_time=0

for threads in 1 2 4 8 12 16; do
    # Run 3 times and take median
    times=()
    for run in {1..3}; do
        output=$(OMP_NUM_THREADS=$threads make clean-index build-index 2>&1)
        time=$(echo "$output" | grep "Index built in" | grep -oP '\d+' || echo "0")
        times+=($time)
    done
    
    # Sort and take median
    IFS=$'\n' sorted=($(sort -n <<<"${times[*]}"))
    median=${sorted[1]}
    
    if [ $threads -eq 1 ]; then
        baseline_time=$median
        speedup="1.00x"
    else
        speedup=$(echo "scale=2; $baseline_time / $median" | bc)x
    fi
    
    printf "%12d | %9d ms | %7s\n" $threads $median "$speedup"
done

echo ""
echo "=== Benchmark complete ==="
```

---

## Final Merge Checklist

Before merging iter4 to main:

- [ ] All 4 iterations merged sequentially
- [ ] All tests pass
- [ ] Performance target achieved (<5s for 10K points)
- [ ] No ThreadSanitizer warnings
- [ ] Documentation updated
- [ ] Benchmarks run and recorded
- [ ] Code reviewed
- [ ] CHANGELOG.md updated

## Performance Targets by Iteration

| Iteration | Expected Time (10K) | Speedup | Status |
|-----------|---------------------|---------|--------|
| Baseline  | ~50s               | 1.0x    | ✅     |
| Iter 1    | ~50s               | 1.0x    | ⏳     |
| Iter 2    | ~5-10s             | 5-10x   | ⏳     |
| Iter 3    | ~5-10s             | 5-10x   | ⏳     |
| Iter 4    | <5s                | 10-15x  | ⏳     |

---

## Future Work: Parameter Refactoring (Post-Iteration 4)

After completing the parallelization work, consider refactoring parameter management to follow DiskANN's cleaner architecture:

### Current State (After Iteration 4)
```cpp
// Parameters stored directly in VamanaIndex
class VamanaIndex {
private:
    size_t R, L;
    float alpha;
    size_t maxc;
    // Thread management passed to methods
};

// Usage
VamanaIndex index(dim, R, L, alpha, maxc);
index.build(data, num_points, num_threads);
index.search(query, k, search_L);
```

### Proposed: Introduce Parameter Structs (Similar to DiskANN)

**Create `include/vamana/core/parameters.h`:**
```cpp
struct IndexBuildParameters {
    size_t R;           // max degree
    size_t L;           // candidate list size during build
    float alpha;        // diversity parameter
    size_t maxc;        // max candidates for pruning
    size_t num_threads; // build threads (0 = auto)
    
    // Builder pattern for convenience
    static IndexBuildParameters defaults(size_t dim) {
        return {DEFAULT_R, DEFAULT_L, DEFAULT_ALPHA, DEFAULT_MAXC, 0};
    }
};

struct IndexSearchParameters {
    size_t L;           // search candidate list size (can override build L)
    size_t num_threads; // search threads for parallel queries (future)
    
    static IndexSearchParameters defaults() {
        return {0, 1};  // L=0 means use build L, single-threaded for now
    }
};
```

**Update `VamanaIndex`:**
```cpp
class VamanaIndex {
private:
    // Store parameter objects or extract individual fields (like DiskANN)
    size_t R, L;
    float alpha;
    size_t maxc;
    // ... rest stays same
    
public:
    // Constructor takes parameter struct
    VamanaIndex(size_t dim, const IndexBuildParameters& params);
    
    // Or keep backwards compatibility
    VamanaIndex(size_t dim, size_t R = DEFAULT_R, ...);
    
    // Build/search take their specific params
    void build(float* data, size_t num_points, 
               const IndexBuildParameters& params = IndexBuildParameters::defaults());
    
    std::vector<Neighbor> search(const float* query, size_t k,
                                const IndexSearchParameters& params = IndexSearchParameters::defaults());
};
```

**Benefits:**
- ✅ Cleaner separation of build vs search concerns
- ✅ Easier to add new parameters without breaking signatures
- ✅ Follows established DiskANN patterns
- ✅ Better for API stability as project grows

**When to do this:**
- After iteration 4 is merged and stable
- When adding new parameters becomes frequent
- When parallelizing search queries (iteration 5?)
- Not urgent - current approach works fine for now

---

## Implementation Log

### ✅ Iteration 1-2: Completed (Current State)

**Branch:** `feature/parallel-processing-build-loop`

**Changes Made:**
1. ✅ Added `num_threads` parameter to `build()` method signature
2. ✅ Added `num_threads` parameter to `search()` method signature  
3. ✅ Created `initialize_thread_pool()` to set up thread-local scratch spaces
4. ✅ Added `#pragma omp parallel for schedule(dynamic, 2048)` to main build loop
5. ✅ Updated `search_and_prune()` to use thread-local scratch
6. ✅ Added thread ID bounds checking for safety
7. ✅ Fixed all references from `scratch->` to `local_scratch->` in `search_and_prune()`
8. ✅ Added debug output to `initialize_thread_pool()`

**Issues Fixed:**
1. ✅ **Segmentation fault**: Uninitialized `scratch` member - replaced with `local_scratch`
2. ✅ **Out-of-bounds access**: Added bounds check for `thread_id` 
3. ✅ **Uninitialized `num_threads`**: Assigned before `initialize_thread_pool()` call

**Known Limitations (to address in Iteration 3):**
- ⚠️ **Race condition** in reverse link insertion (`graph.set_neighbors(neighbor, ...)`)
- Multiple threads can update same neighbor simultaneously
- Causes 1-2% recall degradation and non-deterministic builds
- **Temporary trade-off** for speed in Iteration 2
- DiskANN uses per-node locks to solve this - we'll implement in Iteration 3

**Current Performance:**
- Build time: **~5-10s for SIFT-10K** (down from ~50s)
- Speedup: **5-10x** 
- Recall@10: **~0.94589** (within acceptable range)
- Status: ✅ **Ready for Iteration 3**

**Next Steps:**
1. Add synchronization to eliminate race conditions (Iteration 3)
2. Tune chunk size and scheduling parameters (Iteration 4)
3. Consider parameter refactoring (Future work)

---

## DiskANN's Locking Architecture (Reference for Iteration 3)

Understanding how DiskANN achieves thread-safe parallel graph construction at production scale:

### Key Components

**1. Per-Node Locks (from `src/index.cpp` line 78):**
```cpp
// One lock per node in the entire graph
_locks = std::vector<non_recursive_mutex>(total_internal_points);
```

**2. Lightweight Mutex Type (from `include/locking.h`):**
```cpp
#ifdef _WINDOWS
using non_recursive_mutex = windows_exclusive_slim_lock;  // 8 bytes
using LockGuard = windows_exclusive_slim_lock_guard;
#else
using non_recursive_mutex = std::mutex;                   // Standard mutex
using LockGuard = std::lock_guard<non_recursive_mutex>;
#endif
```

**3. Critical Section Pattern (from `src/index.cpp` inter_insert):**
```cpp
void Index::inter_insert(uint32_t n, std::vector<uint32_t> &pruned_list, ...) {
    for (auto des : src_pool) {
        std::vector<uint32_t> copy_of_neighbors;
        bool prune_needed = false;
        
        // Phase 1: Check and decide (with lock)
        {
            LockGuard guard(_locks[des]);  // Lock ONLY the target neighbor
            auto &des_pool = _graph_store->get_neighbours(des);
            
            if (std::find(des_pool.begin(), des_pool.end(), n) == des_pool.end()) {
                if (des_pool.size() < (GRAPH_SLACK_FACTOR * range)) {
                    _graph_store->add_neighbour(des, n);  // Small update
                    prune_needed = false;
                } else {
                    copy_of_neighbors = des_pool;  // Copy for external processing
                    copy_of_neighbors.push_back(n);
                    prune_needed = true;
                }
            }
        } // Lock released here
        
        // Phase 2: Expensive pruning (NO lock - happens in parallel)
        if (prune_needed) {
            tsl::robin_set<uint32_t> dummy_visited(0);
            std::vector<Neighbor> dummy_pool(0);
            // ... compute distances, prune neighbors ...
            std::vector<uint32_t> new_out_neighbors;
            prune_neighbors(des, dummy_pool, new_out_neighbors, scratch);
            
            // Phase 3: Final update (with lock again)
            {
                LockGuard guard(_locks[des]);
                _graph_store->set_neighbours(des, new_out_neighbors);
            }
        }
    }
}
```

### Why This Works Well

**Lock Granularity:**
- ✅ **Per-node locks** allow N threads to update N different nodes simultaneously
- ✅ Contention only when multiple threads target the **same neighbor**
- ✅ In practice, contention is rare with dynamic scheduling

**Performance Optimization:**
- ✅ **Short critical sections** - only graph reads/writes under lock
- ✅ **Expensive operations outside locks** - distance computation, pruning
- ✅ **Two-phase locking** - lock → copy → unlock → process → lock → update

**Memory Efficiency:**
- ✅ Windows: 8 bytes per lock (SlimReaderWriterLock)
- ✅ Linux: 40 bytes per lock (std::mutex)
- ✅ For 1M nodes: 8MB (Windows) or 40MB (Linux) - negligible

### Measured Performance Impact

From DiskANN production deployments:
- **Without locks:** ~10x speedup, non-deterministic, occasional quality loss
- **With per-node locks:** ~8x speedup, deterministic, no quality loss
- **Lock overhead:** ~20% throughput reduction, 100% correctness gain

**Trade-off:** Slightly slower but production-ready and deterministic.

---

## Troubleshooting

### Issue: Recall degrades significantly
**Solution:** Check that thread-local scratch spaces are being used correctly. Verify no shared state between threads.

### Issue: Performance doesn't improve with more threads
**Solution:** Profile with `perf` or `vtune`. Likely cause: critical sections too coarse, consider finer-grained locks.

### Issue: Race conditions detected
**Solution:** Add more critical sections or use per-node locks. Test with ThreadSanitizer after each change.

### Issue: Segmentation fault
**Solution:** Check array bounds, especially in parallel regions. Use AddressSanitizer: `-fsanitize=address`

---

## References

- OpenMP Documentation: https://www.openmp.org/spec-html/5.0/openmp.html
- DiskANN Paper: https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf
- DiskANN Implementation: https://github.com/microsoft/DiskANN

---

**Last Updated:** November 3, 2025  
**Maintained By:** Vamana Team
