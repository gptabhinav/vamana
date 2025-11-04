# Parallel Optimization Plan

**Goal:** Reduce index building time from ~50s to <5s for SIFT-10K through OpenMP parallelization

**Current Performance:** ~50 seconds (sequential, single-threaded)  
**Target Performance:** <5 seconds (parallel, 16+ threads)  
**Expected Speedup:** 10-15x with modern multi-core CPU

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

**Objective:** Add thread management without changing algorithm behavior

**Duration:** 30-45 minutes  
**Risk:** Low  
**Merge Criteria:** Code compiles, existing tests pass, no performance change

### Implementation Details

#### 1.1 Modify `include/vamana/core/index.h`

**Add to private members:**
```cpp
// Thread management
size_t num_threads;
std::vector<std::unique_ptr<ScratchSpace>> thread_scratch;
```

**Update constructor signature:**
```cpp
VamanaIndex(size_t dim, size_t R = DEFAULT_R, size_t L = DEFAULT_L,
            float alpha = DEFAULT_ALPHA, size_t maxc = DEFAULT_MAXC,
            size_t num_threads = 0);  // 0 = use all available cores
```

**Add helper method:**
```cpp
private:
    void initialize_thread_pool();
```

#### 1.2 Modify `src/core/index.cpp`

**Update constructor:**
```cpp
VamanaIndex::VamanaIndex(size_t dim, size_t R, size_t L, float alpha, 
                         size_t maxc, size_t num_threads) 
    : data(nullptr), num_points(0), dimension(dim), medoid(0),
      R(R), L(L), alpha(alpha), maxc(maxc), num_threads(num_threads) {
    
    // Initialize single scratch for now (backwards compatible)
    scratch = std::make_unique<ScratchSpace>();
    
    // Determine thread count
    if (this->num_threads == 0) {
        #ifdef _OPENMP
        this->num_threads = omp_get_max_threads();
        #else
        this->num_threads = 1;
        #endif
    }
    
    // Initialize thread pool
    initialize_thread_pool();
}

void VamanaIndex::initialize_thread_pool() {
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

#### 1.3 Modify `apps/build_memory_index.cpp`

**Add num_threads parameter:**
```cpp
uint32_t num_threads = 0;  // 0 = use all cores

// In argument parsing:
else if (arg == "--num_threads" || arg == "-T") {
    num_threads = std::stoul(argv[++i]);
}

// Update VamanaIndex construction:
VamanaIndex index(data_dim, R, L, alpha, 500, num_threads);
```

**Update usage message:**
```cpp
std::cout << "  -T, --num_threads   Number of threads (0 = use all cores)" << std::endl;
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

- Add num_threads parameter to VamanaIndex
- Create thread-local scratch space pool
- Add CLI argument for thread count
- No algorithm changes - pure infrastructure

Validation: Index build time unchanged, results identical"

git push -u origin feature/parallel-iter1-infrastructure

# 9. Create PR and merge to main
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
**Risk:** Medium (potential race conditions)  
**Merge Criteria:** 5-10x speedup, recall within 2% of baseline

### Implementation Details

#### 2.1 Modify `src/core/index.cpp` - `build()` function

**Before (sequential):**
```cpp
// Iterative improvement
for (size_t i = 0; i < num_points; i++) {
    search_and_prune(i);
    
    if (i % 1000 == 0) {
        std::cout << "Processed " << i << "/" << num_points << " nodes" << std::endl;
    }
}
```

**After (parallel):**
```cpp
// Iterative improvement - PARALLEL
#pragma omp parallel for schedule(dynamic, 2048)
for (size_t i = 0; i < num_points; i++) {
    search_and_prune(i);
    
    // Thread-safe progress reporting
    #pragma omp critical(progress_output)
    {
        if (i % 1000 == 0) {
            std::cout << "Processed " << i << "/" << num_points << " nodes" << std::endl;
        }
    }
}
```

**Key points:**
- `schedule(dynamic, 2048)`: Dynamic scheduling with 2048-node chunks for load balancing
- Critical section around I/O to prevent garbled output
- No changes to `search_and_prune` internals yet

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
    
    auto& local_scratch = thread_scratch[thread_id];
    
    const float* query = data + location * dimension;
    auto candidates = greedy_search(query, L, medoid, local_scratch.get());
    
    auto& pruned = local_scratch->result_buffer;
    // ... rest of function uses local_scratch instead of scratch
}
```

#### 2.3 Update method signatures to accept scratch pointer

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

**Objective:** Add proper synchronization for concurrent graph modifications

**Duration:** 30-45 minutes  
**Risk:** Medium (performance vs correctness)  
**Merge Criteria:** No race conditions, recall <1% difference

### Implementation Details

#### 3.1 Add critical sections for graph updates

**In `search_and_prune()`:**

```cpp
void VamanaIndex::search_and_prune(location_t location) {
    // ... existing search and prune logic ...
    
    // CRITICAL: Protect graph update
    #pragma omp critical(graph_update)
    {
        graph.set_neighbors(location, pruned);
    }
    
    // Reverse link insertion
    for (location_t neighbor : pruned) {
        #pragma omp critical(graph_update)
        {
            auto neighbor_list = graph.get_neighbors(neighbor);
            std::vector<location_t> updated_neighbors(neighbor_list.begin(), 
                                                      neighbor_list.end());
            updated_neighbors.push_back(location);
            
            if (updated_neighbors.size() > R) {
                // Re-prune neighbor's list
                // ... pruning logic ...
                graph.set_neighbors(neighbor, pruned_neighbors);
            } else {
                graph.set_neighbors(neighbor, updated_neighbors);
            }
        }
    }
}
```

**Alternative: Use OpenMP locks for finer granularity**

If critical sections cause bottlenecks, implement per-node locks:

```cpp
// In index.h:
std::vector<omp_lock_t> node_locks;

// Initialize in constructor:
node_locks.resize(num_points);
for (size_t i = 0; i < num_points; i++) {
    omp_init_lock(&node_locks[i]);
}

// In search_and_prune:
omp_set_lock(&node_locks[location]);
graph.set_neighbors(location, pruned);
omp_unset_lock(&node_locks[location]);
```

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
- ✅ Consistent results across multiple runs
- ✅ Performance not significantly degraded (<20% slower than iter2)
- ✅ Recall within 1% of single-threaded

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
