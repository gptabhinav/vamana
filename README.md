# Vamana

C++17 implementation of the Vamana algorithm for approximate nearest neighbor search in high-dimensional spaces.

## What is this?

Vamana is a graph-based algorithm that builds an index for fast approximate nearest neighbor search. It creates a navigable graph where each node represents a data point, then uses greedy search to quickly find approximate neighbors.

## Requirements

- C++17 compiler (GCC 7+, Clang 5+)
- CMake 3.12+
- OpenMP (optional)

## Building

```bash
mkdir build && cd build
cmake .. && make
```

For optimized build:
```bash
cmake -DCMAKE_BUILD_TYPE=Release .. && make
```

## Testing

```bash
# Run all tests
ctest

# Run specific tests
./test/test_index
./test/integration_tests
```

## Quick Example

Complete workflow using SIFT dataset:

```bash
# 1. Get SIFT dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz

# 2. Convert to binary format
./apps/utils/fvecs_to_fbin --input_file siftsmall/siftsmall_base.fvecs --output_file sift_base.fbin
./apps/utils/fvecs_to_fbin --input_file siftsmall/siftsmall_query.fvecs --output_file sift_queries.fbin
./apps/utils/ivecs_to_ibin --input_file siftsmall/siftsmall_groundtruth.ivecs --output_file sift_gt.ibin

# 3. Build index
./apps/build_memory_index --data_type float --dist_fn l2 \
  --data_path sift_base.fbin --index_path_prefix sift_index \
  -R 32 -L 64 --alpha 1.2

# 4. Search with different configurations
./apps/search_memory_index --data_type float --dist_fn l2 \
  --data_path sift_base.fbin --index_path_prefix sift_index \
  --query_file sift_queries.fbin --gt_file sift_gt.ibin \
  -K 10 -L 10 20 50 100 --result_path results.csv
```

**Key Parameters:**
- `R` (32): Max degree per node during construction
- `L` (64): Search list size during construction  
- `alpha` (1.2): Pruning parameter (higher = more diversity)
- `maxc` (750): Max candidates for pruning during construction
- Search `L` (10,20,50,100): Search list sizes for querying (higher = better recall, slower)

**Index Files:** Your index will be saved as `{prefix}.graph` and `{prefix}.meta`

## Easier Workflow Example using MAKE

```bash
# Specify your dataset in config.mk
# Current prebuilt support is for sift and siftsmall
# For other datasets use the workflow described above
dataset = siftsmall
```

```bash
# Do a fresh start
make clean

# creates the build files
make build

# downloads and builds the dataset binaries
make dataset

# build index on the dataset
make build-index

# search index and save results in a csv file (check output directory)
# this right now uses K=10, and a list of different beamwidths to compare on L = 10, 20, 50, 100
make search-index
```

## Directory Structure

```
vamana/
├── apps/                          # Command-line application implementations
│   ├── build_memory_index.cpp     # Index builder
│   ├── search_memory_index.cpp    # Index searcher  
│   └── utils/                     # Utility programs
│       ├── fvecs_to_fbin.cpp      # Format converter (.fvecs → .fbin)
│       ├── ivecs_to_ibin.cpp      # Format converter (.ivecs → .ibin)
│       └── compute_groundtruth.cpp # Ground truth computation
├── include/vamana/core/           # Headers
│   ├── index.h                    # VamanaIndex class
│   ├── graph.h                    # Graph operations
│   ├── distance.h                 # Distance functions (SIMD optimized)
│   ├── neighbor.h                 # Neighbor utilities
│   ├── scratch.h                  # Memory management
│   ├── io.h                       # File I/O (.fbin/.ibin formats)
│   └── types.h                    # Type definitions
├── src/core/                      # Implementation corresponding to headers in include/vamana/core/
│   ├── index.cpp                  # Main algorithm
│   ├── graph.cpp                  # Graph construction
│   ├── distance.cpp               # Distance computations
│   └── io.cpp                     # Binary file handling
├── scripts/                       # Utility script implementations
│   ├── download_datasets.sh       # Download script for common datasets like sift, siftsmall
├── test/                          # Test suite
├── docs/                          # Documentation and Examples
```

## Applications

After building, executables are organized in `build/`:
- `apps/build_memory_index` - Build Vamana index from dataset
- `apps/search_memory_index` - Search index with recall evaluation  
- `apps/utils/fvecs_to_fbin` - Convert .fvecs files to .fbin format
- `apps/utils/ivecs_to_ibin` - Convert .ivecs files to .ibin format
- `apps/utils/compute_groundtruth` - Generate ground truth for evaluation

## Examples and Documentation

- [Complete SIFT Tutorial](docs/sift_tutorial.md) - Step-by-step SIFT dataset example
- [Parameter Tuning Guide](docs/parameters.md) - How to configure R, L, alpha
- [File Format Reference](docs/formats.md) - .fbin/.ibin format specifications
- [Performance Analysis](docs/performance.md) - Benchmarking and optimization tips
- [Downloading Datasets](docs/download_datasets.md) - Downloading standard datasets like sift, siftsmall, etc. 

## Algorithm

Vamana builds a graph where each node represents a data point. During construction, it uses diversity-aware pruning (occlude_list) to select good neighbors. For search, it performs greedy traversal starting from a computed medoid to find approximate nearest neighbors efficiently.
