# Vamana - High-Performance Approximate Nearest Neighbor Search

A C++17 implementation of the Vamana algorithm for efficient approximate nearest neighbor (ANN) search in high-dimensional spaces.

## Features

- **Fast Graph-based Search**: Implements the Vamana algorithm with diversity-aware neighbor selection
- **SIMD Optimizations**: AVX2-optimized distance calculations for improved performance
- **Memory Efficient**: Scratch space pooling to minimize allocations during search
- **Comprehensive Testing**: Individual unit tests for each component plus integration tests
- **CMake Build System**: Modern C++ build configuration with OpenMP support

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+)
- CMake 3.12 or higher
- OpenMP (optional, for parallel operations)

## Building the Project

### 1. Clone and Setup
```bash
git clone <repository-url>
cd vamana
mkdir build
cd build
```

### 2. Configure and Build
```bash
# Configure with CMake
cmake ..

# Build the project
make

# Or build with parallel jobs
make -j$(nproc)
```

### 3. Build Options

For optimized release build:
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

For debug build with symbols:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Testing the Project

The project includes comprehensive tests organized into individual unit tests and integration tests.

### Running All Tests
```bash
# From the build directory
ctest --verbose
```

### Running Individual Unit Tests
```bash
# From build/test directory
cd test

# Test individual components
./test_neighbor     # Neighbor struct tests
./test_distance     # Distance function tests
./test_graph        # Graph structure tests
./test_scratch      # Scratch space tests
./test_index        # VamanaIndex class tests
```

### Running Integration Tests
```bash
# From build/test directory
./integration_tests  # End-to-end pipeline tests
```

### Test Summary
The test suite includes:
- **Unit Tests**: 5 separate test executables for each core component
- **Integration Tests**: End-to-end workflow validation
- **Coverage**: Construction, basic operations, edge cases, and full pipeline testing

## Running the Demo Program

After building, you can run the demo program:
```bash
# From the build directory
./vamana
```

> **Note**: If the executable builds as `vamana_overnight`, delete `src/CMakeLists.txt` and rebuild to use the main CMakeLists.txt configuration.

## Usage Example

```cpp
#include "vamana/core/index.h"

// Create dataset (example: 1000 points in 128 dimensions)
const size_t num_points = 1000;
const size_t dimension = 128;
float* data = new float[num_points * dimension];
// ... fill data with your vectors ...

// Create Vamana index
VamanaIndex index(dimension, 32, 64, 1.2f, 500);

// Build the index
index.build(data, num_points);

// Search for k nearest neighbors
float query[128];  // your query vector
auto results = index.search(query, 10);  // find 10 nearest neighbors

// Results contain Neighbor objects with id and distance
for (const auto& neighbor : results) {
    std::cout << "ID: " << neighbor.id 
              << ", Distance: " << neighbor.distance << std::endl;
}
```

## Project Structure

```
vamana/
├── CMakeLists.txt              # Main build configuration
├── README.md                   # This file
├── include/vamana/core/        # Header files
│   ├── index.h                 # Main VamanaIndex class
│   ├── graph.h                 # Graph structure
│   ├── neighbor.h              # Neighbor struct and utilities
│   ├── distance.h              # Distance functions
│   ├── scratch.h               # Scratch space for operations
│   └── types.h                 # Type definitions
├── src/                        # Implementation files
│   ├── main.cpp               # Example/demo program
│   └── core/                  # Core algorithm implementations
└── test/                      # Test suite
    ├── CMakeLists.txt         # Test build configuration
    ├── test_neighbor.cpp      # Neighbor unit tests
    ├── test_distance.cpp      # Distance function tests
    ├── test_graph.cpp         # Graph structure tests
    ├── test_scratch.cpp       # Scratch space tests
    ├── test_index.cpp         # Index class tests
    └── integration_tests.cpp  # End-to-end tests
```

## Algorithm Overview

The Vamana algorithm builds a graph-based index for approximate nearest neighbor search:

1. **Graph Construction**: Creates a directed graph where each node represents a data point
2. **Diversity-Aware Pruning**: Uses the `occlude_list` algorithm to select diverse neighbors
3. **Greedy Search**: Traverses the graph to find approximate nearest neighbors efficiently
4. **Medoid-based Entry**: Uses a computed medoid as the starting point for searches

## Performance Notes

- Compile with `-O3 -march=native -mavx2` for best performance
- The implementation uses AVX2 SIMD instructions when available
- OpenMP is used for parallel operations during index construction
- Scratch space pooling minimizes memory allocations during search

## Quick Reference

### Essential Commands
```bash
# Build everything
mkdir build && cd build
cmake .. && make

# Run all tests
ctest

# Run demo program
./vamana

# Run specific unit test
./test/test_index
```

### Project Status
- ✅ Core Vamana algorithm implemented
- ✅ Graph-based index construction
- ✅ Diversity-aware neighbor selection (occlude_list)
- ✅ Greedy graph search
- ✅ SIMD-optimized distance functions
- ✅ Comprehensive test suite (6 test executables)
- ✅ Memory-efficient scratch space management

## Contributing

1. Ensure all tests pass: `ctest`
2. Follow the existing code style
3. Add tests for new functionality
4. Update documentation as needed
