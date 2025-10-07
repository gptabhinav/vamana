# File Format Reference

Documentation for the binary file formats used by the Vamana implementation.

## .fbin Format (Float Binary)

Used for storing vector datasets (base vectors, query vectors).

### Structure
```
Header:
  4 bytes: number of vectors (N) [uint32_t, little-endian]  
  4 bytes: dimension (D) [uint32_t, little-endian]

Data:
  N × D × 4 bytes: vector data [float32, little-endian, row-major]
```

### Memory Layout
```
Vector 0: [f0, f1, f2, ..., f(D-1)]
Vector 1: [f0, f1, f2, ..., f(D-1)]  
...
Vector (N-1): [f0, f1, f2, ..., f(D-1)]
```

### Reading .fbin Files
```cpp
#include "vamana/core/io.h"

float* data;
uint32_t num_vectors, dimension;
vamana::io::load_fbin("dataset.fbin", data, num_vectors, dimension);

// Use data...

vamana::io::free_data(data);  // Clean up
```

### Writing .fbin Files
```cpp
#include "vamana/core/io.h"

float* data = ...; // Your vector data
uint32_t num_vectors = 1000;
uint32_t dimension = 128;

vamana::io::save_fbin("dataset.fbin", data, num_vectors, dimension);
```

## .ibin Format (Integer Binary)

Used for storing ground truth data (nearest neighbor IDs).

### Structure
```
Header:
  4 bytes: number of queries (N) [uint32_t, little-endian]
  4 bytes: k value (K) [uint32_t, little-endian]

Data:  
  N × K × 4 bytes: neighbor IDs [uint32_t, little-endian, row-major]
```

### Memory Layout
```
Query 0 neighbors: [id0, id1, id2, ..., id(K-1)]
Query 1 neighbors: [id0, id1, id2, ..., id(K-1)]
...
Query (N-1) neighbors: [id0, id1, id2, ..., id(K-1)]
```

### Reading .ibin Files
```cpp
#include "vamana/core/io.h"

uint32_t* groundtruth;
uint32_t num_queries, k;
vamana::io::load_ibin("groundtruth.ibin", groundtruth, num_queries, k);

// Access ground truth for query i: groundtruth + i * k

std::free(groundtruth);  // Clean up
```

### Writing .ibin Files  
```cpp
#include "vamana/core/io.h"

uint32_t* groundtruth = ...; // Your ground truth data
uint32_t num_queries = 100;  
uint32_t k = 10;

vamana::io::save_ibin("groundtruth.ibin", groundtruth, num_queries, k);
```

## Index Files

### .graph Format

Binary file storing the graph structure:
```
4 bytes: number of nodes (N)
4 bytes: dimension (D) 
4 bytes: max degree (R)

For each node i (0 to N-1):
  4 bytes: actual degree of node i
  degree × 4 bytes: neighbor IDs
```

### .meta Format  

Text file with index metadata:
```
dimension=128
num_points=10000
max_degree=32
medoid_id=5847
```

## Legacy Format Conversion

### From .fvecs Format
```bash  
./build/apps/utils/fvecs_to_fbin \
  --input_file vectors.fvecs \
  --output_file vectors.fbin
```

### From .ivecs Format
```bash
./build/apps/utils/ivecs_to_ibin \
  --input_file groundtruth.ivecs \
  --output_file groundtruth.ibin  
```

## Memory Alignment

The .fbin loader automatically ensures proper memory alignment for SIMD operations:
- Data is aligned to 32-byte boundaries for AVX2
- Use `vamana::io::free_data()` to properly deallocate aligned memory

## Endianness

All binary formats use little-endian byte order. The implementation automatically handles endianness conversion on big-endian systems.

## File Size Calculations

### .fbin Files
Size = 8 + (N × D × 4) bytes
- Example: 10,000 vectors × 128D = 8 + (10,000 × 128 × 4) = 5,120,008 bytes

### .ibin Files  
Size = 8 + (N × K × 4) bytes
- Example: 100 queries × 100 neighbors = 8 + (100 × 100 × 4) = 40,008 bytes

### .graph Files
Size ≈ 12 + (N × avg_degree × 4) bytes
- Actual size depends on degree distribution after pruning