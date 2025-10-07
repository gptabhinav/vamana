# Vamana Applications

Command-line applications for the Vamana approximate nearest neighbor search algorithm.

## Applications

### Core Applications

#### `build_memory_index`
Builds a Vamana index from base dataset.

```bash
./build_memory_index --data_type float --dist_fn l2 \
  --data_path dataset.fbin --index_path_prefix index_name \
  -R 32 -L 64 --alpha 1.2
```

**Parameters:**
- `--data_type`: Data type (currently `float` only)
- `--dist_fn`: Distance function (currently `l2` only)  
- `--data_path`: Input dataset in .fbin format
- `--index_path_prefix`: Output index prefix (creates .graph and .meta files)
- `-R`: Max out-degree per node (typically 32-64)
- `-L`: Search list size during construction (typically 64-128)
- `--alpha`: Pruning parameter (typically 1.0-1.2)

#### `search_memory_index`
Searches the index with multiple L values for performance analysis.

```bash
./search_memory_index --data_type float --dist_fn l2 \
  --data_path dataset.fbin --index_path_prefix index_name \
  --query_file queries.fbin --gt_file groundtruth.ibin \
  -K 10 -L 10 20 50 100 --result_path results.csv
```

**Parameters:**
- `--data_path`: Base dataset used to build the index (.fbin format)
- `--index_path_prefix`: Index prefix to load
- `--query_file`: Query vectors (.fbin format)
- `--gt_file`: Ground truth file (.ibin format) for recall evaluation
- `-K`: Number of nearest neighbors to return
- `-L`: Space-separated list of search list sizes to test
- `--result_path`: Output CSV file with QPS and recall results

### Utility Applications

#### `fvecs_to_fbin`
Converts legacy .fvecs format to DiskANN .fbin format.

```bash
./fvecs_to_fbin --input_file vectors.fvecs --output_file vectors.fbin
```

#### `ivecs_to_ibin`
Converts legacy .ivecs format to DiskANN .ibin format.

```bash  
./ivecs_to_ibin --input_file groundtruth.ivecs --output_file groundtruth.ibin
```

#### `compute_groundtruth`
Computes brute-force ground truth for evaluation datasets.

```bash
./compute_groundtruth --data_type float --dist_fn l2 \
  --base_file dataset.fbin --query_file queries.fbin \
  --gt_file groundtruth.ibin -K 100
```

## File Formats

### .fbin Format (DiskANN Float Binary)
Binary format for float vectors:
- 4 bytes: number of vectors (N)
- 4 bytes: dimension (D)  
- N×D×4 bytes: vector data (row-major, float32)

### .ibin Format (DiskANN Integer Binary)  
Binary format for ground truth data:
- 4 bytes: number of queries (N)
- 4 bytes: k value (number of neighbors per query)
- N×k×4 bytes: neighbor IDs (row-major, uint32)

### Index Files
- `.graph`: Binary graph structure with adjacency lists
- `.meta`: Metadata including dimension, parameters, and medoid ID

## Performance Features

- **SIMD Optimizations**: AVX2 acceleration for L2 distance calculations (3.3x speedup)
- **Memory Alignment**: Aligned memory allocation for optimal SIMD performance
- **Index Persistence**: Fast loading (18,600x faster than building)
- **Clean CLI**: Standard command-line interfaces
- **Comprehensive Evaluation**: Built-in recall metrics with ground truth comparison

## Example Workflow

```bash
# 1. Convert legacy formats to DiskANN formats
./fvecs_to_fbin --input_file sift_base.fvecs --output_file sift_base.fbin
./fvecs_to_fbin --input_file sift_query.fvecs --output_file sift_queries.fbin  
./ivecs_to_ibin --input_file sift_groundtruth.ivecs --output_file sift_groundtruth.ibin

# 2. Build index
./build_memory_index --data_type float --dist_fn l2 \
  --data_path sift_base.fbin --index_path_prefix sift_index \
  -R 32 -L 64 --alpha 1.2

# 3. Search and evaluate  
./search_memory_index --data_type float --dist_fn l2 \
  --data_path sift_base.fbin --index_path_prefix sift_index \
  --query_file sift_queries.fbin --gt_file sift_groundtruth.ibin \
  -K 10 -L 10 20 50 100 --result_path results.csv
```

## Building

All applications are built automatically when building the main project:

```bash
mkdir build && cd build
cmake .. && make
```

This creates all executables in the build directory, ready for deployment.