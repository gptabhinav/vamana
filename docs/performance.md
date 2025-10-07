# Performance Analysis

Benchmarking and optimization guide for the Vamana implementation.

## Performance Characteristics

### SIMD Optimizations
The implementation includes AVX2-optimized L2 distance calculations:
- **3.3x speedup** over scalar implementation
- Automatic detection and fallback to scalar on older hardware
- Memory-aligned data loading for optimal SIMD performance

### Index Persistence
- **18,600x faster** loading compared to construction  
- Index construction: ~71 seconds for SIFT-10K
- Index loading: ~4 milliseconds for SIFT-10K

### Memory Usage
- **Graph storage**: ~N × R × 4 bytes (where N=points, R=max degree)
- **Runtime memory**: Additional scratch space for search operations
- **SIMD alignment**: Data aligned to 32-byte boundaries

## Benchmarking Results

### SIFT-10K Dataset (10,000 vectors, 128D)

**Construction Performance** (R=32, L=64, α=1.2):
- Build time: 71.5 seconds
- Index size: ~1.3 MB
- Memory usage: ~52 MB during construction

**Search Performance**:
| L   | QPS   | Recall@1 | Recall@10 |
|-----|-------|----------|-----------|
| 10  | 847   | 0.91     | 0.915     |
| 20  | 465   | 1.00     | 0.988     |
| 50  | 196   | 1.00     | 1.000     |
| 100 | 101   | 1.00     | 1.000     |
| 200 | 53    | 1.00     | 1.000     |

## Optimization Guidelines

### Compilation Flags
For optimal performance, compile with:
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -mavx2 -mfma" ..
```

### Hardware Considerations
- **CPU**: Modern x86-64 with AVX2 support recommended
- **Memory**: Ensure sufficient RAM (dataset size + index size + scratch space)
- **Storage**: SSD recommended for faster index loading

### Dataset-Specific Optimizations

**High-Dimensional Data (>500D)**:
- SIMD benefits increase with dimensionality
- Consider batch processing for construction
- Monitor memory usage during construction

**Large Datasets (>1M points)**:
- Construction becomes I/O bound
- Consider using multiple threads for data loading
- May benefit from disk-based construction for very large datasets

## Profiling and Debugging

### Build Configuration
For profiling builds:
```bash
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
```

### Common Performance Issues

**Slow Construction**:
- Reduce construction L parameter
- Ensure release build with optimizations
- Check memory availability

**Slow Search**:
- Use lower search L values  
- Verify SIMD is being used (check CPU flags)
- Ensure data is properly aligned

**High Memory Usage**:
- Reduce max degree R
- Monitor scratch space usage
- Consider streaming for very large datasets

## Scaling Analysis

### Construction Time Complexity
Empirically: O(N × L × log(N))
- N: Number of points
- L: Construction search list size

### Search Time Complexity  
Empirically: O(L × log(degree))
- L: Search list size
- Degree: Average node degree in graph

### Memory Complexity
- **Index**: O(N × R) for graph structure
- **Runtime**: O(L + K) for search operations

## Comparison with Other Methods

### vs. Brute Force
- **Construction**: One-time cost vs. zero
- **Search**: ~100-1000x faster at high recall levels
- **Memory**: Higher (graph + data) vs. just data

### vs. LSH
- **Recall**: Generally higher at same speed
- **Memory**: Comparable or better
- **Tuning**: Fewer parameters to optimize

### vs. IVF/Product Quantization
- **Recall**: Better for high-dimensional data
- **Speed**: Competitive, especially with SIMD
- **Memory**: Uses more memory (stores full vectors)

## Performance Monitoring

### Metrics to Track
- **QPS (Queries Per Second)**: Primary speed metric
- **Recall@K**: Primary accuracy metric  
- **Index Size**: Storage overhead
- **Construction Time**: One-time cost
- **Memory Usage**: Runtime resource requirements

### Automated Benchmarking
```bash
# Run comprehensive evaluation
for R in 16 32 64; do
  for L in 32 64 128; do
    for alpha in 1.0 1.2 1.5; do
      echo "Testing R=$R L=$L alpha=$alpha"
      ./build/apps/build_memory_index ... -R $R -L $L --alpha $alpha
      ./build/apps/search_memory_index ... -L 10 20 50 100 --result_path "results_${R}_${L}_${alpha}.csv"
    done
  done
done
```

## Optimization Roadmap

### Current Optimizations
- ✅ AVX2 SIMD for distance calculations
- ✅ Memory alignment for optimal SIMD usage  
- ✅ Efficient binary formats
- ✅ Scratch space pooling

### Potential Future Optimizations
- Multi-threading during construction
- GPU acceleration for distance calculations
- Compressed graph storage
- Disk-based construction for massive datasets
- Dynamic index updates