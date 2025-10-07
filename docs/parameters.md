# Parameter Tuning Guide

Understanding and tuning Vamana algorithm parameters for optimal performance.

## Construction Parameters

### R (Max Degree)
- **What it is**: Maximum number of outgoing edges per node in the graph
- **Typical values**: 16-64
- **Effect**: 
  - Higher R → Better recall, larger index, slower construction
  - Lower R → Faster construction, smaller index, lower recall
- **Recommendation**: Start with 32, increase to 64 for higher recall needs

### L (Construction Search List Size)  
- **What it is**: Size of candidate list during graph construction
- **Typical values**: 64-128 (should be ≥ R)
- **Effect**:
  - Higher L → Better graph quality, slower construction  
  - Lower L → Faster construction, lower quality graph
- **Recommendation**: Use 2×R as starting point

### Alpha (Pruning Parameter)
- **What it is**: Controls diversity in neighbor selection during pruning
- **Typical values**: 1.0-1.2
- **Effect**:
  - Lower alpha (1.0) → More diversity, better connectivity
  - Higher alpha (1.2+) → Less diversity, more direct connections
- **Recommendation**: 1.2 works well for most datasets

## Search Parameters

### Search L
- **What it is**: Size of candidate list during search
- **Typical values**: 10-200
- **Effect**:
  - Higher L → Better recall, slower search
  - Lower L → Faster search, lower recall
- **Tuning**: Test multiple values to find speed/accuracy sweet spot

### K (Number of Results)
- **What it is**: How many nearest neighbors to return
- **Effect**: Minimal impact on search time for reasonable values (≤100)

## Parameter Selection Strategy

### 1. Start with Defaults
```bash
# Good starting point for most datasets
-R 32 -L 64 --alpha 1.2
```

### 2. Tune for Your Use Case

**High Recall Applications** (search engines, recommendations):
```bash
-R 64 -L 128 --alpha 1.0
# Search with L=100-200
```

**Low Latency Applications** (real-time systems):
```bash  
-R 16 -L 32 --alpha 1.2
# Search with L=10-20
```

**Balanced Applications**:
```bash
-R 32 -L 64 --alpha 1.2  
# Search with L=20-50
```

### 3. Measure and Iterate

Always measure recall@K and QPS for your specific dataset:

```bash
./build/apps/search_memory_index \
  ... -K 10 -L 10 20 50 100 200 \
  --result_path results.csv
```

Plot the results to find your optimal operating point.

## Dataset-Specific Considerations

### High-Dimensional Data (>500D)
- Increase R to 64-128
- Use alpha=1.0 for better connectivity
- May need higher construction L

### Low-Dimensional Data (<50D)  
- R=16-32 often sufficient
- Can use higher alpha (1.2-1.5)
- Lower construction L works fine

### Large Datasets (>1M points)
- Consider higher R for better navigation
- Construction time becomes more important
- May want to reduce construction L for speed

### Clustered Data
- Lower alpha (1.0-1.1) helps bridge clusters
- Higher R provides more connectivity options

## Memory vs Speed Tradeoffs

**Index Size**: Proportional to N×R (where N = number of points)
**Construction Time**: Roughly O(N×L×log(N))  
**Search Time**: Depends on search L and graph quality

## Validation Strategy

1. **Split your data**: Use 90% for index, 10% for validation
2. **Compute ground truth**: Use brute force on validation set  
3. **Test configurations**: Try different R/L/alpha combinations
4. **Measure**: Record recall@K, QPS, index size, build time
5. **Choose**: Based on your application requirements

## Common Issues

**Low Recall**: Increase R, increase construction L, decrease alpha
**Slow Construction**: Decrease L, decrease R  
**Slow Search**: Use lower search L values
**Large Index**: Decrease R
**Disconnected Components**: Decrease alpha, increase R