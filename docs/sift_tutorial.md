# Complete SIFT Tutorial

This tutorial walks you through using the Vamana implementation with the SIFT dataset from start to finish.

## Step 1: Download SIFT Dataset

```bash
# Download SIFT-10K dataset
wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
tar -xzf siftsmall.tar.gz

# You should now have:
# siftsmall/siftsmall_base.fvecs     (10,000 vectors, 128D)
# siftsmall/siftsmall_query.fvecs    (100 query vectors, 128D) 
# siftsmall/siftsmall_groundtruth.ivecs (100 ground truth results)
```

## Step 2: Organize Data

```bash
# Create organized directory structure
mkdir -p datasets/raw/siftsmall
mv siftsmall/* datasets/raw/siftsmall/
mkdir -p datasets/indexes
```

## Step 3: Convert to Binary Format

The Vamana implementation uses efficient binary formats:

```bash
# Convert base vectors (.fvecs → .fbin)
./build/apps/utils/fvecs_to_fbin \
  --input_file datasets/raw/siftsmall/siftsmall_base.fvecs \
  --output_file datasets/indexes/sift_base.fbin

# Convert query vectors  
./build/apps/utils/fvecs_to_fbin \
  --input_file datasets/raw/siftsmall/siftsmall_query.fvecs \
  --output_file datasets/indexes/sift_queries.fbin

# Convert ground truth (.ivecs → .ibin)
./build/apps/utils/ivecs_to_ibin \
  --input_file datasets/raw/siftsmall/siftsmall_groundtruth.ivecs \
  --output_file datasets/indexes/sift_groundtruth.ibin
```

## Step 4: Build Index

```bash
./build/apps/build_memory_index \
  --data_type float --dist_fn l2 \
  --data_path datasets/indexes/sift_base.fbin \
  --index_path_prefix datasets/indexes/sift_index \
  -R 32 -L 64 --alpha 1.2
```

This creates:
- `datasets/indexes/sift_index.graph` - Graph structure
- `datasets/indexes/sift_index.meta` - Index metadata

## Step 5: Search and Evaluate

Test different search configurations:

```bash
./build/apps/search_memory_index \
  --data_type float --dist_fn l2 \
  --data_path datasets/indexes/sift_base.fbin \
  --index_path_prefix datasets/indexes/sift_index \
  --query_file datasets/indexes/sift_queries.fbin \
  --gt_file datasets/indexes/sift_groundtruth.ibin \
  -K 10 -L 10 20 50 100 200 \
  --result_path results.csv
```

## Expected Results

You should see output like:
```
L       QPS     Recall@10
10      813     0.915
20      454     0.988  
50      192     1.000
100     99      1.000
200     52      1.000
```

This shows the speed/accuracy tradeoff - higher L gives better recall but lower QPS.

## Parameter Exploration

Try different construction parameters:

```bash
# High precision build (slower construction, better recall)
./build/apps/build_memory_index ... -R 64 -L 128 --alpha 1.0

# Fast build (faster construction, lower recall)
./build/apps/build_memory_index ... -R 16 -L 32 --alpha 1.5
```

## Custom Ground Truth

If you don't have ground truth, compute it:

```bash
./build/apps/utils/compute_groundtruth \
  --data_type float --dist_fn l2 \
  --base_file datasets/indexes/sift_base.fbin \
  --query_file datasets/indexes/sift_queries.fbin \
  --gt_file datasets/indexes/my_groundtruth.ibin \
  -K 100
```

This computes exact k-NN using brute force search.