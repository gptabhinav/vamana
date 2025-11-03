# use this for runtime configuration
# for static constants in the code, consider looking at types.h

# this seems to work well for sift and siftsmall
dataset = siftsmall

dataset_root =  datasets/${dataset}

dataset.base = ${dataset_root}/$(dataset)_base.fvecs
dataset.query = ${dataset_root}/$(dataset)_query.fvecs
dataset.groundtruth = ${dataset_root}/$(dataset)_groundtruth.ivecs

dataset_bin_root = ${dataset_root}/bin

dataset.base.bin = ${dataset_bin_root}/$(dataset)_base.fbin
dataset.query.bin = ${dataset_bin_root}/$(dataset)_query.fbin
dataset.groundtruth.bin = ${dataset_bin_root}/$(dataset)_groundtruth.ibin