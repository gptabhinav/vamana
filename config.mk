# use this for runtime configuration
# for static constants in the code, consider looking at types.h

#graph construction parameters
R = 32
L = 64
alpha = 1.2

# this configuration format seems to work well for sift and siftsmall
# might have to update this when i work with glove, gist and others
dataset = siftsmall

dataset_root =  datasets/${dataset}

dataset.base = ${dataset_root}/$(dataset)_base.fvecs
dataset.query = ${dataset_root}/$(dataset)_query.fvecs
dataset.groundtruth = ${dataset_root}/$(dataset)_groundtruth.ivecs

dataset_bin_root = ${dataset_root}/bin

dataset.base.bin = ${dataset_bin_root}/$(dataset)_base.fbin
dataset.query.bin = ${dataset_bin_root}/$(dataset)_query.fbin
dataset.groundtruth.bin = ${dataset_bin_root}/$(dataset)_groundtruth.ibin

index_root = index/

dataset.index = ${index_root}/$(dataset)

results_root = output/

dataset.results = ${results_root}/${dataset}.csv