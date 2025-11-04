include config.mk

.PHONY: download-dataset dataset-bins build-index search-index clean-dataset clean-build clean-index clean help

download-dataset:
# the script itself ensures that the datasets directory is created, but we can also be explicit about it here to be agnostic
	$(info === Downloading dataset: ${dataset} ===)
	scripts/download_datasets.sh ${dataset}

# Build C++ executables - depend on CMake and source files
# The stamp file approach avoids unnecessary rebuilds, we just check its timestamp to see when build files were last modified
build/.build-stamp: CMakeLists.txt $(wildcard src/**/*.cpp) $(wildcard include/**/*.h) $(wildcard apps/*.cpp)
	$(info === Building project ===)
	mkdir -p build && cd build && cmake .. && make
	touch build/.build-stamp

# Convenience alias for building
build: build/.build-stamp

# Ensure utility and application binaries are built before dataset bin conversion, index building, or searching
build/apps/utils/fvecs_to_fbin build/apps/utils/ivecs_to_ibin build/apps/build_memory_index build/apps/search_memory_index : build/.build-stamp

# Individual target rules for building each binary file -- base, query and groundtruth
# will only be run, our base file is missing. recompilation of build/apps/utils/fvecs_to_fbin wont lead to new binary generation
# we just ensure build/apps/utils/fvecs_to_fbin is there, timestamp isnt checked for it
${dataset.base.bin}: ${dataset.base} | build/apps/utils/fvecs_to_fbin
	mkdir -p ${dataset_bin_root}
	build/apps/utils/fvecs_to_fbin --input_file ${dataset.base} --output_file ${dataset.base.bin}

${dataset.query.bin}: ${dataset.query} | build/apps/utils/fvecs_to_fbin
	mkdir -p ${dataset_bin_root}
	build/apps/utils/fvecs_to_fbin --input_file ${dataset.query} --output_file ${dataset.query.bin}

${dataset.groundtruth.bin}: ${dataset.groundtruth} | build/apps/utils/ivecs_to_ibin
	mkdir -p ${dataset_bin_root}
	build/apps/utils/ivecs_to_ibin --input_file ${dataset.groundtruth} --output_file ${dataset.groundtruth.bin}

# This alias is just for convenience
dataset-bins: ${dataset.base.bin} ${dataset.query.bin} ${dataset.groundtruth.bin}

dataset: download-dataset dataset-bins

# Index files depend on the binary dataset and build executables
# Using &: (grouped target) to ensure the command runs only once for both output files
${dataset.index}.graph ${dataset.index}.meta &: ${dataset.base.bin} | build/apps/build_memory_index
	$(info === Building index files ===)
	mkdir -p ${index_root}
	build/apps/build_memory_index --data_type float --dist_fn l2 --data_path ${dataset.base.bin} --index_path_prefix ${dataset.index} -R ${R} -L ${L} --alpha ${alpha}

# Alias for convenience
build-index: ${dataset.index}.graph ${dataset.index}.meta

# Results depend on index files and query files
${dataset.results}: ${dataset.index}.graph ${dataset.index}.meta ${dataset.query.bin} ${dataset.groundtruth.bin} | build/apps/search_memory_index
	mkdir -p ${results_root}
	build/apps/search_memory_index --data_type float --dist_fn l2 --data_path ${dataset.base.bin} --index_path_prefix ${dataset.index} --query_file ${dataset.query.bin} --gt_file ${dataset.groundtruth.bin} -K 10 -L 10 20 50 100 --result_path ${dataset.results} -T 0

# Alias for convenience
search-index: ${dataset.results}

clean-dataset:
	$(info === Cleaning dataset directory ===)
	rm -rf datasets/${dataset}

clean-build:
	$(info === Cleaning build directory ===)
	rm -rf build/

clean-index:
	$(info === Cleaning index directory ===)
	rm -rf ${index_root}

clean-results:
	$(info === Cleaning results directory ===)
	rm -rf ${results_root}

clean:
	$(info === Cleaning dataset, build, index, and results directories ===)
	rm -rf ${dataset_root} build/ ${index_root} ${results_root}

help:
	@echo "Makefile targets:"
	@echo "	download-dataset		Download the specified dataset (set dataset variable)"
	@echo "	dataset-bins			Convert dataset files to binary format"
	@echo "	build-index			Build the index for the dataset"
	@echo "	search-index			Search the dataset index and produce results"
	@echo "	build				Build the project"
	@echo "	clean-dataset			Remove the downloaded dataset"
	@echo "	clean-build			Remove the build directory"
	@echo "	clean-index			Remove the index directory"
	@echo "	clean				Remove dataset, build, index, results directories"
	@echo "	help				Show this help message"


## add commands for building and searching index

## work on improving performance of vamana on sift (a.k.a sift1m)