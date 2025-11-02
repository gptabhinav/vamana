include config.mk

download-dataset:
	scripts/download_datasets.sh ${dataset}

dataset-bins:
	mkdir -p ${dataset_bin_root}
	build/apps/utils/fvecs_to_fbin --input_file ${dataset.base} --output_file ${dataset.base.bin}
	build/apps/utils/fvecs_to_fbin --input_file ${dataset.query} --output_file ${dataset.query.bin}
	build/apps/utils/ivecs_to_ibin --input_file ${dataset.groundtruth} --output_file ${dataset.groundtruth.bin}

build:
	mkdir build && cd build && cmake .. && make

clean-dataset:
	rm -rf datasets/${dataset}

clean-build:
	rm -rf build/

clean:
	rm -rf datasets/${dataset} build/

help:
	@echo "Makefile targets:"
	@echo "	download-dataset		Download the specified dataset (set dataset variable)"
	@echo "	dataset-bins			Convert dataset files to binary format"
	@echo "	build				Build the project"
	@echo "	clean-dataset			Remove the downloaded dataset"
	@echo "	clean-build			Remove the build directory"
	@echo "	clean				Remove both dataset and build directories"
	@echo "	help				Show this help message"