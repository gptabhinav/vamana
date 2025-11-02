#!/bin/bash

DATASET=$1
DATASETS_DIR="datasets"

#  -z checks if the string is null (empty)
# If no dataset name is provided, display usage message and exit
if [ -z "$DATASET" ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Available datasets: sift, siftsmall"
    exit 1
fi

# check if datasets directory exists, if not create it
if [ ! -d "$DATASETS_DIR" ]; then
    echo "Creating datasets directory at '$DATASETS_DIR'..."
    mkdir -p "$DATASETS_DIR"
fi

# check if dataset is already downloaded
if [ -d "$DATASETS_DIR/$DATASET" ]; then
    echo "Dataset '$DATASET' already exists in '$DATASETS_DIR'. Skipping download."
    echo "To re-download, please delete the existing dataset directory '${DATASETS_DIR}/$DATASET' and run the script again."
    exit 0
fi

# workflow based  on dataset name
case $DATASET in
    sift)
    echo "Downloading SIFT dataset..."
    mkdir -p $DATASETS_DIR/sift
    wget -O $DATASETS_DIR/sift/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
    tar -xzf $DATASETS_DIR/sift/sift.tar.gz -C $DATASETS_DIR/sift
    rm $DATASETS_DIR/sift/sift.tar.gz
    echo "'$DATASET' dataset downloaded and extracted to '$DATASETS_DIR/$DATASET'."
    ;;
    siftsmall)
    echo "Downloading SIFT10K dataset..."
    mkdir -p $DATASETS_DIR/siftsmall
    wget -O $DATASETS_DIR/siftsmall/siftsmall.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
    tar -xzf $DATASETS_DIR/siftsmall/siftsmall.tar.gz -C $DATASETS_DIR/siftsmall
    rm $DATASETS_DIR/siftsmall/siftsmall.tar.gz
    echo "'$DATASET' dataset downloaded and extracted to '$DATASETS_DIR/$DATASET'."
    ;;
esac