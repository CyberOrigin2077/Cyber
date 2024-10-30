#!/bin/bash

# List of datasets to download
datasets=(
    "cyberorigin/cyber_pipette"
    "cyberorigin/cyber_twist_the_tube"
    "cyberorigin/cyber_fold_towels"
    "cyberorigin/cyber_take_the_item"
)

# Function to download a dataset
download_dataset() {
    local dataset=$1
    local dataset_name=$(basename "$dataset")
    local target_dir="data/$dataset_name"
    
    echo "Downloading $dataset..."
    mkdir -p "$target_dir"
    huggingface-cli download "$dataset" --repo-type dataset --local-dir "$target_dir"
    if [ $? -eq 0 ]; then
        echo "Successfully downloaded: $dataset to $target_dir"
    else
        echo "Error downloading $dataset"
    fi
}

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Please install it using: pip install huggingface_hub"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download each dataset
for dataset in "${datasets[@]}"; do
    download_dataset "$dataset"
done

echo "All downloads completed."