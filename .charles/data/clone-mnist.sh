#!/bin/bash

# Define the target directory
TARGET_DIR="../.data/ylecun"

# Ensure the parent directory exists
mkdir -p "$(dirname "$TARGET_DIR")"

# Clone the dataset into the target directory
git clone https://huggingface.co/datasets/ylecun/mnist "$TARGET_DIR"

# Print success message
if [ $? -eq 0 ]; then
  echo "Dataset successfully cloned into $TARGET_DIR"
else
  echo "Failed to clone the dataset."
fi