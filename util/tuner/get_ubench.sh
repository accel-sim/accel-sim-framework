#!/bin/bash

# Configuration
REPO_URL="https://github.com/accel-sim/gpu-app-collection.git"
CLONE_DIR="gpu-app-collection-partial"
BRANCH="dev" 
SPARSE_PATHS=(
  "src/cuda/GPU_Microbenchmark"
  "src/cuda/cuda-samples"
)

# Step 1: Clone repo with sparse checkout enabled
git clone --recurse-submodules -j8 --filter=blob:none --no-checkout -b "$BRANCH" "$REPO_URL" "$CLONE_DIR"
cd "$CLONE_DIR"

# Step 2: Enable sparse checkout
git sparse-checkout init --cone
git sparse-checkout set "${SPARSE_PATHS[@]}"
git checkout

# Step 3: Manually initialize the submodule (if not already checked out)
git submodule update --init --recursive -- src/cuda/cuda-samples
