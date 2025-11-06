#!/bin/bash
# Setup conda environment

echo "Creating conda environment..."
conda env create -f environment.yml

echo "Environment created. Activate with: conda activate drone-delivery-rl"
