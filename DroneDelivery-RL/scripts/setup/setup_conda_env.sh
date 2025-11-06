#!/bin/bash
"""
Conda Environment Setup Script
Quick conda environment setup specifically for DroneDelivery-RL.
Optimized for research and development workflow.
"""

set -e

# Configuration
ENV_NAME="drone_delivery_rl"
PYTHON_VERSION="3.9"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== DRONEDELIVERY-RL CONDA SETUP ===${NC}\n"

# Check if conda/mamba exists
if command -v mamba >/dev/null 2>&1; then
    CONDA_CMD="mamba"
    echo -e "${GREEN}✓ Using Mamba (faster)${NC}"
elif command -v conda >/dev/null 2>&1; then
    CONDA_CMD="conda"
    echo -e "${GREEN}✓ Using Conda${NC}"
else
    echo -e "${RED}❌ Neither conda nor mamba found!${NC}"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Remove existing environment if it exists
if $CONDA_CMD env list | grep -q "$ENV_NAME"; then
    echo -e "${YELLOW}Environment '$ENV_NAME' already exists${NC}"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing existing environment...${NC}"
        $CONDA_CMD env remove -n "$ENV_NAME" -y
        echo -e "${GREEN}✓ Environment removed${NC}"
    else
        echo -e "${YELLOW}Using existing environment${NC}"
        exit 0
    fi
fi

# Create new environment
echo -e "${BLUE}Creating new conda environment: $ENV_NAME${NC}"
$CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo -e "${GREEN}✓ Conda environment created${NC}"

# Install core packages
echo -e "\n${BLUE}Installing core packages...${NC}"

# PyTorch (CPU version for broader compatibility)
echo "Installing PyTorch..."
$CONDA_CMD run -n "$ENV_NAME" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# RL and Environment packages
echo "Installing RL packages..."
$CONDA_CMD run -n "$ENV_NAME" pip install \
    gymnasium[classic_control] \
    pybullet \
    stable-baselines3[extra]

# Scientific computing
echo "Installing scientific packages..."
$CONDA_CMD run -n "$ENV_NAME" pip install \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    pandas \
    scikit-learn

# Computer Vision
echo "Installing CV packages..."
$CONDA_CMD run -n "$ENV_NAME" pip install \
    opencv-python \
    pillow

# Logging and visualization
echo "Installing logging packages..."
$CONDA_CMD run -n "$ENV_NAME" pip install \
    tensorboard \
    wandb \
    plotly \
    h5py

# Utilities
echo "Installing utilities..."
$CONDA_CMD run -n "$ENV_NAME" pip install \
    pyyaml \
    tqdm \
    psutil \
    pytest \
    black \
    flake8

echo -e "${GREEN}✓ All packages installed${NC}"

# Verify installation
echo -e "\n${BLUE}Verifying installation...${NC}"

$CONDA_CMD run -n "$ENV_NAME" python -c "
import sys
print(f'Python version: {sys.version}')

import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

import numpy as np
print(f'NumPy version: {np.__version__}')

import gymnasium as gym
print(f'Gymnasium version: {gym.__version__}')

try:
    import pybullet
    print('✓ PyBullet available')
except ImportError:
    print('✗ PyBullet not available')

print('\n🎉 Installation verification successful!')
"

# Final instructions
echo -e "\n${GREEN}=== SETUP COMPLETED SUCCESSFULLY ===${NC}\n"

echo "To use the environment:"
echo -e "${YELLOW}conda activate $ENV_NAME${NC}"
echo ""
echo "To verify project setup:"
echo -e "${YELLOW}python scripts/setup/verify_installation.py${NC}"
echo ""
echo "To start training:"
echo -e "${YELLOW}python scripts/training/train_ppo.py${NC}"
echo ""
echo -e "${GREEN}Happy coding! 🚁✨${NC}"
