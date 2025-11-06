#!/bin/bash
"""
Dependency Installation Script
Installs all required dependencies for DroneDelivery-RL project.
Supports Ubuntu/Debian, macOS, and handles both CPU and GPU setups.
"""

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

# Configuration
CONDA_ENV_NAME="drone_delivery_rl"
PYTHON_VERSION="3.9"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

print_header "DRONEDELIVERY-RL DEPENDENCY INSTALLATION"

print_info "Project root: $PROJECT_ROOT"
print_info "Conda environment: $CONDA_ENV_NAME"
print_info "Python version: $PYTHON_VERSION"

# Detect system
OS="$(uname -s)"
case "${OS}" in
    Linux*)     SYSTEM=Linux;;
    Darwin*)    SYSTEM=Mac;;
    CYGWIN*)    SYSTEM=Cygwin;;
    MINGW*)     SYSTEM=MinGw;;
    MSYS*)      SYSTEM=MinGw;;
    *)          SYSTEM="UNKNOWN:${OS}"
esac

print_info "Detected system: $SYSTEM"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_header "CHECKING PREREQUISITES"

# Check conda/mamba
if command_exists mamba; then
    CONDA_CMD="mamba"
    print_success "Mamba found (faster than conda)"
elif command_exists conda; then
    CONDA_CMD="conda"
    print_success "Conda found"
else
    print_error "Neither conda nor mamba found!"
    print_error "Please install Miniconda/Anaconda first:"
    print_error "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check Python version
if command_exists python3; then
    PYTHON_VERSION_INSTALLED=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python3 found: $PYTHON_VERSION_INSTALLED"
else
    print_warning "Python3 not found in PATH"
fi

# Install system dependencies
print_header "INSTALLING SYSTEM DEPENDENCIES"

case $SYSTEM in
    Linux)
        if command_exists apt; then
            print_info "Installing Ubuntu/Debian packages..."
            sudo apt update
            sudo apt install -y \
                build-essential \
                cmake \
                git \
                wget \
                curl \
                unzip \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                python3-dev \
                python3-pip
            print_success "System packages installed"
        elif command_exists yum; then
            print_info "Installing CentOS/RHEL packages..."
            sudo yum install -y \
                gcc \
                gcc-c++ \
                cmake \
                git \
                wget \
                curl \
                unzip \
                mesa-libGL \
                glib2 \
                python3-devel \
                python3-pip
            print_success "System packages installed"
        else
            print_warning "Unknown package manager - skipping system packages"
        fi
        ;;
    Mac)
        print_info "macOS detected"
        if command_exists brew; then
            print_info "Installing packages via Homebrew..."
            brew install cmake git wget curl
            print_success "Homebrew packages installed"
        else
            print_warning "Homebrew not found - install manually if needed"
            print_info "Install Homebrew: https://brew.sh/"
        fi
        ;;
    *)
        print_warning "Skipping system packages for $SYSTEM"
        ;;
esac

# Setup conda environment
print_header "SETTING UP CONDA ENVIRONMENT"

# Check if environment exists
if $CONDA_CMD env list | grep -q "$CONDA_ENV_NAME"; then
    print_info "Environment '$CONDA_ENV_NAME' already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing environment..."
        $CONDA_CMD env remove -n "$CONDA_ENV_NAME" -y
    else
        print_info "Using existing environment"
    fi
fi

# Create environment if it doesn't exist
if ! $CONDA_CMD env list | grep -q "$CONDA_ENV_NAME"; then
    print_info "Creating conda environment: $CONDA_ENV_NAME"
    $CONDA_CMD create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    print_success "Conda environment created"
fi

# Install Python packages
print_header "INSTALLING PYTHON PACKAGES"

print_info "Activating environment and installing packages..."

# Install PyTorch (CPU version by default)
print_info "Installing PyTorch (CPU)..."
$CONDA_CMD run -n "$CONDA_ENV_NAME" pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install core ML packages
print_info "Installing core ML packages..."
$CONDA_CMD run -n "$CONDA_ENV_NAME" pip install \
    gymnasium[classic_control] \
    pybullet \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    tensorboard \
    wandb \
    h5py \
    pandas \
    scikit-learn

# Install computer vision packages
print_info "Installing computer vision packages..."
$CONDA_CMD run -n "$CONDA_ENV_NAME" pip install \
    opencv-python \
    pillow \
    plotly

# Install utility packages
print_info "Installing utility packages..."
$CONDA_CMD run -n "$CONDA_ENV_NAME" pip install \
    pyyaml \
    tqdm \
    psutil \
    pytest \
    black \
    flake8

print_success "All Python packages installed"

# Install project in development mode
print_header "INSTALLING PROJECT"

cd "$PROJECT_ROOT"
print_info "Installing DroneDelivery-RL in development mode..."
$CONDA_CMD run -n "$CONDA_ENV_NAME" pip install -e .

print_success "Project installed in development mode"

# Verify installation
print_header "VERIFYING INSTALLATION"

print_info "Testing package imports..."
$CONDA_CMD run -n "$CONDA_ENV_NAME" python -c "
import torch
import numpy as np
import gymnasium as gym
import pybullet
import matplotlib.pyplot as plt
print('✓ Core packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NumPy version: {np.__version__}')
"

# Test project imports
print_info "Testing project imports..."
$CONDA_CMD run -n "$CONDA_ENV_NAME" python -c "
try:
    from src.utils import load_config
    from src.environment import DroneEnvironment
    from src.rl.agents import PPOAgent
    print('✓ Project imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    exit(1)
"

print_success "Installation verification complete"

# Final instructions
print_header "INSTALLATION COMPLETE"

echo -e "${GREEN}🎉 DroneDelivery-RL environment setup successful!${NC}\n"

echo "NEXT STEPS:"
echo "1. Activate the environment:"
echo "   conda activate $CONDA_ENV_NAME"
echo ""
echo "2. Verify installation:"
echo "   python scripts/setup/verify_installation.py"
echo ""
echo "3. Start training:"
echo "   python scripts/training/train_ppo.py"
echo ""
echo "4. Run evaluation:"
echo "   python scripts/evaluation/evaluate_model.py --model models/checkpoints/ppo_final.pt"
echo ""
echo "📖 See scripts/evaluation/HUONG_DAN_SU_DUNG.md for detailed usage instructions"

print_success "Ready to start development!"
