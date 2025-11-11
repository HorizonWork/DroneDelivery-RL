# Installation Guide - DroneDelivery-RL

## System Requirements

### Hardware Requirements
- **CPU**: Intel i7 or AMD Ryzen 7 (minimum 8 cores recommended)
- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space minimum
- **OS**: Ubuntu 20.04/22.04 LTS or Windows 10/11

### Software Prerequisites
- **Python**: 3.8-3.10
- **CUDA**: 11.3 or later (for GPU acceleration)
- **Unreal Engine**: 4.26+ (for AirSim)
- **ROS2**: Humble Hawksbill (optional, for SLAM integration)

## Installation Steps

### Step 1: Environment Setup

#### Option A: Using Conda (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd DroneDelivery-RL

# Create conda environment
conda env create -f environment.yml
conda activate drone-delivery-rl

# Install package in development mode
pip install -e .
```

#### Option B: Using Virtual Environment
```bash
# Clone the repository
git clone <repository-url>
cd DroneDelivery-RL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

For more detailed installation steps, see the full documentation.
