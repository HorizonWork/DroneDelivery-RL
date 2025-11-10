#!/usr/bin/env python3
"""
Build Environment Script
Creates and configures the complete development environment.
Sets up conda environment, installs dependencies, and initializes project structure.
"""

import os
import sys
import subprocess
import platform
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import urllib.request
import zipfile

class EnvironmentBuilder:
    """
    Complete environment builder for DroneDelivery-RL project.
    Handles conda environment creation, dependency installation, and setup verification.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_basic_logging()
        
        # System information
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Project paths
        self.project_root = Path(__file__).parent.parent.parent
        self.conda_env_name = "drone_delivery_rl"
        
        # Requirements
        self.python_requirements = [
            "torch>=2.0.0",
            "torchvision>=0.15.0", 
            "gymnasium>=0.29.0",
            "pybullet>=3.2.5",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "opencv-python>=4.8.0",
            "pillow>=10.0.0",
            "pyyaml>=6.0",
            "tensorboard>=2.13.0",
            "wandb>=0.15.0",
            "h5py>=3.9.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "plotly>=5.15.0",
            "tqdm>=4.65.0",
            "psutil>=5.9.0"
        ]
        
        # System packages (Ubuntu/Debian)
        self.system_packages = [
            "build-essential",
            "cmake",
            "git",
            "wget",
            "curl",
            "unzip",
            "libgl1-mesa-glx",
            "libglib2.0-0", 
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "libgomp1"
        ]
        
        self.logger.info("Environment Builder initialized")
        self.logger.info(f"System: {self.system}, Python: {self.python_version}")
    
    def setup_basic_logging(self):
        """Setup basic logging for environment builder."""
        # Set UTF-8 encoding for logging on Windows
        import locale
        if platform.system() == 'Windows':
            # Use ASCII-compatible characters instead of Unicode symbols
            pass  # Just ensuring UTF-8 support where possible
        
        # Create a more compatible logging setup that works across Python versions
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Create console handler without encoding parameter for compatibility
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Create file handler with encoding (this is supported in all relevant Python versions)
        file_handler = logging.FileHandler('setup.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    def build_complete_environment(self) -> bool:
        """
        Build complete development environment.
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("=== BUILDING DRONEDELIVERY-RL ENVIRONMENT ===")
        
        try:
            # Step 1: Check system prerequisites
            self.logger.info("Step 1: Checking system prerequisites...")
            if not self._check_system_prerequisites():
                return False
            
            # Step 2: Install system packages
            self.logger.info("Step 2: Installing system packages...")
            if not self._install_system_packages():
                self.logger.warning("System package installation failed (may require manual installation)")
            
            # Step 3: Setup conda environment
            self.logger.info("Step 3: Setting up conda environment...")
            if not self._setup_conda_environment():
                return False
            
            # Step 4: Install Python packages
            self.logger.info("Step 4: Installing Python dependencies...")
            if not self._install_python_packages():
                return False
            
            # Step 5: Create project structure
            self.logger.info("Step 5: Creating project structure...")
            if not self._create_project_structure():
                return False
            
            # Step 6: Download required data
            self.logger.info("Step 6: Downloading required data...")
            if not self._download_required_data():
                self.logger.warning("Data download failed (can be done manually)")
            
            # Step 7: Verify installation
            self.logger.info("Step 7: Verifying installation...")
            if not self._verify_installation():
                return False
            
            self.logger.info("=== ENVIRONMENT SETUP COMPLETED SUCCESSFULLY ===")
            self._print_next_steps()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Environment setup failed: {e}")
            return False
    
    def _check_system_prerequisites(self) -> bool:
        """Check system prerequisites."""
        prerequisites_ok = True
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.logger.error(f"Python 3.8+ required, found {sys.version_info}")
            prerequisites_ok = False
        else:
            self.logger.info(f"[OK] Python {self.python_version}")
        
        # Check conda/mamba
        conda_available = shutil.which('conda') is not None
        mamba_available = shutil.which('mamba') is not None
        
        if not (conda_available or mamba_available):
            self.logger.error("Conda or Mamba required but not found")
            self.logger.error("Please install Miniconda/Anaconda or Mamba")
            prerequisites_ok = False
        else:
            package_manager = 'mamba' if mamba_available else 'conda'
            self.logger.info(f"[OK] {package_manager.title()} package manager")
        
        # Check git
        if not shutil.which('git'):
            self.logger.warning("Git not found (recommended for development)")
        else:
            self.logger.info("[OK] Git")
        
        return prerequisites_ok
    
    def _install_system_packages(self) -> bool:
        """Install system-level packages."""
        if self.system == 'linux':
            try:
                # Detect package manager
                if shutil.which('apt'):
                    cmd = ['sudo', 'apt', 'update']
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    cmd = ['sudo', 'apt', 'install', '-y'] + self.system_packages
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                elif shutil.which('yum'):
                    cmd = ['sudo', 'yum', 'install', '-y'] + self.system_packages
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                self.logger.info("✓ System packages installed")
                return True
                
            except subprocess.CalledProcessError as e:
                self.logger.warning(f"System package installation failed: {e}")
                return False
        
        elif self.system == 'darwin':  # macOS
            self.logger.info("macOS detected - skipping system packages (install via Homebrew if needed)")
            return True
        
        else:  # Windows
            self.logger.info("Windows detected - skipping system packages")
            return True
    
    def _setup_conda_environment(self) -> bool:
        """Setup conda environment."""
        try:
            # Check if conda is available
            conda_cmd = shutil.which('conda')
            if not conda_cmd:
                self.logger.error("Conda command not found. Please ensure conda is installed and in PATH.")
                return False
            
            # Check if environment already exists
            cmd = [conda_cmd, 'env', 'list']
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            
            if self.conda_env_name in result.stdout:
                self.logger.info(f"Conda environment '{self.conda_env_name}' already exists")
                return True
            
            # Create new environment
            self.logger.info(f"Creating conda environment: {self.conda_env_name}")
            cmd = [
                conda_cmd, 'create', '-n', self.conda_env_name,
                f'python={self.python_version}', '-y'
            ]
            subprocess.run(cmd, check=True, capture_output=True, shell=True)
            
            self.logger.info(f"[OK] Conda environment '{self.conda_env_name}' created")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Conda environment setup failed: {e}")
            return False
        except FileNotFoundError:
            self.logger.error("Conda executable not found. Please ensure conda is properly installed and in system PATH.")
            return False
    
    def _install_python_packages(self) -> bool:
        """Install Python packages in conda environment."""
        try:
            conda_cmd = shutil.which('conda')
            if not conda_cmd:
                self.logger.error("Conda command not found. Please ensure conda is installed and in PATH.")
                return False
            
            # Install PyTorch first (special handling for CUDA)
            self.logger.info("Installing PyTorch...")
            # Use the exact package specifications from python_requirements
            # For PyTorch, we need to handle it specially due to the --index-url requirement
            # First, extract the base package name for PyTorch installation
            torch_spec = self.python_requirements[0]  # torch>=2.0.0
            torchvision_spec = self.python_requirements[1]  # torchvision>=0.15.0
            
            # For PyTorch, we'll install with the version requirement but use the CPU version
            pytorch_cmd = [
                conda_cmd, 'run', '-n', self.conda_env_name,
                'pip', 'install', torch_spec, torchvision_spec, 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cpu'  # CPU version
            ]
            subprocess.run(pytorch_cmd, check=True, capture_output=True, shell=True)
            self.logger.info(f"  [OK] {self.python_requirements[0]}")
            self.logger.info(f"  [OK] {self.python_requirements[1]}")
            
            # Install remaining packages (including gymnasium which was skipped in torch installation)
            self.logger.info("Installing remaining Python packages...")
            # Install gymnasium separately as it's not part of the torch installation
            gym_cmd = [
                conda_cmd, 'run', '-n', self.conda_env_name,
                'pip', 'install', self.python_requirements[2]  # gymnasium
            ]
            subprocess.run(gym_cmd, check=True, capture_output=True, shell=True)
            self.logger.info(f"  [OK] {self.python_requirements[2]}")
            
            # Install the rest of the packages
            for package in self.python_requirements[3:]:  # Skip torch, torchvision, gymnasium
                cmd = [
                    conda_cmd, 'run', '-n', self.conda_env_name,
                    'pip', 'install', package
                ]
                subprocess.run(cmd, check=True, capture_output=True, shell=True)
                self.logger.info(f"  [OK] {package}")
            
            self.logger.info("[OK] All Python packages installed")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Python package installation failed: {e}")
            return False
        except FileNotFoundError:
            self.logger.error("Conda executable not found. Please ensure conda is properly installed and in system PATH.")
            return False
    
    def _create_project_structure(self) -> bool:
        """Create project directory structure."""
        try:
            directories = [
                'data/trajectories',
                'data/maps',
                'models/checkpoints',
                'models/pretrained',
                'results/evaluations',
                'results/training',
                'results/visualizations',
                'logs',
                'config',
                'temp'
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create example config files
            self._create_example_configs()
            
            self.logger.info("[OK] Project structure created")
            return True
            
        except Exception as e:
            self.logger.error(f"Project structure creation failed: {e}")
            return False
    
    def _create_example_configs(self):
        """Create example configuration files."""
        config_dir = self.project_root / 'config'
        
        # Main configuration
        main_config = {
            'environment': {
                'building': {
                    'floors': 5,
                    'x_max': 20.0,
                    'y_max': 40.0,
                    'floor_height': 3.0
                },
                'obstacles': {
                    'density': 0.15,
                    'dynamic_obstacles': True
                }
            },
            'rl': {
                'ppo': {
                    'learning_rate': 3e-4,
                    'rollout_length': 2048,
                    'batch_size': 256,
                    'epochs': 10
                },
                'training': {
                    'total_timesteps': 5_000_000,
                    'eval_frequency': 50_000
                }
            },
            'evaluation': {
                'num_episodes': 100,
                'timeout': 300.0
            },
            'logging': {
                'level': 'INFO',
                'console_logging': True,
                'file_logging': True
            }
        }
        
        with open(config_dir / 'main_config.yaml', 'w') as f:
            import yaml
            yaml.dump(main_config, f, default_flow_style=False, indent=2)
        
        # Evaluation configuration
        eval_config = {
            'evaluation': {
                'num_episodes': 100,
                'episode_timeout': 300.0,
                'deterministic_policy': True,
                'record_trajectories': True
            },
            'visualization': {
                'save_plots': True,
                'figure_size': [12, 8],
                'dpi': 100
            }
        }
        
        with open(config_dir / 'evaluation_config.yaml', 'w') as f:
            import yaml
            yaml.dump(eval_config, f, default_flow_style=False, indent=2)
    
    def _download_required_data(self) -> bool:
        """Download required data files."""
        try:
            data_dir = self.project_root / 'data'
            
            # Example: Download building map data (placeholder)
            self.logger.info("Setting up example data...")
            
            # Create example building map
            building_map = {
                'building_config': {
                    'floors': 5,
                    'dimensions': [20.0, 40.0, 15.0],
                    'floor_height': 3.0
                },
                'obstacles': [],
                'staircases': [
                    {'position': [3, 3], 'floors': [1, 2, 3, 4, 5]},
                    {'position': [17, 3], 'floors': [1, 2, 3, 4, 5]}
                ],
                'elevators': [
                    {'position': [3, 37], 'floors': [1, 2, 3, 4, 5]},
                    {'position': [17, 37], 'floors': [1, 2, 3, 4, 5]}
                ]
            }
            
            with open(data_dir / 'building_map.json', 'w') as f:
                json.dump(building_map, f, indent=2)
            
            self.logger.info("✓ Example data created")
            return True
            
        except Exception as e:
            self.logger.warning(f"Data download failed: {e}")
            return False
    
    def _verify_installation(self) -> bool:
        """Verify installation completeness."""
        try:
            conda_cmd = shutil.which('conda')
            if not conda_cmd:
                self.logger.error("Conda command not found. Please ensure conda is installed and in PATH.")
                return False
            
            # Test conda environment
            cmd = [conda_cmd, 'run', '-n', self.conda_env_name, 'python', '-c',
                   'import torch, numpy, gymnasium, matplotlib; print("All imports successful")']
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
            
            if "All imports successful" in result.stdout:
                self.logger.info("[OK] Package imports verified")
            
            # Test PyTorch
            cmd = [conda_cmd, 'run', '-n', self.conda_env_name, 'python', '-c',
                   'import torch; print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")']
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
            self.logger.info(f"[OK] {result.stdout.strip()}")
            
            # Check project structure
            required_dirs = ['src', 'scripts', 'config', 'data', 'models', 'results']
            for directory in required_dirs:
                if not (self.project_root / directory).exists():
                    self.logger.error(f"Missing directory: {directory}")
                    return False
            
            self.logger.info("[OK] Project structure verified")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Installation verification failed: {e}")
            return False
        except FileNotFoundError:
            self.logger.error("Conda executable not found. Please ensure conda is properly installed and in system PATH.")
            return False
    
    def _print_next_steps(self):
        """Print next steps for user."""
        print("\n" + "="*60)
        print("🎉 DRONEDELIVERY-RL ENVIRONMENT SETUP COMPLETE!")
        print("="*60)
        print("\nNEXT STEPS:")
        print("1. Activate the conda environment:")
        print(f"   conda activate {self.conda_env_name}")
        print("\n2. Train the PPO model:")
        print("   python scripts/training/train_ppo.py --config config/main_config.yaml")
        print("\n3. Evaluate the trained model:")
        print("   python scripts/evaluation/evaluate_model.py --model models/checkpoints/ppo_final.pt")
        print("\n4. Generate Table 3 results:")
        print("   bash scripts/evaluation/run_full_evaluation.sh")
        print("\n" + "="*60)
        print("📖 See scripts/evaluation/HUONG_DAN_SU_DUNG.md for detailed instructions")
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build DroneDelivery-RL environment')
    parser.add_argument('--env-name', type=str, default='drone_delivery_rl',
                       help='Conda environment name')
    parser.add_argument('--force', action='store_true',
                       help='Force reinstall even if environment exists')
    
    args = parser.parse_args()
    
    # Create builder
    builder = EnvironmentBuilder()
    builder.conda_env_name = args.env_name
    
    # Force reinstall if requested
    if args.force:
        try:
            subprocess.run(['conda', 'env', 'remove', '-n', args.env_name, '-y'],
                         capture_output=True)
            print(f"Removed existing environment: {args.env_name}")
        except:
            pass
    
    # Build environment
    success = builder.build_complete_environment()
    
    if success:
        print("🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        print("❌ Setup failed. Check setup.log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
