#!/usr/bin/env python3
"""
Installation Verification Script
Comprehensive verification of DroneDelivery-RL installation.
Tests all components and dependencies.
"""

import os
import sys
import logging
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

class InstallationVerifier:
    """
    Comprehensive installation verification system.
    Tests all components, dependencies, and configurations.
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Project root
        self.project_root = Path(__file__).parent.parent.parent
        
        # Test results
        self.test_results = {}
        self.failed_tests = []
        
        self.logger.info("Installation Verifier initialized")
    
    def setup_logging(self):
        """Setup basic logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    
    def verify_complete_installation(self) -> bool:
        """
        Verify complete installation.
        
        Returns:
            True if all tests pass
        """
        print("🔍 VERIFYING DRONEDELIVERY-RL INSTALLATION")
        print("=" * 50)
        
        # Run all verification tests
        tests = [
            ("Python Environment", self._test_python_environment),
            ("Core Dependencies", self._test_core_dependencies),
            ("ML/RL Packages", self._test_ml_packages),
            ("Computer Vision", self._test_cv_packages),
            ("Project Structure", self._test_project_structure),
            ("Project Imports", self._test_project_imports),
            ("Configuration Files", self._test_configuration_files),
            ("System Resources", self._test_system_resources)
        ]
        
        all_passed = True
        
        for test_name, test_function in tests:
            print(f"\n🧪 Testing {test_name}...")
            
            try:
                success = test_function()
                if success:
                    print(f"✅ {test_name}: PASSED")
                else:
                    print(f"❌ {test_name}: FAILED")
                    all_passed = False
                    self.failed_tests.append(test_name)
                    
                self.test_results[test_name] = success
                
            except Exception as e:
                print(f"❌ {test_name}: ERROR - {e}")
                all_passed = False
                self.failed_tests.append(test_name)
                self.test_results[test_name] = False
        
        # Print final results
        self._print_verification_summary(all_passed)
        
        return all_passed
    
    def _test_python_environment(self) -> bool:
        """Test Python environment."""
        # Python version
        if sys.version_info < (3, 8):
            print(f"  ❌ Python 3.8+ required, found {sys.version_info}")
            return False
        
        print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print(f"  ✅ Virtual environment active: {os.environ.get('CONDA_DEFAULT_ENV', 'venv')}")
        else:
            print("  ⚠️  No virtual environment detected (recommended)")
        
        return True
    
    def _test_core_dependencies(self) -> bool:
        """Test core dependencies."""
        core_packages = [
            ('numpy', '1.20.0'),
            ('scipy', '1.7.0'),
            ('matplotlib', '3.5.0'),
            ('pyyaml', None),
            ('tqdm', None),
            ('psutil', None)
        ]
        
        all_passed = True
        
        for package_name, min_version in core_packages:
            try:
                package = importlib.import_module(package_name)
                version = getattr(package, '__version__', 'unknown')
                print(f"  ✅ {package_name}: {version}")
            except ImportError:
                print(f"  ❌ {package_name}: NOT FOUND")
                all_passed = False
        
        return all_passed
    
    def _test_ml_packages(self) -> bool:
        """Test ML/RL packages."""
        ml_packages = [
            'torch',
            'gymnasium',
            'pybullet',
            'tensorboard',
            'wandb'
        ]
        
        all_passed = True
        
        for package_name in ml_packages:
            try:
                package = importlib.import_module(package_name)
                version = getattr(package, '__version__', 'unknown')
                print(f"  ✅ {package_name}: {version}")
                
                # Special PyTorch tests
                if package_name == 'torch':
                    cuda_available = package.cuda.is_available()
                    print(f"    CUDA available: {cuda_available}")
                    
                # Special Gymnasium test
                if package_name == 'gymnasium':
                    import gymnasium as gym
                    try:
                        env = gym.make('CartPole-v1')
                        env.close()
                        print(f"    ✅ Environment creation test passed")
                    except Exception as e:
                        print(f"    ⚠️  Environment test failed: {e}")
                
            except ImportError as e:
                print(f"  ❌ {package_name}: NOT FOUND - {e}")
                all_passed = False
        
        return all_passed
    
    def _test_cv_packages(self) -> bool:
        """Test computer vision packages."""
        cv_packages = ['cv2', 'PIL']
        package_names = ['opencv-python', 'pillow']
        
        all_passed = True
        
        for package_name, display_name in zip(cv_packages, package_names):
            try:
                package = importlib.import_module(package_name)
                version = getattr(package, '__version__', 'unknown')
                print(f"  ✅ {display_name}: {version}")
            except ImportError:
                print(f"  ❌ {display_name}: NOT FOUND")
                all_passed = False
        
        return all_passed
    
    def _test_project_structure(self) -> bool:
        """Test project directory structure."""
        required_dirs = [
            'src',
            'src/rl',
            'src/environment', 
            'src/localization',
            'src/planning',
            'src/utils',
            'scripts',
            'scripts/evaluation',
            'scripts/setup',
            'config',
            'data',
            'models',
            'results'
        ]
        
        all_passed = True
        
        for directory in required_dirs:
            dir_path = self.project_root / directory
            if dir_path.exists():
                print(f"  ✅ {directory}/")
            else:
                print(f"  ❌ {directory}/ - MISSING")
                all_passed = False
        
        return all_passed
    
    def _test_project_imports(self) -> bool:
        """Test project package imports."""
        # Add project to path
        sys.path.insert(0, str(self.project_root))
        
        project_imports = [
            'src.utils',
            'src.environment',
            'src.rl.agents',
            'src.planning',
            'src.localization'
        ]
        
        all_passed = True
        
        for import_path in project_imports:
            try:
                module = importlib.import_module(import_path)
                print(f"  ✅ {import_path}")
            except ImportError as e:
                print(f"  ❌ {import_path}: {e}")
                all_passed = False
        
        return all_passed
    
    def _test_configuration_files(self) -> bool:
        """Test configuration files."""
        config_files = [
            'config/main_config.yaml',
            'config/evaluation_config.yaml'
        ]
        
        all_passed = True
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                    print(f"  ✅ {config_file} - Valid YAML")
                except Exception as e:
                    print(f"  ❌ {config_file} - Invalid: {e}")
                    all_passed = False
            else:
                print(f"  ⚠️  {config_file} - Not found (will use defaults)")
        
        return all_passed
    
    def _test_system_resources(self) -> bool:
        """Test system resources."""
        import psutil
        
        # Memory check
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb >= 8:
            print(f"  ✅ RAM: {memory_gb:.1f} GB")
        else:
            print(f"  ⚠️  RAM: {memory_gb:.1f} GB (8GB+ recommended)")
        
        # Disk space check
        disk_usage = psutil.disk_usage(str(self.project_root))
        free_gb = disk_usage.free / (1024**3)
        if free_gb >= 10:
            print(f"  ✅ Disk space: {free_gb:.1f} GB free")
        else:
            print(f"  ⚠️  Disk space: {free_gb:.1f} GB free (10GB+ recommended)")
        
        # CPU check
        cpu_count = psutil.cpu_count()
        print(f"  ✅ CPU cores: {cpu_count}")
        
        return True
    
    def _print_verification_summary(self, all_passed: bool):
        """Print verification summary."""
        print("\n" + "="*50)
        
        if all_passed:
            print("🎉 INSTALLATION VERIFICATION: ALL TESTS PASSED")
            print("\nYour DroneDelivery-RL installation is ready!")
            print("\nNext steps:")
            print("1. conda activate drone_delivery_rl")
            print("2. python scripts/training/train_ppo.py")
            print("3. python scripts/evaluation/evaluate_model.py --model models/checkpoints/ppo_final.pt")
        else:
            print("❌ INSTALLATION VERIFICATION: SOME TESTS FAILED")
            print(f"\nFailed tests: {', '.join(self.failed_tests)}")
            print("\nPlease resolve the issues and run verification again.")
        
        print("="*50)
        
        # Save test results
        results_path = self.project_root / 'verification_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"\n📄 Detailed results saved to: {results_path}")

def main():
    verifier = InstallationVerifier()
    success = verifier.verify_complete_installation()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
