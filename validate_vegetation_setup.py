#!/usr/bin/env python3
"""
VEGETATION DETECTION - SETUP VALIDATION SCRIPT
Verify all components are correctly installed and configured
"""

import sys
import os
import subprocess
from pathlib import Path
import importlib
from packaging import version


class VegetationSetupValidator:
    def __init__(self):
        self.results = {'passed': [], 'failed': [], 'warnings': []}
        self.root_dir = Path(__file__).parent
    
    def print_header(self):
        print("\n" + "="*70)
        print("🌳 VEGETATION DETECTION - SETUP VALIDATION")
        print("="*70 + "\n")
    
    def check_python_version(self):
        """Verify Python version >= 3.8"""
        print("📌 Checking Python version...", end=" ")
        if sys.version_info >= (3, 8):
            version_str = f"{sys.version_info.major}.{sys.version_info.minor}"
            print(f"✅ {version_str}")
            self.results['passed'].append(f"Python {version_str}")
        else:
            print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} < 3.8")
            self.results['failed'].append(f"Python version < 3.8")
    
    def check_package(self, package_name, import_name=None, min_version=None):
        """Check if a Python package is installed"""
        import_name = import_name or package_name
        print(f"📌 Checking {package_name}...", end=" ")
        
        try:
            module = importlib.import_module(import_name)
            
            # Check version if specified
            if min_version and hasattr(module, '__version__'):
                current_version = version.parse(module.__version__)
                min_ver = version.parse(min_version)
                
                if current_version >= min_ver:
                    print(f"✅ {module.__version__}")
                    self.results['passed'].append(f"{package_name} {module.__version__}")
                else:
                    print(f"⚠️  {module.__version__} < {min_version}")
                    self.results['warnings'].append(
                        f"{package_name} version {module.__version__} < recommended {min_version}"
                    )
            else:
                print("✅")
                self.results['passed'].append(package_name)
        
        except ImportError:
            print("❌")
            self.results['failed'].append(f"{package_name} not installed")
    
    def check_gpu(self):
        """Check NVIDIA GPU availability"""
        print("📌 Checking GPU (CUDA)...", end=" ")
        
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                print(f"✅ {device_count} GPU(s) - {device_name}")
                self.results['passed'].append(f"CUDA available ({device_count} GPU)")
            else:
                print("⚠️  CUDA not available (CPU mode)")
                self.results['warnings'].append("CUDA not available - using CPU (slower)")
        
        except Exception as e:
            print(f"⚠️  Could not detect GPU: {e}")
            self.results['warnings'].append("GPU detection failed")
    
    def check_files(self):
        """Check required files exist"""
        print("\n📌 Checking required files...")
        
        required_files = [
            "vegetation_detection_training.py",
            "vegetation_inference.py",
            "create_vegetation_tiles.py",
            "vegetation_pipeline.py",
            "vegetation_ensemble.py",
            "pytorch_model_training/vegetation_detection_training.py",
            "config/config_vegetation.yaml",
            "VEGETATION_DETECTION_GUIDE.md",
            "QUICK_REFERENCE.md",
        ]
        
        for file_path in required_files:
            full_path = self.root_dir / file_path
            if full_path.exists():
                print(f"   ✅ {file_path}")
                self.results['passed'].append(f"File: {file_path}")
            else:
                print(f"   ❌ {file_path} not found")
                self.results['failed'].append(f"Missing file: {file_path}")
    
    def check_directories(self):
        """Check required directories"""
        print("\n📌 Checking required directories...")
        
        required_dirs = [
            "pytorch_model_training",
            "config",
        ]
        
        for dir_path in required_dirs:
            full_path = self.root_dir / dir_path
            if full_path.exists() and full_path.is_dir():
                print(f"   ✅ {dir_path}/")
                self.results['passed'].append(f"Directory: {dir_path}")
            else:
                print(f"   ❌ {dir_path}/ not found")
                self.results['failed'].append(f"Missing directory: {dir_path}")
    
    def check_dependencies(self):
        """Check all required Python packages"""
        print("\n📌 Checking Python dependencies...")
        
        # Core dependencies
        packages = [
            ("torch", "torch", "2.0.0"),
            ("torch vision", "torchvision", "0.15.0"),
            ("numpy", "numpy", "1.21.0"),
            ("rasterio", "rasterio", "1.3.0"),
            ("geopandas", "geopandas", "0.12.0"),
            ("shapely", "shapely", "2.0.0"),
            ("scipy", "scipy", "1.10.0"),
            ("scikit-image", "skimage", "0.21.0"),
            ("opencv-python", "cv2", None),
            ("albumentations", "albumentations", "1.3.0"),
            ("segmentation-models-pytorch", "segmentation_models_pytorch", "0.3.3"),
            ("pandas", "pandas", "1.3.0"),
            ("tqdm", "tqdm", "4.60.0"),
            ("PyYAML", "yaml", "6.0"),
        ]
        
        for package_name, import_name, min_version in packages:
            self.check_package(package_name, import_name, min_version)
    
    def check_geospatial(self):
        """Check geospatial tools"""
        print("\n📌 Checking geospatial tools...")
        
        try:
            import rasterio
            print(f"   ✅ rasterio {rasterio.__version__}")
            self.results['passed'].append("Rasterio")
        except ImportError:
            print("   ❌ rasterio not installed")
            self.results['failed'].append("Rasterio not installed")
        
        try:
            import geopandas
            print(f"   ✅ geopandas {geopandas.__version__}")
            self.results['passed'].append("GeoPandas")
        except ImportError:
            print("   ❌ geopandas not installed")
            self.results['failed'].append("GeoPandas not installed")
    
    def check_config_file(self):
        """Check configuration file"""
        print("\n📌 Checking configuration file...")
        
        config_path = self.root_dir / "config" / "config_vegetation.yaml"
        
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                    # Verify key sections
                    required_sections = ['data', 'model', 'training', 'inference']
                    for section in required_sections:
                        if section in config:
                            print(f"   ✅ [{section}] section present")
                            self.results['passed'].append(f"Config: [{section}]")
                        else:
                            print(f"   ❌ [{section}] section missing")
                            self.results['failed'].append(f"Config: [{section}] missing")
            
            except Exception as e:
                print(f"   ❌ Error reading config: {e}")
                self.results['failed'].append(f"Config error: {e}")
        else:
            print(f"   ❌ config_vegetation.yaml not found")
            self.results['failed'].append("Configuration file not found")
    
    def check_documentation(self):
        """Check documentation files"""
        print("\n📌 Checking documentation...")
        
        docs = [
            "VEGETATION_README.md",
            "VEGETATION_DETECTION_GUIDE.md",
            "QUICK_REFERENCE.md",
            "IMPLEMENTATION_SUMMARY.md",
        ]
        
        for doc in docs:
            doc_path = self.root_dir / doc
            if doc_path.exists():
                size_kb = doc_path.stat().st_size / 1024
                print(f"   ✅ {doc} ({size_kb:.0f}KB)")
                self.results['passed'].append(f"Doc: {doc}")
            else:
                print(f"   ⚠️  {doc} not found")
                self.results['warnings'].append(f"Documentation: {doc} not found")
    
    def test_imports(self):
        """Test critical imports"""
        print("\n📌 Testing critical imports...")
        
        critical_imports = [
            ("torch", "PyTorch"),
            ("rasterio", "Rasterio"),
            ("geopandas", "GeoPandas"),
            ("segmentation_models_pytorch", "Segmentation Models"),
            ("albumentations", "Albumentations"),
        ]
        
        for module_name, display_name in critical_imports:
            try:
                __import__(module_name)
                print(f"   ✅ {display_name}")
                self.results['passed'].append(f"Import: {display_name}")
            except ImportError as e:
                print(f"   ❌ {display_name}: {e}")
                self.results['failed'].append(f"Import {display_name} failed: {e}")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*70)
        print("📊 VALIDATION SUMMARY")
        print("="*70)
        
        # Passed
        if self.results['passed']:
            print(f"\n✅ PASSED ({len(self.results['passed'])})")
            for item in self.results['passed'][:5]:  # Show first 5
                print(f"   • {item}")
            if len(self.results['passed']) > 5:
                print(f"   ... and {len(self.results['passed']) - 5} more")
        
        # Warnings
        if self.results['warnings']:
            print(f"\n⚠️  WARNINGS ({len(self.results['warnings'])})")
            for item in self.results['warnings']:
                print(f"   • {item}")
        
        # Failed
        if self.results['failed']:
            print(f"\n❌ FAILED ({len(self.results['failed'])})")
            for item in self.results['failed']:
                print(f"   • {item}")
        
        # Status
        print("\n" + "-"*70)
        if self.results['failed']:
            print("❌ VALIDATION FAILED - Please fix errors above")
            print("\nTo fix:")
            print("  pip install -r vegetation_requirements.txt")
            return False
        elif self.results['warnings']:
            print("⚠️  VALIDATION PASSED WITH WARNINGS")
            print("System is usable but some features may be limited")
            return True
        else:
            print("✅ VALIDATION PASSED - Ready to use!")
            return True
    
    def run_all_checks(self):
        """Run all validation checks"""
        self.print_header()
        
        self.check_python_version()
        self.check_dependencies()
        self.check_geospatial()
        self.check_gpu()
        self.check_files()
        self.check_directories()
        self.check_config_file()
        self.check_documentation()
        self.test_imports()
        
        return self.print_summary()


def main():
    validator = VegetationSetupValidator()
    success = validator.run_all_checks()
    
    # Print next steps
    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70)
    
    if success:
        print("\n1. Read the Quick Start guide:")
        print("   cat QUICK_REFERENCE.md")
        print("\n2. Create training tiles:")
        print("   python create_vegetation_tiles.py \\")
        print("       --input_tif data/ortho.tif \\")
        print("       --input_shp data/trees.shp \\")
        print("       --output_dir ./tiles")
        print("\n3. Run the complete pipeline:")
        print("   python vegetation_pipeline.py \\")
        print("       --input_tiff data/ortho.tif \\")
        print("       --vegetation_shp data/trees.shp \\")
        print("       --output_dir ./output")
    else:
        print("\nPlease install missing dependencies:")
        print("  pip install -r vegetation_requirements.txt")
        print("\nThen run this script again to verify.")
    
    print("\n📚 Documentation:")
    print("   • VEGETATION_README.md - Overview")
    print("   • VEGETATION_DETECTION_GUIDE.md - Complete guide")
    print("   • QUICK_REFERENCE.md - Commands and tips")
    print("   • IMPLEMENTATION_SUMMARY.md - Technical details")
    print()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
