#!/usr/bin/env python3
"""
VEGETATION DETECTION COMPLETE PIPELINE
This script runs the entire vegetation detection workflow:
1. Create tiles and masks from RGB raster and vegetation shapefile
2. Train vegetation detection model
3. Run inference on new images
"""

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


class VegetationPipeline:
    def __init__(self, input_tiff, vegetation_shp, output_dir, model_weights=None):
        self.input_tiff = Path(input_tiff).absolute()
        self.vegetation_shp = Path(vegetation_shp).absolute()
        self.output_dir = Path(output_dir).absolute()
        self.model_weights = Path(model_weights).absolute() if model_weights else None
        
        # Validate inputs
        self.validate_inputs()
        
        # Create subdirectories
        self.tiles_dir = self.output_dir / "tiles_masks" / "tiles_veg"
        self.masks_dir = self.output_dir / "tiles_masks" / "masks_veg"
        self.model_dir = self.output_dir / "model"
        self.predictions_dir = self.output_dir / "predictions"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline status
        self.status = {}
        self.config = {
            'tile_size': 512,
            'overlap': 0.25,
            'batch_size': 8,
            'epochs': 150,
            'learning_rate': 0.001
        }
    
    def validate_inputs(self):
        """Validate input files exist"""
        errors = []
        
        if not self.input_tiff.exists():
            errors.append(f"❌ Input TIFF not found: {self.input_tiff}")
        
        if not self.vegetation_shp.exists():
            errors.append(f"❌ Vegetation shapefile not found: {self.vegetation_shp}")
        
        if self.model_weights and not self.model_weights.exists():
            errors.append(f"❌ Model weights not found: {self.model_weights}")
        
        if errors:
            for error in errors:
                print(error)
            sys.exit(1)
        
        print(f"✅ Input validation passed")
        print(f"   📁 Input TIFF: {self.input_tiff}")
        print(f"   📁 Vegetation SHP: {self.vegetation_shp}")
    
    def step_1_create_tiles(self):
        """Step 1: Create tiles and masks"""
        print("\n" + "="*70)
        print("STEP 1️⃣  CREATE VEGETATION TILES")
        print("="*70)
        
        cmd = [
            sys.executable, "create_vegetation_tiles.py",
            "--input_tif", str(self.input_tiff),
            "--input_shp", str(self.vegetation_shp),
            "--output_dir", str(self.output_dir / "tiles_masks"),
            "--tile_size", str(self.config['tile_size']),
            "--overlap", str(self.config['overlap'])
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            # Verify tiles were created
            if not self.tiles_dir.exists() or not list(self.tiles_dir.glob('*.tif')):
                print("❌ No tiles were created. Check input data.")
                self.status['tile_creation'] = 'FAILED'
                return False
            
            num_tiles = len(list(self.tiles_dir.glob('*.tif')))
            print(f"✅ Created {num_tiles} vegetation tiles")
            self.status['tile_creation'] = f'SUCCESS ({num_tiles} tiles)'
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Tile creation failed: {e}")
            self.status['tile_creation'] = 'FAILED'
            return False
    
    def step_2_train_model(self):
        """Step 2: Train vegetation detection model"""
        print("\n" + "="*70)
        print("STEP 2️⃣  TRAIN VEGETATION DETECTION MODEL")
        print("="*70)
        
        # Skip if model weights provided
        if self.model_weights:
            print(f"⏭️  Using provided model: {self.model_weights}")
            self.status['model_training'] = 'SKIPPED (provided weights)'
            return True
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            sys.executable,
            "pytorch_model_training/vegetation_detection_training.py",
            "--input_tiles_dir", str(self.tiles_dir),
            "--input_masks_dir", str(self.masks_dir),
            "--model_path", str(self.model_dir)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            # Find best model
            best_models = list(self.model_dir.glob('*best*.pt'))
            if best_models:
                self.model_weights = best_models[0]
                print(f"✅ Model trained: {self.model_weights}")
                self.status['model_training'] = 'SUCCESS'
                return True
            else:
                print("❌ Training completed but no best model found")
                self.status['model_training'] = 'FAILED'
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Model training failed: {e}")
            self.status['model_training'] = 'FAILED'
            return False
    
    def step_3_run_inference(self, threshold=0.5, tile_size=512, overlap=64):
        """Step 3: Run inference on the input image"""
        print("\n" + "="*70)
        print("STEP 3️⃣  RUN VEGETATION DETECTION INFERENCE")
        print("="*70)
        
        if not self.model_weights:
            print("❌ No model weights available for inference")
            self.status['inference'] = 'FAILED'
            return False
        
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        output_prediction = self.predictions_dir / "vegetation_prediction.tif"
        
        cmd = [
            sys.executable, "vegetation_inference.py",
            "--input_image", str(self.input_tiff),
            "--output_path", str(output_prediction),
            "--model_path", str(self.model_weights),
            "--tile_size", str(tile_size),
            "--overlap", str(overlap),
            "--threshold", str(threshold),
            "--device", "cuda"
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            if output_prediction.exists():
                print(f"✅ Inference completed: {output_prediction}")
                self.status['inference'] = 'SUCCESS'
                return True
            else:
                print("❌ Inference failed - output file not created")
                self.status['inference'] = 'FAILED'
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference failed: {e}")
            self.status['inference'] = 'FAILED'
            return False
    
    def run_full_pipeline(self, skip_training=False, threshold=0.5):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print("🌳 VEGETATION DETECTION PIPELINE")
        print("="*70)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 1: Create tiles
        if not self.step_1_create_tiles():
            print("\n❌ Pipeline failed at tile creation")
            return False
        
        # Step 2: Train model
        if not skip_training and not self.model_weights:
            if not self.step_2_train_model():
                print("\n❌ Pipeline failed at model training")
                return False
        
        # Step 3: Run inference
        if not self.step_3_run_inference(threshold=threshold):
            print("\n❌ Pipeline failed at inference")
            return False
        
        # Summary
        self.print_summary()
        return True
    
    def print_summary(self):
        """Print pipeline summary"""
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
        print("="*70)
        
        print(f"\n📊 Pipeline Status:")
        for step, status in self.status.items():
            status_icon = "✅" if status.startswith("SUCCESS") else "⏭️" if status.startswith("SKIPPED") else "❌"
            print(f"   {status_icon} {step.upper()}: {status}")
        
        print(f"\n📁 Output Files:")
        print(f"   📊 Tiles directory:    {self.tiles_dir}")
        print(f"   🎭 Masks directory:    {self.masks_dir}")
        
        if self.model_weights:
            print(f"   🧠 Model weights:      {self.model_weights}")
        
        predictions = list(self.predictions_dir.glob("*.tif"))
        for pred in predictions:
            print(f"   🌳 Prediction:         {pred}")
        
        print(f"\n⏱️  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review predictions in GIS software")
        print(f"   2. Validate accuracy on test set")
        print(f"   3. Adjust threshold if needed (currently: 0.5)")
        print(f"   4. Post-process if necessary (morphological operations)")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Complete Vegetation Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: create tiles → train → infer
  python vegetation_pipeline.py --input_tiff ortho.tif --vegetation_shp trees.shp --output_dir output
  
  # Use existing model
  python vegetation_pipeline.py --input_tiff ortho.tif --vegetation_shp trees.shp --output_dir output --model_weights best_model.pt
  
  # Inference only on large image with existing tiles/model
  python vegetation_pipeline.py --input_tiff large_image.tif --vegetation_shp dummy.shp --output_dir output --model_weights best_model.pt --skip_tiles
        """
    )
    
    parser.add_argument('--input_tiff', required=True,
                       help='Path to input RGB TIFF file')
    parser.add_argument('--vegetation_shp', required=True,
                       help='Path to vegetation mask shapefile')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for all results')
    parser.add_argument('--model_weights', default=None,
                       help='Path to pre-trained model (optional)')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip model training (use provided weights)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Vegetation confidence threshold (default: 0.5)')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Tile size for inference (default: 512)')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Tile overlap for inference (default: 64)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = VegetationPipeline(
        args.input_tiff,
        args.vegetation_shp,
        args.output_dir,
        args.model_weights
    )
    
    # Run pipeline
    success = pipeline.run_full_pipeline(
        skip_training=args.skip_training or (args.model_weights is not None),
        threshold=args.threshold
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
