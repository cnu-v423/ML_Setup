#!/usr/bin/env python3
"""
Complete pipeline using Triton for vegetation detection
Orchestrates tile creation, model training, and Triton-based inference
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import time
from datetime import datetime


class TritonVegetationPipeline:
    """Orchestrate vegetation detection with Triton inference server"""
    
    def __init__(self, config_path=None, base_dir="./"):
        self.base_dir = base_dir
        self.config_path = config_path or os.path.join(base_dir, 'config_vegetation.yaml')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def step_1_create_tiles(self, input_tiff, input_shp, output_dir=None):
        """Create vegetation tiles and masks"""
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, f"tiles_veg_{self.timestamp}")
        
        print("\n" + "="*70)
        print("🌳 STEP 1: CREATE VEGETATION TILES")
        print("="*70)
        print(f"Input TIFF: {input_tiff}")
        print(f"Input SHP:  {input_shp}")
        print(f"Output Dir: {output_dir}")
        
        # Run tile creation
        cmd = [
            "python", "create_vegetation_tiles.py",
            "--input_tif", input_tiff,
            "--input_shp", input_shp,
            "--output_dir", output_dir
        ]
        
        result = subprocess.run(cmd, cwd=self.base_dir)
        if result.returncode != 0:
            print("❌ Tile creation failed")
            return False
        
        print("✅ Tiles created successfully")
        return True
    
    def step_2_train_model(self, tiles_dir, masks_dir, model_save_dir=None, skip=False):
        """Train vegetation detection model"""
        if skip:
            print("\n⏭️  Skipping model training (using pre-trained weights)")
            return True
        
        if model_save_dir is None:
            model_save_dir = os.path.join(self.base_dir, f"models_{self.timestamp}")
        
        print("\n" + "="*70)
        print("🎓 STEP 2: TRAIN VEGETATION DETECTION MODEL")
        print("="*70)
        print(f"Tiles Dir:  {tiles_dir}")
        print(f"Masks Dir:  {masks_dir}")
        print(f"Model Dir:  {model_save_dir}")
        
        # Run training
        cmd = [
            "python", "vegetation_detection_training.py",
            "--input_tiles_dir", tiles_dir,
            "--input_masks_dir", masks_dir,
            "--model_path", model_save_dir
        ]
        
        result = subprocess.run(cmd, cwd=self.base_dir)
        if result.returncode != 0:
            print("❌ Training failed")
            return False
        
        print("✅ Model trained successfully")
        return True
    
    def step_3_export_to_triton(self, model_path, triton_output_dir=None):
        """Export model to Triton format"""
        if triton_output_dir is None:
            triton_output_dir = os.path.join(self.base_dir, "triton_models")
        
        print("\n" + "="*70)
        print("📦 STEP 3: EXPORT MODEL TO TRITON FORMAT")
        print("="*70)
        print(f"Model Path:        {model_path}")
        print(f"Triton Output Dir: {triton_output_dir}")
        
        cmd = [
            "python", "export_to_triton.py",
            "--model_path", model_path,
            "--output_dir", triton_output_dir
        ]
        
        result = subprocess.run(cmd, cwd=self.base_dir)
        if result.returncode != 0:
            print("❌ Export to Triton failed")
            return False
        
        print("✅ Model exported successfully")
        return True
    
    def step_4_launch_triton_server(self, triton_model_repo, gpu_id=0, port=8000):
        """Launch Triton Inference Server"""
        print("\n" + "="*70)
        print("🚀 STEP 4: LAUNCH TRITON INFERENCE SERVER")
        print("="*70)
        print(f"Model Repository: {triton_model_repo}")
        print(f"GPU ID:          {gpu_id}")
        print(f"Port:            {port}")
        
        # Make launch script executable
        launch_script = os.path.join(self.base_dir, "launch_triton.sh")
        if os.path.exists(launch_script):
            os.chmod(launch_script, 0o755)
            
            cmd = [
                "bash", launch_script,
                "--gpu", str(gpu_id),
                "--port", str(port),
                "--model_repo", triton_model_repo
            ]
            
            print(f"Launching: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=self.base_dir)
        else:
            print(f"⚠️  Launch script not found: {launch_script}")
    
    def step_5_run_inference(self, image_path, output_dir, 
                            triton_url="localhost:8000", 
                            fallback_model=None, use_local=False):
        """Run Triton-based inference on large image"""
        print("\n" + "="*70)
        print("🔮 STEP 5: RUN TRITON-BASED INFERENCE")
        print("="*70)
        print(f"Input Image:  {image_path}")
        print(f"Output Dir:   {output_dir}")
        print(f"Triton URL:   {triton_url}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 
                                   os.path.splitext(os.path.basename(image_path))[0] + 
                                   "_vegetation.tif")
        
        cmd = [
            "python", "triton_vegetation_inference.py",
            "--image", image_path,
            "--output", output_path,
            "--triton_url", triton_url,
            "--model_name", "vegetation_detector"
        ]
        
        if fallback_model:
            cmd.extend(["--fallback_model", fallback_model])
        
        if use_local:
            cmd.append("--use_local")
        
        result = subprocess.run(cmd, cwd=self.base_dir)
        if result.returncode != 0:
            print("❌ Inference failed")
            return False
        
        print("✅ Inference completed successfully")
        return True
    
    def run_full_pipeline(self, input_tiff, input_shp, image_for_inference,
                         skip_training=False, gpu_id=0, triton_port=8000):
        """Execute complete pipeline with Triton"""
        print("\n" + "="*80)
        print("🌳 VEGETATION DETECTION PIPELINE WITH TRITON INFERENCE")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print("="*80)
        
        # Create directories
        tiles_dir = os.path.join(self.base_dir, f"tiles_veg_{self.timestamp}")
        masks_dir = os.path.join(self.base_dir, f"masks_veg_{self.timestamp}")
        model_dir = os.path.join(self.base_dir, f"models_{self.timestamp}")
        triton_dir = os.path.join(self.base_dir, "triton_models")
        output_dir = os.path.join(self.base_dir, f"predictions_{self.timestamp}")
        
        # Step 1: Create tiles
        if not self.step_1_create_tiles(input_tiff, input_shp, tiles_dir):
            return False
        
        # Step 2: Train model
        if not skip_training:
            if not self.step_2_train_model(tiles_dir, masks_dir, model_dir):
                return False
            model_path = os.path.join(model_dir, "vegetation_unet_best.pt")
        else:
            model_path = None
        
        # Step 3: Export to Triton (if model path available)
        if model_path and os.path.exists(model_path):
            if not self.step_3_export_to_triton(model_path, triton_dir):
                print("⚠️  Continuing with fallback...")
        
        # Step 4: Launch Triton server (in background)
        print("\n💡 Launch Triton server separately:")
        print(f"   bash launch_triton.sh --gpu {gpu_id} --port {triton_port} --model_repo {triton_dir}")
        print(f"   Then update --triton_url parameter in step 5")
        
        # Step 5: Run inference
        triton_url = f"localhost:{triton_port}"
        if not self.step_5_run_inference(image_for_inference, output_dir, 
                                        triton_url=triton_url,
                                        fallback_model=model_path,
                                        use_local=False):
            print("⚠️  Trying local inference as fallback...")
            if not self.step_5_run_inference(image_for_inference, output_dir,
                                            fallback_model=model_path,
                                            use_local=True):
                return False
        
        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Outputs saved to: {output_dir}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton vegetation detection pipeline")
    parser.add_argument('--input_tiff', required=True, help="Input RGB TIFF")
    parser.add_argument('--input_shp', required=True, help="Vegetation shapefile")
    parser.add_argument('--inference_image', required=True, help="Image for inference")
    parser.add_argument('--skip_training', action='store_true', help="Skip training phase")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID")
    parser.add_argument('--triton_port', type=int, default=8000, help="Triton port")
    parser.add_argument('--base_dir', default='./', help="Base directory")
    
    args = parser.parse_args()
    
    pipeline = TritonVegetationPipeline(base_dir=args.base_dir)
    success = pipeline.run_full_pipeline(
        input_tiff=args.input_tiff,
        input_shp=args.input_shp,
        image_for_inference=args.inference_image,
        skip_training=args.skip_training,
        gpu_id=args.gpu,
        triton_port=args.triton_port
    )
    
    sys.exit(0 if success else 1)
