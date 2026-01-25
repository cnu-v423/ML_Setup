"""
Data Validation Script for Multi-Class Road Segmentation
Ensures tiles and masks are correctly formatted and aligned
"""

import os
import glob
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
import argparse


class DataValidator:
    """Validate training data for multi-class segmentation"""
    
    def __init__(self, tiles_dir, masks_dir):
        self.tiles_dir = Path(tiles_dir)
        self.masks_dir = Path(masks_dir)
        self.class_names = ['Background', 'Thar Road', 'CC Road', 'Mud/Gravel Road']
        self.issues = []
        
    def get_filename_without_extension(self, filepath):
        """Extract filename without extension"""
        basename = os.path.basename(filepath)
        return os.path.splitext(basename)[0]
    
    def validate_all(self, verbose=True):
        """Run all validation checks"""
        print("\n🔍 STARTING DATA VALIDATION")
        print("="*70)
        
        # Check file existence
        tile_files = list(self.tiles_dir.glob('*.tif'))
        mask_files = list(self.masks_dir.glob('*.tif'))
        
        print(f"\n📊 File Count:")
        print(f"  • Tiles: {len(tile_files)}")
        print(f"  • Masks: {len(mask_files)}")
        
        if len(tile_files) == 0 or len(mask_files) == 0:
            print("❌ ERROR: No TIF files found!")
            return False
        
        # Match tiles and masks
        tile_dict = {self.get_filename_without_extension(f): f for f in tile_files}
        mask_dict = {self.get_filename_without_extension(f): f for f in mask_files}
        
        common_files = set(tile_dict.keys()).intersection(set(mask_dict.keys()))
        
        print(f"\n🔗 Paired Files:")
        print(f"  • Matched pairs: {len(common_files)}")
        
        if len(common_files) == 0:
            print("❌ ERROR: No matching tile-mask pairs!")
            return False
        
        missing_masks = set(tile_dict.keys()) - set(mask_dict.keys())
        missing_tiles = set(mask_dict.keys()) - set(tile_dict.keys())
        
        if missing_masks:
            print(f"  ⚠️  Tiles without masks: {len(missing_masks)}")
            for name in list(missing_masks)[:3]:
                print(f"       - {name}.tif")
        
        if missing_tiles:
            print(f"  ⚠️  Masks without tiles: {len(missing_tiles)}")
            for name in list(missing_tiles)[:3]:
                print(f"       - {name}.tif")
        
        # Check individual files
        print(f"\n✓ Checking {len(common_files)} tile-mask pairs...")
        
        valid_pairs = 0
        total_pixels_per_class = np.zeros(4)
        image_shapes = []
        mask_shapes = []
        class_distributions = {}
        
        for name in tqdm(list(common_files)[:100]):  # Sample 100 for speed
            tile_path = tile_dict[name]
            mask_path = mask_dict[name]
            
            # Check tile
            try:
                with rasterio.open(tile_path) as src:
                    tile = src.read()
                    image_shapes.append(tile.shape)
                    
                    # Check for NaN/Inf
                    if np.any(np.isnan(tile)) or np.any(np.isinf(tile)):
                        self.issues.append(f"❌ {tile_path}: Contains NaN or Inf values")
                    
                    # Check data range
                    if tile.min() < 0 or tile.max() > 65535:
                        self.issues.append(f"⚠️  {tile_path}: Unusual value range [{tile.min()}, {tile.max()}]")
            
            except Exception as e:
                self.issues.append(f"❌ {tile_path}: {str(e)}")
                continue
            
            # Check mask
            try:
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)
                    mask_shapes.append(mask.shape)
                    
                    # Check class values
                    unique_classes = np.unique(mask)
                    
                    if not all(c in [0, 1, 2, 3] for c in unique_classes):
                        self.issues.append(
                            f"❌ {mask_path}: Invalid class values {unique_classes}. "
                            f"Expected [0, 1, 2, 3]"
                        )
                    
                    # Track class distribution
                    for c in range(4):
                        total_pixels_per_class[c] += (mask == c).sum()
                    
                    # Per-file distribution
                    if name not in class_distributions:
                        class_distributions[name] = {}
                    for c in unique_classes:
                        class_distributions[name][c] = (mask == c).sum()
                    
                    valid_pairs += 1
            
            except Exception as e:
                self.issues.append(f"❌ {mask_path}: {str(e)}")
                continue
            
            # Check alignment
            if tile.shape[1:] != mask.shape:
                self.issues.append(
                    f"❌ {name}: Size mismatch. Tile {tile.shape[1:]} vs Mask {mask.shape}"
                )
        
        # Print results
        print(f"\n✅ Valid Pairs: {valid_pairs} / {len(list(common_files)[:100])}")
        
        # Check shape consistency
        if image_shapes:
            print(f"\n📐 Shape Consistency:")
            unique_shapes = set(image_shapes)
            print(f"  • Unique image shapes: {len(unique_shapes)}")
            for shape in unique_shapes:
                print(f"    - {shape}")
            
            if len(unique_shapes) > 1:
                self.issues.append("⚠️  Inconsistent image shapes detected")
        
        if mask_shapes:
            unique_mask_shapes = set(mask_shapes)
            print(f"  • Unique mask shapes: {len(unique_mask_shapes)}")
            for shape in unique_mask_shapes:
                print(f"    - {shape}")
        
        # Print class distribution
        print(f"\n🗺️  Class Distribution (sampled):")
        total = total_pixels_per_class.sum()
        
        for c in range(4):
            count = total_pixels_per_class[c]
            if total > 0:
                percentage = (count / total) * 100
                print(f"  • {self.class_names[c]}: {count:,} pixels ({percentage:.2f}%)")
        
        # Check for class imbalance
        if total > 0:
            percentages = (total_pixels_per_class / total) * 100
            max_pct = percentages.max()
            min_pct = percentages.min()
            imbalance_ratio = max_pct / (min_pct + 1e-7)
            
            print(f"\n⚖️  Class Balance:")
            print(f"  • Max percentage: {max_pct:.2f}%")
            print(f"  • Min percentage: {min_pct:.2f}%")
            print(f"  • Imbalance ratio: {imbalance_ratio:.2f}x")
            
            if imbalance_ratio > 10:
                print(f"  ⚠️  HIGH IMBALANCE - Will use class weighting during training")
        
        # Report issues
        if self.issues:
            print(f"\n⚠️  ISSUES FOUND: {len(self.issues)}")
            print("="*70)
            for issue in self.issues[:10]:  # Show first 10
                print(f"  {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more issues")
        else:
            print(f"\n✅ NO ISSUES FOUND!")
        
        print("\n" + "="*70)
        
        # Summary
        success = len(self.issues) == 0
        print(f"\n🎯 Validation {'PASSED ✅' if success else 'FAILED ❌'}")
        
        return success
    
    def check_specific_pair(self, tile_name):
        """Check a specific tile-mask pair"""
        tile_files = list(self.tiles_dir.glob('*.tif'))
        mask_files = list(self.masks_dir.glob('*.tif'))
        
        tile_dict = {self.get_filename_without_extension(f): f for f in tile_files}
        mask_dict = {self.get_filename_without_extension(f): f for f in mask_files}
        
        if tile_name not in tile_dict:
            print(f"❌ Tile {tile_name} not found")
            return
        
        if tile_name not in mask_dict:
            print(f"❌ Mask {tile_name} not found")
            return
        
        tile_path = tile_dict[tile_name]
        mask_path = mask_dict[tile_name]
        
        print(f"\n🔍 Checking: {tile_name}")
        print("="*70)
        
        # Tile info
        print(f"\n📷 Tile: {tile_path}")
        with rasterio.open(tile_path) as src:
            tile = src.read()
            print(f"  • Shape: {tile.shape}")
            print(f"  • Dtype: {tile.dtype}")
            print(f"  • Min: {tile.min()}, Max: {tile.max()}")
            print(f"  • Mean: {tile.mean():.2f}, Std: {tile.std():.2f}")
            
            for i in range(tile.shape[0]):
                ch = tile[i]
                print(f"  • Channel {i}: [{ch.min()}, {ch.max()}]")
        
        # Mask info
        print(f"\n🗺️  Mask: {mask_path}")
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            print(f"  • Shape: {mask.shape}")
            print(f"  • Dtype: {mask.dtype}")
            print(f"  • Unique values: {np.unique(mask)}")
            
            for c in range(4):
                count = (mask == c).sum()
                pct = (count / mask.size) * 100
                print(f"  • {self.class_names[c]}: {count:,} ({pct:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Validate multi-class road segmentation data')
    parser.add_argument('--tiles_dir', required=True, help='Directory with tile images')
    parser.add_argument('--masks_dir', required=True, help='Directory with mask tiles')
    parser.add_argument('--check', help='Check specific tile (by name without extension)')
    args = parser.parse_args()
    
    validator = DataValidator(args.tiles_dir, args.masks_dir)
    
    if args.check:
        validator.check_specific_pair(args.check)
    else:
        success = validator.validate_all()
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
