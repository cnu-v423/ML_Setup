# VEGETATION DETECTION INFERENCE - OPTIMIZED FOR PREDICTION
# This script performs vegetation detection on large raster images
# Uses the vegetation-optimized model with RGB input and computed vegetation indices

import os
import sys
import argparse
import numpy as np
import rasterio
import torch
import torch.nn as nn
from pathlib import Path
from scipy import ndimage
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_vegetation_indices(rgb_image):
    """
    Compute vegetation indices from RGB image for inference.
    
    Args:
        rgb_image: (3, H, W) RGB array normalized to 0-1
        
    Returns:
        combined: (7, H, W) array with RGB + vegetation indices
    """
    red = rgb_image[0].astype(np.float32)
    green = rgb_image[1].astype(np.float32)
    blue = rgb_image[2].astype(np.float32)
    
    # Prevent division by zero
    red = np.clip(red, 1e-6, 1.0)
    green = np.clip(green, 1e-6, 1.0)
    blue = np.clip(blue, 1e-6, 1.0)
    
    # Vegetation indices
    exg = 2 * green - red - blue
    exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    
    ndvi_rgb = (green - red) / (green + red + 1e-6)
    ndvi_rgb = (ndvi_rgb + 1) / 2
    
    gli = (2 * green - red - blue) / (2 * green + red + blue + 1e-6)
    gli = (gli + 1) / 2
    
    color_index = green / (red + blue + 1e-6)
    color_index = np.clip(color_index / 2, 0, 1)
    
    # Stack: RGB + ExG + NDVI-RGB + GLI + ColorIndex
    combined = np.stack([red, green, blue, exg, ndvi_rgb, gli, color_index])
    
    return combined


class VegetationPredictor:
    """Efficient vegetation detection predictor for large images"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.model.eval()
        print(f"✅ Model loaded successfully on {self.device}")
    
    def _load_model(self, model_path):
        """Load the vegetation detection model"""
        # Simple UNet++ with adapter for 7 channels
        import segmentation_models_pytorch as smp
        
        model = smp.UnetPlusPlus(
            encoder_name='senet154',
            encoder_weights=None,  # Weights loaded from checkpoint
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",
            decoder_channels=(256, 128, 64, 32, 16),
            activation='sigmoid'
        )
        
        # Add channel adapter for 7-channel input
        adapter = nn.Conv2d(7, 3, kernel_size=1, padding=0)
        
        class VegetationModel(nn.Module):
            def __init__(self, model, adapter):
                super().__init__()
                self.adapter = adapter
                self.model = model
            
            def forward(self, x):
                x = self.adapter(x)
                return self.model(x)
        
        model = VegetationModel(model, adapter)
        
        # Load weights
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded weights from {model_path}")
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        return model.to(self.device)
    
    def predict_tile(self, tile):
        """
        Predict vegetation on a single tile.
        
        Args:
            tile: (7, H, W) array with RGB + vegetation indices
            
        Returns:
            prediction: (1, H, W) probability map [0-1]
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            tile_tensor = torch.from_numpy(tile).float().unsqueeze(0)  # (1, 7, H, W)
            tile_tensor = tile_tensor.to(self.device)
            
            # Forward pass
            output = self.model(tile_tensor)  # (1, 1, H, W)
            
            # Convert to numpy
            prediction = output.squeeze(0).cpu().numpy()  # (1, H, W)
            
        return prediction
    
    def predict_image(self, image_path, output_path, tile_size=512, overlap=64, 
                     threshold=0.5):
        """
        Predict vegetation on a large image using tiling strategy.
        
        Args:
            image_path: Path to input image
            output_path: Path for output prediction
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles
            threshold: Confidence threshold for vegetation
        """
        print(f"\n🌳 Processing image: {image_path}")
        
        with rasterio.open(image_path) as src:
            profile = src.profile
            height, width = src.height, src.width
            
            # Verify 3 bands (RGB)
            if src.count != 3:
                raise ValueError(f"Expected 3 bands (RGB), got {src.count}")
            
            print(f"📊 Image size: {width}x{height}")
            print(f"🔹 Tile size: {tile_size}x{tile_size} with {overlap}px overlap")
            
            # Initialize output array
            prediction = np.zeros((1, height, width), dtype=np.float32)
            weight_map = np.zeros((height, width), dtype=np.float32)
            
            # Calculate stride
            stride = tile_size - overlap
            
            # Number of tiles
            num_tiles_h = (height - tile_size) // stride + 1
            num_tiles_w = (width - tile_size) // stride + 1
            total_tiles = num_tiles_h * num_tiles_w
            
            print(f"🔄 Processing {total_tiles} tiles...\n")
            
            # Process tiles with progress bar
            pbar = tqdm(total=total_tiles, desc="🌳 Vegetation Detection", ncols=80)
            
            for i in range(0, height - tile_size + 1, stride):
                for j in range(0, width - tile_size + 1, stride):
                    # Read tile
                    window = rasterio.windows.Window(j, i, tile_size, tile_size)
                    rgb_tile = src.read(window=window)  # (3, tile_size, tile_size)
                    
                    # Normalize
                    rgb_tile = rgb_tile.astype(np.float32) / 255.0
                    
                    # Compute vegetation indices
                    tile_features = compute_vegetation_indices(rgb_tile)  # (7, H, W)
                    
                    # Predict
                    tile_pred = self.predict_tile(tile_features)  # (1, H, W)
                    
                    # Add to output with blending weights
                    prediction[0, i:i+tile_size, j:j+tile_size] += tile_pred[0]
                    weight_map[i:i+tile_size, j:j+tile_size] += 1.0
                    
                    pbar.update(1)
                    gc.collect()
            
            pbar.close()
            
            # Normalize by weights (blend overlapping regions)
            prediction[0] = prediction[0] / (weight_map + 1e-6)
            
            # Create output array
            output_data = np.zeros((4, height, width), dtype=np.float32)
            output_data[0:3] = src.read()  # Copy original RGB
            output_data[3] = prediction[0]  # Add prediction
            
            # Update profile for 4 bands
            profile.update(count=4, dtype='float32')
            
            # Write output
            print(f"\n💾 Writing output to {output_path}...")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(output_data)
            
            print(f"✅ Prediction saved to {output_path}")
            
            # Generate binary vegetation map
            binary_path = output_path.replace('.tif', '_binary.tif')
            binary_vegetation = (prediction[0] > threshold).astype(np.uint8) * 255
            
            binary_data = np.zeros((4, height, width), dtype=np.uint8)
            binary_data[0:3] = (src.read() / 255).astype(np.uint8)
            binary_data[3] = binary_vegetation
            
            profile_binary = profile.copy()
            profile_binary.update(count=4, dtype='uint8')
            
            with rasterio.open(binary_path, 'w', **profile_binary) as dst:
                dst.write(binary_data)
            
            print(f"✅ Binary map saved to {binary_path}")
            
            # Compute statistics
            veg_percentage = (prediction[0] > threshold).sum() / (height * width) * 100
            
            print(f"\n📊 Vegetation Statistics:")
            print(f"   • Vegetation coverage: {veg_percentage:.2f}%")
            print(f"   • Mean confidence: {prediction[0].mean():.4f}")
            print(f"   • Median confidence: {np.median(prediction[0]):.4f}")
            print(f"   • Max confidence: {prediction[0].max():.4f}")
            
            return {
                'prediction_path': output_path,
                'binary_path': binary_path,
                'vegetation_percentage': veg_percentage,
                'mean_confidence': prediction[0].mean()
            }


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description='Vegetation Detection Inference'
    )
    parser.add_argument('--input_image', required=True,
                       help='Path to input RGB image')
    parser.add_argument('--output_path', required=True,
                       help='Path for output prediction')
    parser.add_argument('--model_path', required=True,
                       help='Path to trained model weights')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Tile size for processing')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Overlap between tiles')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Vegetation confidence threshold')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Verify input exists
    if not os.path.exists(args.input_image):
        print(f"❌ Input image not found: {args.input_image}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print("\n" + "="*70)
    print("🌳 VEGETATION DETECTION INFERENCE 🌳")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = VegetationPredictor(args.model_path, device=args.device)
    
    # Run prediction
    stats = predictor.predict_image(
        args.input_image,
        args.output_path,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold
    )
    
    print("\n" + "="*70)
    print("✅ Inference completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
