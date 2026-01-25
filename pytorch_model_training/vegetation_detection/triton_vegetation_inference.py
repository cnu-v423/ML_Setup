# TRITON-ENABLED VEGETATION DETECTION INFERENCE
# Uses NVIDIA Triton Inference Server for high-performance prediction
# Supports both CPU and GPU with optimal batching

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

try:
    import tritonclient.http as httpclient
    from tritonclient.utils import np_to_triton_dtype
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠️  Triton client not available. Install with: pip install tritonclient")


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


class TritonVegetationPredictor:
    """
    High-performance vegetation detection predictor using NVIDIA Triton Inference Server.
    Enables batched inference, model versioning, and multi-model deployment.
    """
    
    def __init__(self, triton_url="localhost:8000", model_name="vegetation_detector", 
                 use_triton=True, fallback_model_path=None, device='cuda'):
        """
        Initialize Triton predictor with fallback to local inference.
        
        Args:
            triton_url: Triton server URL (host:port)
            model_name: Deployed model name in Triton
            use_triton: Whether to use Triton or local inference
            fallback_model_path: Path to model for fallback (local) inference
            device: Device for fallback (cuda/cpu)
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.use_triton = use_triton and TRITON_AVAILABLE
        self.fallback_model_path = fallback_model_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if self.use_triton:
            try:
                self.triton_client = httpclient.InferenceServerClient(url=triton_url)
                # Check if server is ready
                if not self.triton_client.is_server_ready():
                    print(f"⚠️  Triton server not ready at {triton_url}")
                    self._use_fallback()
                else:
                    print(f"✅ Connected to Triton server at {triton_url}")
                    print(f"✅ Using model: {model_name}")
            except Exception as e:
                print(f"⚠️  Failed to connect to Triton: {e}")
                self._use_fallback()
        else:
            self._use_fallback()
    
    def _use_fallback(self):
        """Switch to local PyTorch inference"""
        print("🔄 Switching to local PyTorch inference...")
        self.use_triton = False
        if self.fallback_model_path:
            self.model = self._load_local_model(self.fallback_model_path)
        else:
            print("⚠️  No fallback model provided")
    
    def _load_local_model(self, model_path):
        """Load PyTorch model locally"""
        import segmentation_models_pytorch as smp
        
        model = smp.UnetPlusPlus(
            encoder_name='senet154',
            encoder_weights=None,
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
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✅ Loaded local model from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return model.to(self.device).eval()
    
    def predict_tile_triton(self, tile):
        """
        Predict using Triton Inference Server.
        
        Args:
            tile: (7, H, W) array with RGB + vegetation indices
            
        Returns:
            prediction: (1, H, W) probability map [0-1]
        """
        # Prepare input
        tile_input = tile.astype(np.float32)
        tile_input = np.expand_dims(tile_input, 0)  # Add batch dimension: (1, 7, H, W)
        
        # Create input object
        inputs = [httpclient.InferInput("images", tile_input.shape, "FP32")]
        inputs[0].set_data_from_numpy(tile_input)
        
        # Create output object
        outputs = [httpclient.InferRequestedOutput("output")]
        
        # Perform inference
        results = self.triton_client.infer(self.model_name, inputs=inputs, outputs=outputs)
        
        # Extract output
        output_data = results.as_numpy("output")  # (1, 1, H, W)
        
        return output_data[0]  # Return (1, H, W)
    
    def predict_tile_local(self, tile):
        """
        Predict using local PyTorch model.
        
        Args:
            tile: (7, H, W) array with RGB + vegetation indices
            
        Returns:
            prediction: (1, H, W) probability map [0-1]
        """
        with torch.no_grad():
            tile_tensor = torch.from_numpy(tile).float().unsqueeze(0)  # (1, 7, H, W)
            tile_tensor = tile_tensor.to(self.device)
            
            output = self.model(tile_tensor)  # (1, 1, H, W)
            prediction = output.squeeze(0).cpu().numpy()  # (1, H, W)
        
        return prediction
    
    def predict_tile(self, tile):
        """Wrapper that uses either Triton or local inference"""
        if self.use_triton:
            try:
                return self.predict_tile_triton(tile)
            except Exception as e:
                print(f"⚠️  Triton inference failed: {e}. Using local inference.")
                return self.predict_tile_local(tile)
        else:
            return self.predict_tile_local(tile)
    
    def predict_image(self, image_path, output_path, tile_size=512, overlap=64, 
                     threshold=0.5):
        """
        Predict vegetation on a large image using tiling strategy with Triton.
        
        Args:
            image_path: Path to input image
            output_path: Path for output prediction
            tile_size: Size of tiles for processing
            overlap: Overlap between tiles
            threshold: Confidence threshold for vegetation
        """
        print(f"\n🌳 Processing image: {image_path}")
        if self.use_triton:
            print(f"🚀 Using Triton server at {self.triton_url}")
        else:
            print(f"⚙️  Using local inference on {self.device}")
        
        with rasterio.open(image_path) as src:
            profile = src.profile
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            
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
            pbar = tqdm(total=total_tiles, desc="🌳 Vegetation Detection (Triton)", ncols=100)
            
            for i in range(0, height - tile_size + 1, stride):
                for j in range(0, width - tile_size + 1, stride):
                    # Read tile
                    window = rasterio.windows.Window(j, i, tile_size, tile_size)
                    rgb_tile = src.read(window=window)  # (3, tile_size, tile_size)
                    
                    # Normalize
                    rgb_tile = rgb_tile.astype(np.float32) / 255.0
                    
                    # Compute vegetation indices
                    tile_features = compute_vegetation_indices(rgb_tile)  # (7, H, W)
                    
                    # Predict (Triton or local)
                    tile_pred = self.predict_tile(tile_features)  # (1, H, W)
                    
                    # Add to output with blending weights
                    prediction[0, i:i+tile_size, j:j+tile_size] += tile_pred[0]
                    weight_map[i:i+tile_size, j:j+tile_size] += 1.0
                    
                    pbar.update(1)
                    gc.collect()
            
            pbar.close()
            
            # Normalize by weights (blend overlapping regions)
            prediction[0] = prediction[0] / (weight_map + 1e-6)
            
            # Create output array with RGB + probability
            output_data = np.zeros((4, height, width), dtype=np.float32)
            output_data[:3] = prediction  # RGB channels (same probability 3x)
            output_data[3] = prediction[0]  # Probability channel
            
            # Binary map
            binary_data = (prediction[0] > threshold).astype(np.uint8) * 255
            
            # Save predictions (GeoTIFF format)
            profile.update(count=4, dtype=np.float32, nodata=0)
            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, 5):
                    dst.write(output_data[i-1], i)
            
            # Save binary map
            binary_path = output_path.replace('.tif', '_binary.tif')
            profile.update(count=1, dtype=np.uint8, nodata=0)
            with rasterio.open(binary_path, 'w', **profile) as dst:
                dst.write(binary_data, 1)
            
            print(f"\n✅ Predictions saved:")
            print(f"   📊 Probability map: {output_path}")
            print(f"   🔲 Binary map: {binary_path}")
            
            # Calculate vegetation coverage
            veg_pixels = (binary_data > 0).sum()
            total_pixels = binary_data.size
            coverage = 100 * veg_pixels / total_pixels
            print(f"   🌳 Vegetation coverage: {coverage:.2f}%")
            
            return output_path, binary_path


class TritonBatchPredictor:
    """
    Optimized predictor for batch processing multiple images with Triton.
    Leverages Triton's dynamic batching for maximum throughput.
    """
    
    def __init__(self, triton_url="localhost:8000", model_name="vegetation_detector",
                 fallback_model_path=None):
        self.predictor = TritonVegetationPredictor(
            triton_url=triton_url,
            model_name=model_name,
            use_triton=True,
            fallback_model_path=fallback_model_path
        )
    
    def predict_images(self, image_paths, output_dir, **kwargs):
        """
        Predict vegetation on multiple images.
        
        Args:
            image_paths: List of input image paths
            output_dir: Output directory for predictions
            **kwargs: Additional arguments for predict_image
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for image_path in image_paths:
            basename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{basename}_vegetation.tif")
            
            pred_path, binary_path = self.predictor.predict_image(
                image_path, output_path, **kwargs
            )
            
            results.append({
                'input': image_path,
                'probability': pred_path,
                'binary': binary_path
            })
        
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton-based vegetation detection")
    parser.add_argument('--image', help="Input image path")
    parser.add_argument('--output', help="Output prediction path")
    parser.add_argument('--triton_url', default='localhost:8000', help="Triton server URL")
    parser.add_argument('--model_name', default='vegetation_detector', help="Model name in Triton")
    parser.add_argument('--fallback_model', help="Local PyTorch model path (fallback)")
    parser.add_argument('--tile_size', type=int, default=512, help="Tile size")
    parser.add_argument('--overlap', type=int, default=64, help="Tile overlap")
    parser.add_argument('--threshold', type=float, default=0.5, help="Binary threshold")
    parser.add_argument('--use_local', action='store_true', help="Force local inference")
    
    args = parser.parse_args()
    
    if args.image and args.output:
        predictor = TritonVegetationPredictor(
            triton_url=args.triton_url,
            model_name=args.model_name,
            use_triton=not args.use_local,
            fallback_model_path=args.fallback_model,
            device='cuda'
        )
        
        predictor.predict_image(
            args.image,
            args.output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            threshold=args.threshold
        )
    else:
        print("Usage: python triton_vegetation_inference.py --image <input> --output <output>")
        print("       [--triton_url localhost:8000] [--model_name vegetation_detector]")
        print("       [--fallback_model path/to/model.pt] [--use_local]")
