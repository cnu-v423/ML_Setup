import os
import tempfile
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling
import yaml
import gc
from pathlib import Path
from skimage import exposure
from tqdm import tqdm
from model import create_unet_model


class MemoryEfficientPredictor:
    """Memory-efficient predictor that processes large images tile by tile with proper resampling"""
    
    def __init__(self, config_path='config/config_v1.yaml', input_image=''):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tile_size = self.config['data']['input_size']
        self.target_resolution = self.config['data']['resolution']
        self.channels = self.config['data']['channels']
        overlap_percentage = self.config['data']['overlap_percentage']
        self.overlap = int(self.tile_size * overlap_percentage)
        self.input_image = input_image
        self.models = {}
        self.thresholds = {}
        
        # Create temporary directory for storing tile predictions
        self.temp_dir = tempfile.mkdtemp(prefix='prediction_tiles_')
        print(f"Using temporary directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def load_models(self, model_paths):
        """Load models with their thresholds"""
        for name, (model_path, threshold) in model_paths.items():
            if model_path and os.path.exists(model_path):
                print(f"Loading {name} model from {model_path}...")
                model = create_unet_model(self.config)
                model.load_weights(model_path)
                self.models[name] = model
                self.thresholds[name] = threshold

    def calculate_zoom_factor(self, src_resolution, target_resolution):
        """Calculate zoom factor based on source and target resolutions"""
        if isinstance(src_resolution, tuple):
            src_res = min(abs(src_resolution[0]), abs(src_resolution[1]))
        else:
            src_res = abs(src_resolution)
        
        target_res = abs(target_resolution)
        zoom_factor = src_res / target_res
        
        print(f"Source resolution: {src_res:.4f}m, Target resolution: {target_res:.4f}m")
        print(f"Zoom factor: {zoom_factor:.4f}")
        
        return zoom_factor

    def calculate_tile_grid_with_overlap(self, width, height, tile_size, overlap_percentage):
        """
        Calculate tile positions with overlap
        Returns list of (row_start, col_start, actual_height, actual_width, i, j) for each tile
        """
        overlap_pixels = int(tile_size * overlap_percentage)
        step_size = tile_size - overlap_pixels
        
        print(f"Tile size: {tile_size}, Overlap: {overlap_pixels} pixels ({overlap_percentage*100}%)")
        print(f"Step size: {step_size}")
        
        tiles = []
        
        # Calculate number of tiles needed
        n_tiles_height = int(np.ceil((height - overlap_pixels) / step_size))
        n_tiles_width = int(np.ceil((width - overlap_pixels) / step_size))
        
        print(f"Grid size: {n_tiles_height} x {n_tiles_width} tiles")
        
        for i in range(n_tiles_height):
            for j in range(n_tiles_width):
                # Calculate tile position
                row_start = i * step_size
                col_start = j * step_size
                
                # Ensure we don't exceed image boundaries
                row_end = min(row_start + tile_size, height)
                col_end = min(col_start + tile_size, width)
                
                actual_height = row_end - row_start
                actual_width = col_end - col_start
                
                # Only add tiles that have reasonable size
                if actual_height >= tile_size // 2 and actual_width >= tile_size // 2:
                    tiles.append((row_start, col_start, actual_height, actual_width, i, j))
        
        return tiles

    def process_tile_with_zoom(self, src, window, zoom_factor, target_resolution):
        """
        Process a single tile with zoom/resampling, using first 3 bands
        """
        # Read only the first three bands of the original tile
        original_tile = src.read([1, 2, 3], window=window)
        original_height, original_width = window.height, window.width
        
        # Calculate new dimensions based on zoom factor
        new_height = int(original_height * zoom_factor)
        new_width = int(original_width * zoom_factor)
        
        # Get the window transform
        window_transform = rasterio.windows.transform(window, src.transform)
        
        # Create new transform for the resampled tile
        new_transform = rasterio.Affine(
            window_transform.a / zoom_factor,  # new pixel width
            window_transform.b,
            window_transform.c,  # same x origin
            window_transform.d,
            window_transform.e / zoom_factor,  # new pixel height (negative for top-to-bottom)
            window_transform.f   # same y origin
        )
        
        # Create output array for exactly 3 bands
        resampled_tile = np.zeros((3, new_height, new_width), dtype=original_tile.dtype)
        
        # Resample each of the three bands
        for band_idx in range(3):
            reproject(
                source=original_tile[band_idx],
                destination=resampled_tile[band_idx],
                src_transform=window_transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear
            )
        
        return resampled_tile, new_transform, new_height, new_width

    def pad_or_crop_to_tile_size(self, data, tile_size):
        """Pad or crop data to exact tile size"""
        if len(data.shape) == 3:  # Multi-band
            bands, height, width = data.shape
            result = np.zeros((bands, tile_size, tile_size), dtype=data.dtype)
            
            # Calculate how much to copy
            copy_height = min(height, tile_size)
            copy_width = min(width, tile_size)
            
            result[:, :copy_height, :copy_width] = data[:, :copy_height, :copy_width]
            
        else:  # Single band
            height, width = data.shape
            result = np.zeros((tile_size, tile_size), dtype=data.dtype)
            
            # Calculate how much to copy
            copy_height = min(height, tile_size)
            copy_width = min(width, tile_size)
            
            result[:copy_height, :copy_width] = data[:copy_height, :copy_width]
        
        return result

    def image_scaler_tile(self, tile_data):
        """Apply image scaling to a single tile"""
        if len(tile_data.shape) == 2:
            tile_data = tile_data[np.newaxis, :, :]
        
        dra_img = []
        for band in range(tile_data.shape[0]):
            arr = tile_data[band]
            arr1 = arr.copy()
            
            if np.any(arr1 > 0):
                nonzero_vals = arr[arr > 0]
                if nonzero_vals.size > 0:
                    thr1 = round(np.percentile(nonzero_vals, 2.5))
                    thr2 = round(np.percentile(nonzero_vals, 99))
                    arr1[arr1 < thr1] = thr1
                    arr1[arr1 > thr2] = thr2
                    if (thr2 - thr1) > 0:
                        arr1 = (arr1 - thr1) / (thr2 - thr1)
                        arr1 = arr1 * 255.0
                        arr1 = np.uint8(arr1)
                        arr1[arr1 == 0] = 1
                    else:
                        arr1 = np.zeros_like(arr1, dtype=np.uint8)
                else:
                    arr1 = np.zeros_like(arr1, dtype=np.uint8)
            else:
                arr1 = np.zeros_like(arr1, dtype=np.uint8)
            
            dra_img.append(arr1)
        
        return np.stack(dra_img)

    def preprocess_tile(self, tile):
        """Preprocess a single tile for prediction"""
        # Convert to float32 and normalize to [0,1]
        tile = tile.astype(np.float32) / 255.0
        
        # Apply histogram equalization per channel
        for i in range(tile.shape[-1]):
            band = tile[:, :, i]
            if np.max(band) > np.min(band):
                tile[:, :, i] = exposure.equalize_adapthist(band)
        
        return tile

    def predict_single_tile(self, tile, model_name):
        """Predict a single tile with a specific model"""
        model = self.models[model_name]
        
        # Add batch dimension and predict
        tile_batch = np.expand_dims(tile, axis=0)
        prediction = model.predict(tile_batch, verbose=0)
        
        # Remove batch dimension
        return prediction[0]

    def save_tile_prediction(self, prediction, tile_id, model_name):
        """Save tile prediction to temporary file"""
        filename = f"tile_{tile_id}_{model_name}.npy"
        filepath = os.path.join(self.temp_dir, filename)
        np.save(filepath, prediction)
        return filepath

    def load_tile_prediction(self, tile_id, model_name):
        """Load tile prediction from temporary file"""
        filename = f"tile_{tile_id}_{model_name}.npy"
        filepath = os.path.join(self.temp_dir, filename)
        return np.load(filepath)

    def process_image_tiles(self, input_image_path):
        """Process all tiles from input image and save predictions"""
        with rasterio.open(input_image_path) as src:
            height, width = src.height, src.width
            transform = src.transform
            crs = src.crs
            
            # Calculate zoom factor
            src_resolution = src.res
            zoom_factor = self.calculate_zoom_factor(src_resolution, self.target_resolution)

            process_tile_size = int(self.tile_size / zoom_factor) + 100 
            
            print(f"Original image size: {width}x{height}")
            
            # Step 1: Calculate tile grid with overlap
            overlap_percentage = self.config['data']['overlap_percentage']
            tile_positions = self.calculate_tile_grid_with_overlap(
                width, height, process_tile_size, overlap_percentage
            )
            
            print(f"Processing {len(tile_positions)} tiles with overlap")
            
            # Process each tile
            tile_metadata = []
            
            for tile_id, (row_start, col_start, actual_height, actual_width, i, j) in enumerate(
                tqdm(tile_positions, desc="Processing tiles")
            ):
                try:
                    # Step 2: Create window for this tile
                    window = Window(col_start, row_start, actual_width, actual_height)
                    
                    # Step 3: Process tile with zoom/resampling
                    resampled_tile, tile_transform, new_height, new_width = self.process_tile_with_zoom(
                        src, window, zoom_factor, self.target_resolution
                    )
                    
                    # Skip empty tiles
                    if np.mean(resampled_tile) < 0.01:
                        continue
                    
                    # Step 4: Apply image scaling (enhancement)
                    scaled_tile = self.image_scaler_tile(resampled_tile)
                    
                    # Step 5: Pad to target size and convert to HWC format
                    padded_tile = self.pad_or_crop_to_tile_size(scaled_tile, self.tile_size)
                    tile_hwc = np.moveaxis(padded_tile, 0, -1)  # CHW to HWC
                    
                    # Step 6: Preprocess for prediction
                    processed_tile = self.preprocess_tile(tile_hwc)
                    
                    # Step 7: Predict with each model and save results
                    for model_name in self.models.keys():
                        prediction = self.predict_single_tile(processed_tile, model_name)
                        self.save_tile_prediction(prediction, tile_id, model_name)
                    
                    # Store tile metadata with original positions for merging
                    tile_metadata.append({
                        'tile_id': tile_id,
                        'original_row': row_start,
                        'original_col': col_start,
                        'original_height': actual_height,
                        'original_width': actual_width,
                        'resampled_height': new_height,
                        'resampled_width': new_width,
                        'grid_i': i,
                        'grid_j': j,
                        'transform': tile_transform,
                        'zoom_factor': zoom_factor
                    })
                    
                    # Clean up tile data from memory
                    del resampled_tile, scaled_tile, padded_tile, processed_tile
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing tile {tile_id} at ({i}, {j}): {str(e)}")
                    continue
            
            return tile_metadata, (height, width), transform, crs

    def merge_predictions_memory_efficient(self, tile_metadata, original_shape, original_transform, crs, output_path):
        """Merge tile predictions into final output without loading all into memory"""
        height, width = original_shape
        
        # Calculate output dimensions based on zoom factor
        if tile_metadata:
            zoom_factor = tile_metadata[0]['zoom_factor']
            out_height = int(height * zoom_factor)
            out_width = int(width * zoom_factor)
            out_transform = rasterio.Affine(
                self.target_resolution, 0, original_transform.c,
                0, -self.target_resolution, original_transform.f
            )
        else:
            out_height, out_width = height, width
            out_transform = original_transform
            zoom_factor = 1.0
        
        print(f"Output dimensions: {out_width}x{out_height} (zoom factor: {zoom_factor:.4f})")
        
        # Process each model separately
        for model_name in self.models.keys():
            print(f"Merging predictions for {model_name}...")
            
            # Create output file
            individual_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(self.input_image))[0]}_{model_name}_prediction_prob.tif"
            )
            
            with rasterio.open(
                individual_output, 'w',
                driver='GTiff',
                height=out_height,
                width=out_width,
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=out_transform,
                compress='lzw'
            ) as dst:
                
                # Initialize arrays for weighted average
                chunk_size = 1024  # Process in chunks to save memory
                
                for chunk_row in range(0, out_height, chunk_size):
                    for chunk_col in range(0, out_width, chunk_size):
                        chunk_h = min(chunk_size, out_height - chunk_row)
                        chunk_w = min(chunk_size, out_width - chunk_col)
                        
                        merged_chunk = np.zeros((chunk_h, chunk_w), dtype=np.float32)
                        weights_chunk = np.zeros((chunk_h, chunk_w), dtype=np.float32)
                        
                        # Process all tiles that intersect with this chunk
                        for tile_info in tile_metadata:
                            tile_id = tile_info['tile_id']
                            
                            # Load tile prediction
                            try:
                                tile_pred = self.load_tile_prediction(tile_id, model_name)
                                tile_pred = tile_pred[:, :, 0]  # Remove channel dimension
                                
                                # Calculate tile position in output space
                                tile_row = int(tile_info['original_row'] * zoom_factor)
                                tile_col = int(tile_info['original_col'] * zoom_factor)
                                
                                # Check if tile intersects with current chunk
                                if (tile_row < chunk_row + chunk_h and tile_row + self.tile_size > chunk_row and
                                    tile_col < chunk_col + chunk_w and tile_col + self.tile_size > chunk_col):
                                    
                                    # Calculate intersection
                                    start_row = max(0, tile_row - chunk_row)
                                    end_row = min(chunk_h, tile_row + self.tile_size - chunk_row)
                                    start_col = max(0, tile_col - chunk_col)
                                    end_col = min(chunk_w, tile_col + self.tile_size - chunk_col)
                                    
                                    tile_start_row = max(0, chunk_row - tile_row)
                                    tile_start_col = max(0, chunk_col - tile_col)
                                    tile_end_row = tile_start_row + (end_row - start_row)
                                    tile_end_col = tile_start_col + (end_col - start_col)
                                    
                                    # Create weight matrix (higher weights in center, lower at edges)
                                    tile_weight = np.ones((self.tile_size, self.tile_size))
                                    if self.overlap > 0:
                                        for i in range(self.overlap):
                                            w = i / self.overlap
                                            tile_weight[i, :] *= w
                                            tile_weight[-(i+1), :] *= w
                                            tile_weight[:, i] *= w
                                            tile_weight[:, -(i+1)] *= w
                                    
                                    # Add weighted prediction to chunk
                                    merged_chunk[start_row:end_row, start_col:end_col] += (
                                        tile_pred[tile_start_row:tile_end_row, tile_start_col:tile_end_col] *
                                        tile_weight[tile_start_row:tile_end_row, tile_start_col:tile_end_col]
                                    )
                                    weights_chunk[start_row:end_row, start_col:end_col] += (
                                        tile_weight[tile_start_row:tile_end_row, tile_start_col:tile_end_col]
                                    )
                                
                            except Exception as e:
                                print(f"Error loading tile {tile_id} for {model_name}: {str(e)}")
                                continue
                        
                        # Normalize by weights
                        merged_chunk = np.divide(merged_chunk, weights_chunk, 
                                               where=weights_chunk > 0, out=np.zeros_like(merged_chunk))
                        
                        # Write chunk to file
                        dst.write(merged_chunk, 1, window=Window(chunk_col, chunk_row, chunk_w, chunk_h))
                        
                        del merged_chunk, weights_chunk
                        gc.collect()
            
            print(f"Saved {model_name} prediction to {individual_output}")
            
            # Create binary mask
            final_prediction_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(self.input_image))[0]}_{model_name}_prediction.tif"
            )
            with rasterio.open(individual_output) as src:
                with rasterio.open(
                    final_prediction_output, 'w',
                    driver='GTiff',
                    height=out_height,
                    width=out_width,
                    count=1,
                    dtype=np.uint8,
                    crs=crs,
                    transform=out_transform,
                    compress='lzw'
                ) as dst:
                    # Process in chunks
                    for chunk_row in range(0, out_height, chunk_size):
                        for chunk_col in range(0, out_width, chunk_size):
                            chunk_h = min(chunk_size, out_height - chunk_row)
                            chunk_w = min(chunk_size, out_width - chunk_col)
                            
                            window = Window(chunk_col, chunk_row, chunk_w, chunk_h)
                            chunk_data = src.read(1, window=window)
                            binary_chunk = (chunk_data > self.thresholds[model_name]).astype(np.uint8)
                            dst.write(binary_chunk, 1, window=window)

    def predict(self, input_image_path, output_path):
        """Main prediction function"""
        print("Starting memory-efficient prediction with proper tiling and resampling...")
        
        # Process all tiles and save predictions
        tile_metadata, original_shape, original_transform, crs = self.process_image_tiles(input_image_path)
        
        if not tile_metadata:
            raise ValueError("No valid tiles were processed!")
        
        print(f"Processed {len(tile_metadata)} tiles successfully")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Merge predictions
        self.merge_predictions_memory_efficient(tile_metadata, original_shape, original_transform, crs, output_path)
        
        print("Prediction completed successfully!")
        
        # Clean up temporary files
        shutil.rmtree(self.temp_dir)
        print("Temporary files cleaned up")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory-efficient ensemble prediction with proper tiling')
    parser.add_argument('--input_image', required=True, help='Path to input image')
    parser.add_argument('--output_path', required=True, help='Path for output files')
    parser.add_argument('--building_model', required=False, help='Path to building model')
    parser.add_argument('--vegetation_model', required=False, help='Path to vegetation model')  
    parser.add_argument('--water_model', required=False, help='Path to water model')
    parser.add_argument('--config', default='config/config_v1.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MemoryEfficientPredictor(config_path=args.config, input_image=args.input_image)
    
    # Load models
    model_paths = {}
    if args.building_model:
        model_paths['building'] = (args.building_model, 0.39)
    if args.vegetation_model:
        model_paths['vegetation'] = (args.vegetation_model, 0.28)
    if args.water_model:
        model_paths['water'] = (args.water_model, 0.9)
    
    if not model_paths:
        print("Error: No model paths provided!")
        return
    
    predictor.load_models(model_paths)
    
    # Run prediction
    predictor.predict(args.input_image, args.output_path)


if __name__ == "__main__":
    main()