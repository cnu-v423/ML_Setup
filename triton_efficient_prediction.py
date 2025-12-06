import os
import tempfile
import shutil
import numpy as np
import rasterio
from rasterio.windows import Window
import yaml
import gc
from pathlib import Path
from skimage import exposure
from tqdm import tqdm
import tritonclient.http as triton_client
# import torch


class TritonMemoryEfficientPredictor:
    """Memory-efficient predictor using Triton server for large images"""
    
    def __init__(self, config_path='config/config_v1.yaml', input_image='', triton_url='localhost:8000'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tile_size = self.config['data']['input_size']
        self.channels = self.config['data']['channels']
        overlap_percentage = self.config['data']['overlap_percentage']
        self.overlap = int(self.tile_size * overlap_percentage)
        self.input_image = input_image
        self.models = {}
        self.thresholds = {}
        self.max_batch_size = self.config.get('data', {}).get('batch_size', 8)
        
        # Initialize Triton client
        self.client = triton_client.InferenceServerClient(url=triton_url)
        
        # Class definitions
        self.CLASS_BACKGROUND = 0
        self.CLASS_BUILDING = 1
        self.CLASS_VEGETATION = 2
        self.CLASS_WATER = 3
        
        # Create temporary directory for storing tile predictions
        self.temp_dir = tempfile.mkdtemp(prefix='prediction_tiles_')
        print(f"Using temporary directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def set_gpu(self, device_id: int = 0):
        """
        Set the GPU device for Triton inference.
        If GPU is not available, it falls back to CPU.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        # if torch.cuda.is_available():
        #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        #     print(f"Using GPU: {torch.cuda.get_device_name(device_id)}")
        # else:
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #     # print("No GPU found. Falling back to CPU mode.")

    def load_models(self, model_configs):
        """Load Triton model configurations with their thresholds"""
        for name, config in model_configs.items():
            model_name = config.get('model_name')
            input_name = config.get('input_name')
            output_name = config.get('output_name')
            threshold = config.get('threshold')
            
            if model_name:
                print(f"Configuring {name} model: {model_name}")
                self.models[name] = {
                    'model_name': model_name,
                    'input_name': input_name,
                    'output_name': output_name
                }
                self.thresholds[name] = threshold

    def calculate_tile_grid_with_overlap(self, width, height, tile_size, overlap_percentage):
        """Calculate tile positions with overlap"""
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

    def pad_or_crop_to_tile_size(self, data, tile_size):
        """Pad or crop data to exact tile size"""
        if len(data.shape) == 3:  # Multi-band (CHW format)
            bands, height, width = data.shape
            result = np.zeros((bands, tile_size, tile_size), dtype=data.dtype)
            
            copy_height = min(height, tile_size)
            copy_width = min(width, tile_size)
            
            result[:, :copy_height, :copy_width] = data[:, :copy_height, :copy_width]
            
        else:  # Single band
            height, width = data.shape
            result = np.zeros((tile_size, tile_size), dtype=data.dtype)
            
            copy_height = min(height, tile_size)
            copy_width = min(width, tile_size)
            
            result[:copy_height, :copy_width] = data[:copy_height, :copy_width]
        
        return result

    def normalize_tile(self, tile):
        """Normalize tile values to 0-1 range"""
        tile = tile.astype(np.float32)
        # Normalize each band independently
        for i in range(tile.shape[0]):
            band = tile[i]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                tile[i] = (band - band_min) / (band_max - band_min)
        return tile

    def apply_sobel_to_tile_waterbody(self, tile):
        """Apply Sobel filter for water body detection"""
        from skimage import filters
        
        # Convert to grayscale if needed
        if len(tile.shape) == 3:
            gray = np.mean(tile, axis=0)
        else:
            gray = tile
        
        # Apply Sobel filter
        sobel_h = filters.sobel_h(gray)
        sobel_v = filters.sobel_v(gray)
        sobel_combined = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Stack with original channels to maintain 3-channel format
        if len(tile.shape) == 3 and tile.shape[0] == 3:
            result = np.stack([tile[0], tile[1], sobel_combined])
        else:
            result = np.stack([gray, gray, sobel_combined])
        
        return result

    def preprocess_rgb_image(self, tile):
        """Preprocess RGB image tile"""
        # Convert CHW to HWC
        tile_hwc = np.moveaxis(tile, 0, -1)
        
        # Apply histogram equalization per channel
        for i in range(tile_hwc.shape[-1]):
            band = tile_hwc[:, :, i]
            if np.max(band) > np.min(band):
                tile_hwc[:, :, i] = exposure.equalize_adapthist(band, clip_limit=0.03)
        
        return tile_hwc

    def preprocess_sobel_image_old(self, tile):
        """Preprocess Sobel-filtered image tile"""
        # Convert CHW to HWC
        tile_hwc = np.moveaxis(tile, 0, -1)
        
        # Normalize to [0, 1]
        tile_hwc = tile_hwc.astype(np.float32)
        for i in range(tile_hwc.shape[-1]):
            band = tile_hwc[:, :, i]
            band_min, band_max = band.min(), band.max()
            if band_max > band_min:
                tile_hwc[:, :, i] = (band - band_min) / (band_max - band_min)
        
        return tile_hwc

    def apply_augmentation(self, tile):
        """Apply test-time augmentation"""
        augmentations = [tile]  # Original
        
        # Add horizontal flip
        augmentations.append(np.fliplr(tile))
        
        # Add vertical flip
        augmentations.append(np.flipud(tile))
        
        # Add both flips
        augmentations.append(np.flipud(np.fliplr(tile)))
        
        return np.array(augmentations)

    def merge_augmented_predictions(self, predictions):
        """Merge predictions from different augmentations"""
        if len(predictions) == 0:
            return None
        
        # Convert to numpy array
        preds = np.array(predictions)
        
        # Reverse augmentations and average
        original = preds[0]
        h_flip = np.fliplr(preds[1])
        v_flip = np.flipud(preds[2])
        hv_flip = np.flipud(np.fliplr(preds[3]))
        
        # Average all predictions
        merged = (original + h_flip + v_flip + hv_flip) / 4.0
        
        return merged

    def predict_tile_batch_triton(self, tiles, model_config):
        """Predict a batch of tiles using Triton server"""
        batch_size = len(tiles)
        
        # Prepare inputs
        inputs = []
        inputs.append(triton_client.InferInput(
            model_config["input_name"], 
            (batch_size, self.tile_size, self.tile_size, 3), 
            "FP32"
        ))
        inputs[0].set_data_from_numpy(np.array(tiles))

        outputs = []
        outputs.append(triton_client.InferRequestedOutput(model_config["output_name"]))

        # Perform inference
        response = self.client.infer(model_config["model_name"], inputs, outputs=outputs)
        batch_output = response.as_numpy(model_config["output_name"])
        
        return batch_output

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
            
            print(f"Original image size: {width}x{height}")
            
            # Calculate tile grid with overlap
            overlap_percentage = self.config['data']['overlap_percentage']
            tile_positions = self.calculate_tile_grid_with_overlap(
                width, height, self.tile_size, overlap_percentage
            )
            
            print(f"Processing {len(tile_positions)} tiles with overlap")
            
            # Process tiles in batches for each model
            tile_metadata = []
            
            for model_name, model_config in self.models.items():
                print(f"\nProcessing tiles for {model_name} model...")
                
                # Process tiles in batches to manage memory
                batch_size = min(self.max_batch_size, 8)  # Limit batch size for memory

                # Outer tqdm for all tile positions
                for batch_start in tqdm(range(0, len(tile_positions), batch_size), 
                            desc=f"{model_name} - Processing batches", 
                            total=(len(tile_positions) // batch_size) + 1):
                    # pass
                
                # for batch_start in range(0, len(tile_positions), batch_size):
                    batch_end = min(batch_start + batch_size, len(tile_positions))
                    batch_tiles = []
                    batch_tile_ids = []
                    
                    # Prepare batch of tiles
                    for i in range(batch_start, batch_end):
                        tile_id, (row_start, col_start, actual_height, actual_width, grid_i, grid_j) = i, tile_positions[i]
                        
                        try:
                            # Read tile
                            window = Window(col_start, row_start, actual_width, actual_height)
                            tile_data = src.read([1, 2, 3], window=window)  # Read RGB bands
                            
                            # Skip empty tiles
                            if np.mean(tile_data) < 0.01:
                                continue
                            
                            # Pad to tile size
                            padded_tile = self.pad_or_crop_to_tile_size(tile_data, self.tile_size)
                            
                            # Apply preprocessing based on model type
                            if model_name == "building":
                                normalized_tile = self.normalize_tile(padded_tile)
                                processed_tile = self.preprocess_rgb_image(normalized_tile)
                            elif model_name == "vegetation":
                                normalized_tile = self.normalize_tile(padded_tile)
                                processed_tile = self.preprocess_rgb_image(normalized_tile)
                            else:  # water model
                                sobel_tile = self.apply_sobel_to_tile_waterbody(padded_tile)
                                processed_tile = self.preprocess_sobel_image_old(sobel_tile)
                            
                            batch_tiles.append(processed_tile)
                            batch_tile_ids.append(tile_id)
                            
                            # Store metadata only once (for first model)
                            if model_name == list(self.models.keys())[0]:
                                tile_metadata.append({
                                    'tile_id': tile_id,
                                    'original_row': row_start,
                                    'original_col': col_start,
                                    'original_height': actual_height,
                                    'original_width': actual_width,
                                    'grid_i': grid_i,
                                    'grid_j': grid_j
                                })
                            
                        except Exception as e:
                            print(f"Error processing tile {tile_id}: {str(e)}")
                            continue
                    
                    # Process batch if we have tiles
                    if batch_tiles:
                        # Apply augmentations and predict
                        for idx, (tile, tile_id) in enumerate(zip(batch_tiles, batch_tile_ids)):
                            try:
                                # Apply augmentations
                                augmented_tiles = self.apply_augmentation(tile)
                                
                                # Process augmented tiles in sub-batches
                                aug_predictions = []
                                aug_batch_size = min(4, len(augmented_tiles))
                                
                                for aug_start in range(0, len(augmented_tiles), aug_batch_size):
                                    aug_end = min(aug_start + aug_batch_size, len(augmented_tiles))
                                    aug_batch = augmented_tiles[aug_start:aug_end]
                                    
                                    # Predict with Triton
                                    batch_preds = self.predict_tile_batch_triton(aug_batch, model_config)
                                    aug_predictions.extend(batch_preds)
                                
                                # Merge augmented predictions
                                merged_pred = self.merge_augmented_predictions(aug_predictions)
                                
                                # Save prediction
                                self.save_tile_prediction(merged_pred, tile_id, model_name)
                                
                                # Clean up
                                del augmented_tiles, aug_predictions, merged_pred
                                
                            except Exception as e:
                                print(f"Error predicting tile {tile_id} for {model_name}: {str(e)}")
                                continue
                        
                        # Clean up batch
                        del batch_tiles
                        gc.collect()
            
            return tile_metadata, (height, width), transform, crs

    def merge_predictions_memory_efficient(self, tile_metadata, original_shape, original_transform, crs, output_path):
        """Merge tile predictions into final output without loading all into memory"""
        height, width = original_shape
        
        print(f"Output dimensions: {width}x{height}")
        
        # Process each model separately
        for model_name in self.models.keys():
            print(f"Merging predictions for {model_name}...")
            
            # Create probability output file
            prob_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(self.input_image))[0]}_{model_name}_prob.tif"
            )
            
            with rasterio.open(
                prob_output, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=original_transform,
                compress='lzw'
            ) as dst:
                
                # Initialize arrays for weighted average
                chunk_size = self.tile_size  # Process in chunks to save memory
                total_chunks = ((height + chunk_size - 1) // chunk_size) * ((width + chunk_size - 1) // chunk_size)

                with tqdm(total=total_chunks, desc=f"Writing {model_name} chunks") as pbar:

                    for chunk_row in range(0, height, chunk_size):
                        for chunk_col in range(0, width, chunk_size):
                            chunk_h = min(chunk_size, height - chunk_row)
                            chunk_w = min(chunk_size, width - chunk_col)
                            
                            merged_chunk = np.zeros((chunk_h, chunk_w), dtype=np.float32)
                            weights_chunk = np.zeros((chunk_h, chunk_w), dtype=np.float32)
                            
                            # Process all tiles that intersect with this chunk
                            for tile_info in tile_metadata:
                                tile_id = tile_info['tile_id']
                                
                                try:
                                    tile_pred = self.load_tile_prediction(tile_id, model_name)
                                    if len(tile_pred.shape) == 3:
                                        tile_pred = tile_pred[:, :, 0]  # Remove channel dimension
                                    
                                    # Calculate tile position in output space
                                    tile_row = tile_info['original_row']
                                    tile_col = tile_info['original_col']
                                    
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
                                                w = (i + 1) / (self.overlap + 1)
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

                            pbar.update(1)

            
            print(f"Saved {model_name} probability map to {prob_output}")
            
            # Create binary mask
            binary_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(self.input_image))[0]}_{model_name}.tif"
            )
            
            with rasterio.open(prob_output) as src:
                with rasterio.open(
                    binary_output, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=np.uint8,
                    crs=crs,
                    transform=original_transform,
                    compress='lzw'
                ) as dst:
                    # Process in chunks
                    chunk_size = 1024
                    for chunk_row in range(0, height, chunk_size):
                        for chunk_col in range(0, width, chunk_size):
                            chunk_h = min(chunk_size, height - chunk_row)
                            chunk_w = min(chunk_size, width - chunk_col)
                            
                            window = Window(chunk_col, chunk_row, chunk_w, chunk_h)
                            chunk_data = src.read(1, window=window)
                            binary_chunk = (chunk_data > self.thresholds[model_name]).astype(np.uint8)
                            dst.write(binary_chunk, 1, window=window)
                            
                            dst.update_tags(
                                MODEL_NAME=model_name,
                                THRESHOLD=str(self.thresholds[model_name]),
                                CLASS_NAME=model_name.upper()
                            )
            
            print(f"Saved {model_name} binary mask to {binary_output}")

    def resolve_conflicts_new(self, all_predictions):
        """Resolve conflicts between different model predictions"""
        # Simple conflict resolution - highest confidence wins
        final_mask = np.zeros_like(list(all_predictions.values())[0], dtype=np.uint8)
        
        for class_id, (model_name, prediction) in enumerate(all_predictions.items(), 1):
            mask = prediction > self.thresholds[model_name]
            final_mask[mask] = class_id
            
        return final_mask

    def predict(self, input_image_path, output_path):
        """Main prediction function"""
        print("Starting Triton memory-efficient prediction...")
        
        # Process all tiles and save predictions
        tile_metadata, original_shape, original_transform, crs = self.process_image_tiles(input_image_path)
        
        if not tile_metadata:
            raise ValueError("No valid tiles were processed!")
        
        print(f"Processed {len(tile_metadata)} tiles successfully")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        # Merge predictions
        self.merge_predictions_memory_efficient(tile_metadata, original_shape, original_transform, crs, os.path.dirname(output_path))
        
        print("Prediction completed successfully!")
        
        # Clean up temporary files
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("Temporary files cleaned up")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Triton memory-efficient prediction')
    parser.add_argument('--input_image', required=True, help='Path to input image')
    parser.add_argument('--output_path', required=True, help='Path for output files')
    parser.add_argument('--config', default='config/config_v1.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TritonMemoryEfficientPredictor(
        config_path=args.config, 
        input_image=args.input_image,
        triton_url='localhost:8000'
    )

    predictor.set_gpu(0)
    
    # Configure Triton models
    model_configs = {
        'building': {
            'model_name': 'UNET_SENET154',
            'input_name': 'input',
            'output_name': 'sigmoid',
            'threshold': 0.343
        },
        # 'building': {
        #     'model_name': 'cultivation_512',
        #     'input_name': 'input',
        #     'output_name': 'output',
        #     'threshold': 0.343
        # },
        # 'vegetation': {
        #     'model_name': 'vegetation_model', 
        #     'input_name': 'input',
        #     'output_name': 'output',
        #     'threshold': 0.28
        # },
        # 'water': {
        #     'model_name': 'water_model',
        #     'input_name': 'input', 
        #     'output_name': 'output',
        #     'threshold': 0.9
        # }
    }
    
    predictor.load_models(model_configs)
    
    # Run prediction
    predictor.predict(args.input_image, args.output_path)



if __name__ == "__main__":
    
    main()