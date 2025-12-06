import sys
from pathlib import Path
import os
from skimage import io
from osgeo import gdal
from scipy import ndimage
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
from model import create_unet_model, create_old_unet_model
import yaml
from skimage import exposure
import gc
import tempfile
import shutil


class EnsemblePredictor:
    """Memory-efficient predictor class for large TIFF files using streaming merge"""
    
    def __init__(self, config_path='config.yaml', class_priority=None, chunk_size=2048):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tile_size = self.config['data']['input_size']
        self.overlap = 40
        self.channels = self.config['data']['channels']
        self.models = {}
        self.thresholds = {}
        self.chunk_size = chunk_size
        
        # Class mapping
        self.CLASS_BACKGROUND = 0
        self.CLASS_BUILDING = 1
        self.CLASS_VEGETATION = 2
        self.CLASS_WATER = 3
        
        # Define class priority (highest to lowest)
        self.class_priority = class_priority if class_priority is not None else [
            self.CLASS_BUILDING,
            self.CLASS_VEGETATION,
            self.CLASS_WATER
        ]
        self.max_batch_size = 4  # Very small batch size for memory efficiency
        
        # Create temporary directory for intermediate results
        self.temp_dir = tempfile.mkdtemp(prefix="predictor_")
        print(f"Using temporary directory: {self.temp_dir}")
        
        # Validate class priority
        valid_classes = {self.CLASS_BUILDING, self.CLASS_VEGETATION , self.CLASS_WATER}
        if not set(self.class_priority) == valid_classes:
            raise ValueError("Class priority must contain exactly the classes: Building(1), Vegetation(2), and Water(3)")

    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def load_models(self, model_paths):
        """Load multiple models with their respective thresholds"""
        for name, (model_path, threshold) in model_paths.items():
            print(f"Loading {name} model from {model_path}...")
            model = create_unet_model(self.config)
            
            if name == 'building':
                model.load_weights(model_path)
            self.models[name] = model
            self.thresholds[name] = threshold

    def minMax(self, band):
        band = np.float32(band)       
        band = (band - band.min()) / (band.max() - band.min())  
        return band

    def normalize_channel_veg(self, channel):
        """Advanced channel normalization with adaptive histogram equalization"""
        try:
            channel = channel.astype(np.float32)
            
            if np.all(channel == 0):
                return channel
            
            try:
                min_val = np.percentile(channel[channel > 0], 1)
                max_val = np.percentile(channel[channel > 0], 99)
            except Exception:
                min_val = np.min(channel)
                max_val = np.max(channel)
            
            channel = np.clip(channel, min_val, max_val)
            channel = (channel - min_val) / (max_val - min_val)
            channel = exposure.equalize_adapthist(channel)
            channel = np.clip(channel, 0, 1)
            
            return channel
            
        except ImportError:
            print("scikit-image not available. Falling back to basic normalization.")
            if np.all(channel == 0):
                return channel
            
            channel = channel.astype(np.float32)
            channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
            return channel
        
        except Exception as e:
            print(f"Normalization error: {str(e)}")
            return np.zeros_like(channel)

    def preprocess_sobel_image(self, image):
        """Preprocess image using channel normalization"""
        processed = np.zeros_like(image, dtype=np.float32)
        
        for j in range(self.channels):
            channel = image[:, :, j]
            processed[:, :, j] = self.normalize_channel_veg(channel)
        
        return processed

    def preprocess_rgb_image(self, img):
        """Preprocess RGB input image"""
        img = img.astype(np.float32)
        for i in range(img.shape[-1]):
            band = img[:, :, i]
            min_val = np.min(band)
            max_val = np.max(band)
            denom = max_val - min_val
            if denom < 1e-6:
                band = np.zeros_like(band)
            else:
                band = (band - min_val) / denom
            img[:, :, i] = exposure.equalize_adapthist(band)
        return img

    def normalize_tile(self, tile):
        """Normalize tile with percentile-based normalization"""
        tile = np.transpose(tile, (2, 0, 1)) 
        rd_ras = tile.astype(np.float32)
        dra_img = []

        for band in range(rd_ras.shape[0]):
            arr = rd_ras[band]
            arr1 = arr.copy()
            nonzero_vals = arr[arr > 0]

            if nonzero_vals.size == 0:
                arr1 = np.zeros_like(arr)
            else:
                thr1 = round(np.percentile(nonzero_vals, 2.5))
                thr2 = round(np.percentile(nonzero_vals, 99))
                arr1[arr1 < thr1] = thr1
                arr1[arr1 > thr2] = thr2
                denom = thr2 - thr1
                if denom == 0:
                    arr1 = np.zeros_like(arr1)
                else:
                    arr1 = (arr1 - thr1) / denom
                    arr1 = arr1 * 255.0

            arr1 = np.nan_to_num(arr1, nan=0.0)
            arr1 = np.uint8(arr1)
            arr1[arr1 == 0] = 1
            dra_img.append(arr1)

        foo = np.stack(dra_img, axis=-1)
        foo = foo.astype(np.float32) / 255.0
        return foo

    def create_chunk_windows(self, width, height):
        """Create windows for processing image in chunks"""
        windows = []
        
        for row_start in range(0, height, self.chunk_size):
            row_end = min(row_start + self.chunk_size, height)
            for col_start in range(0, width, self.chunk_size):
                col_end = min(col_start + self.chunk_size, width)
                
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                windows.append(window)
        
        return windows

    def create_tiles_from_chunk(self, chunk):
        """Create tiles from a chunk with overlap"""
        h, w, c = chunk.shape
        tiles = []
        positions = []
        
        stride = self.tile_size - self.overlap
        n_h = int(np.ceil((h - self.overlap) / stride))
        n_w = int(np.ceil((w - self.overlap) / stride))
        
        pad_h = stride * n_h + self.overlap - h
        pad_w = stride * n_w + self.overlap - w
        
        if pad_h > 0 or pad_w > 0:
            chunk = np.pad(chunk, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        for i in range(n_h):
            for j in range(n_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + self.tile_size
                w_end = w_start + self.tile_size
                
                tile = chunk[h_start:h_end, w_start:w_end, :]
                position = (h_start, h_end, w_start, w_end)
                
                # Filter out problematic tiles
                if self.is_tile_valid(tile):
                    tiles.append(tile)
                    positions.append(position)
        
        return tiles, positions, (n_h, n_w)

    def is_tile_valid(self, tile):
        """Check if tile is valid for processing"""
        try:
            if tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size:
                return False
            
            for channel in range(tile.shape[-1]):
                channel_data = tile[:, :, channel]
                min_val = np.percentile(channel_data, 2)
                max_val = np.percentile(channel_data, 98)
                
                if min_val >= max_val or np.isnan(min_val) or np.isnan(max_val):
                    return False
                    
            return True
        except Exception:
            return False

    def apply_augmentation(self, tile):
        """Apply test-time augmentation to a tile"""
        augmented_tiles = []

        # Original tile
        augmented_tiles.append(tile)  # 0: original

        # Flips
        augmented_tiles.append(np.fliplr(tile))  # 1: horizontal flip
        augmented_tiles.append(np.flipud(tile))  # 2: vertical flip

        # Rotations
        augmented_tiles.append(np.rot90(tile, k=1))  # 3: 90 degrees
        augmented_tiles.append(np.rot90(tile, k=2))  # 4: 180 degrees
        augmented_tiles.append(np.rot90(tile, k=3))  # 5: 270 degrees

        # Transposes
        augmented_tiles.append(np.transpose(tile, (1, 0, 2)))  # 6: basic transpose
        augmented_tiles.append(np.transpose(np.rot90(tile, k=1), (1, 0, 2)))  # 7: transpose + 90
        augmented_tiles.append(np.transpose(np.rot90(tile, k=2), (1, 0, 2)))  # 8: transpose + 180
        augmented_tiles.append(np.transpose(np.rot90(tile, k=3), (1, 0, 2)))  # 9: transpose + 270

        return np.array(augmented_tiles)

    def merge_augmented_predictions(self, predictions):
        """Merge predictions from augmented tiles"""
        # Reverse the augmentations
        pred_original = predictions[0]
        pred_fliplr = np.fliplr(predictions[1])
        pred_flipud = np.flipud(predictions[2])
        pred_rot90_1 = np.rot90(predictions[3], k=-1)
        pred_rot90_2 = np.rot90(predictions[4], k=-2)
        pred_rot90_3 = np.rot90(predictions[5], k=-3)

        # Reverse transpose operations
        pred_transpose = np.transpose(predictions[6], (1, 0, 2))
        pred_transpose_rot90_1 = np.rot90(np.transpose(predictions[7], (1, 0, 2)), k=-1)
        pred_transpose_rot90_2 = np.rot90(np.transpose(predictions[8], (1, 0, 2)), k=-2)
        pred_transpose_rot90_3 = np.rot90(np.transpose(predictions[9], (1, 0, 2)), k=-3)

        # Average all predictions
        merged = np.mean([
            pred_original,
            pred_fliplr,
            pred_flipud,
            pred_rot90_1,
            pred_rot90_2,
            pred_rot90_3,
            pred_transpose,
            pred_transpose_rot90_1,
            pred_transpose_rot90_2,
            pred_transpose_rot90_3
        ], axis=0)

        return merged

    def merge_chunk_predictions(self, predictions, positions, chunk_shape):
        """Merge predictions within a chunk"""
        h, w = chunk_shape
        merged = np.zeros((h, w, 1), dtype=np.float32)
        counts = np.zeros((h, w, 1), dtype=np.float32)
        
        weight = np.ones((self.tile_size, self.tile_size, 1), dtype=np.float32)
        if self.overlap > 0:
            for i in range(self.overlap):
                weight[i, :, :] *= (i / self.overlap)
                weight[-(i+1), :, :] *= (i / self.overlap)
                weight[:, i, :] *= (i / self.overlap)
                weight[:, -(i+1), :] *= (i / self.overlap)
        
        for pred, (h_start, h_end, w_start, w_end) in zip(predictions, positions):
            h_end = min(h_end, h)
            w_end = min(w_end, w)
            h_size = h_end - h_start
            w_size = w_end - w_start
            
            merged[h_start:h_end, w_start:w_end, :] += (
                pred[:h_size, :w_size, :] * weight[:h_size, :w_size, :]
            )
            counts[h_start:h_end, w_start:w_end, :] += weight[:h_size, :w_size, :]
        
        # Avoid division by zero
        merged = np.divide(merged, counts, out=np.zeros_like(merged), where=counts > 0)
        return merged

    def process_chunk_streaming(self, chunk, model_name, model, output_file, window):
        """Process a chunk and directly write to output file - streaming approach"""
        try:
            # Create tiles from chunk
            tiles, positions, (n_h, n_w) = self.create_tiles_from_chunk(chunk)
            
            if not tiles:
                return False
            
            # Process tiles in very small batches
            predictions = []
            
            for tile in tiles:
                try:
                    # Preprocess tile based on model type
                    if model_name == "building":
                        rgb_tile = self.normalize_tile(tile)
                        processed_tile = self.preprocess_rgb_image(rgb_tile)
                    else:
                        processed_tile = self.preprocess_sobel_image(tile)
                    
                    # Apply augmentations
                    augmented_tiles = self.apply_augmentation(processed_tile)
                    aug_predictions = []
                    
                    # Process augmentations one by one to minimize memory usage
                    for aug_tile in augmented_tiles:
                        pred = model.predict(np.expand_dims(aug_tile, axis=0), verbose=0)
                        aug_predictions.append(pred[0])
                        del pred  # Immediate cleanup
                    
                    # Merge augmented predictions
                    merged_pred = self.merge_augmented_predictions(aug_predictions)
                    predictions.append(merged_pred)
                    
                    # Immediate cleanup
                    del augmented_tiles, aug_predictions, processed_tile, merged_pred
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing tile: {str(e)}")
                    continue
            
            if not predictions:
                return False
            
            # Merge predictions for this chunk
            chunk_h, chunk_w = chunk.shape[:2]
            merged_chunk = self.merge_chunk_predictions(predictions, positions, (chunk_h, chunk_w))
            
            # Write result directly to output file
            with rasterio.open(output_file, 'r+') as dst:
                dst.write(merged_chunk[:, :, 0], 1, window=window)
            
            # Cleanup
            del predictions, tiles, positions, merged_chunk
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error in process_chunk_streaming: {str(e)}")
            return False

    def resolve_conflicts_new(self, all_predictions):
        """Resolve predictions using only the building model"""
        building_conf = all_predictions['building']
        building_mask = building_conf > self.thresholds['building']

        # Initialize final mask as background (0)
        final_mask = np.zeros_like(building_mask, dtype=np.uint8)

        # Assign building class where mask is True
        final_mask[building_mask] = self.CLASS_BUILDING

        return final_mask

    def predict(self, image_path, output_path):
        """Run ensemble prediction with streaming merge approach"""
        print(f"Starting prediction for: {image_path}")
        
        # Open image to get metadata
        with rasterio.open(image_path) as src:
            transform = src.transform
            crs = src.crs
            if crs is None:
                crs = rasterio.crs.CRS.from_epsg(4326)
            height, width = src.height, src.width
            channels = min(src.count, 3)  # Use only first 3 channels
            
            print(f"Image dimensions: {width} x {height}")
            print(f"Using channels: {channels}")
            
            # Create chunk windows
            windows = self.create_chunk_windows(width, height)
            print(f"Processing {len(windows)} chunks")
        
        # Process each model separately with streaming approach
        for model_name, model in self.models.items():
            if model_name != 'building':
                continue  # Skip non-building models
                
            print(f"\nProcessing with {model_name} model...")
            
            # Create output file for this model
            model_output = os.path.join(
                os.path.dirname(output_path),
                f"{os.path.splitext(os.path.basename(output_path))[0]}_{model_name}_prob.tif"
            )
            
            # Initialize output file
            with rasterio.open(
                model_output,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=np.float32,
                crs=crs,
                transform=transform,
                compress='lzw',
                tiled=True,
                blockxsize=512,
                blockysize=512
            ) as dst:
                # Initialize with zeros
                dst.write(np.zeros((height, width), dtype=np.float32), 1)
            
            # Process chunks and write directly to output file
            successful_chunks = 0
            with rasterio.open(image_path) as src:
                for chunk_idx, window in enumerate(tqdm(windows, desc=f"Processing {model_name} chunks")):
                    try:
                        # Read chunk from source
                        chunk = src.read([1, 2, 3], window=window)  # Read RGB channels
                        chunk = np.moveaxis(chunk, 0, -1)  # Convert to HWC format
                        
                        # Process chunk with streaming approach
                        success = self.process_chunk_streaming(chunk, model_name, model, model_output, window)
                        
                        if success:
                            successful_chunks += 1
                        
                        # Clear chunk from memory
                        del chunk
                        gc.collect()
                        
                        # Periodic cleanup
                        if chunk_idx % 100 == 0:
                            gc.collect()
                        
                    except Exception as e:
                        print(f"Error processing chunk {chunk_idx}: {str(e)}")
                        continue
            
            print(f"Successfully processed {successful_chunks}/{len(windows)} chunks for {model_name}")
            
            # Create binary mask from probability map
            binary_output = os.path.join(
                os.path.dirname(output_path),
                f"{os.path.splitext(os.path.basename(output_path))[0]}_{model_name}.tif"
            )
            
            # Process probability map to binary in chunks to avoid memory issues
            with rasterio.open(model_output) as prob_src, \
                 rasterio.open(
                     binary_output,
                     'w',
                     driver='GTiff',
                     height=height,
                     width=width,
                     count=1,
                     dtype=np.uint8,
                     crs=crs,
                     transform=transform,
                     compress='lzw',
                     tiled=True,
                     blockxsize=512,
                     blockysize=512
                 ) as binary_dst:
                
                # Process in chunks to create binary mask
                chunk_windows = self.create_chunk_windows(width, height)
                for window in tqdm(chunk_windows, desc=f"Creating binary mask for {model_name}"):
                    prob_chunk = prob_src.read(1, window=window)
                    binary_chunk = (prob_chunk > self.thresholds[model_name]).astype(np.uint8)
                    binary_dst.write(binary_chunk, 1, window=window)
                    del prob_chunk, binary_chunk
            
            print(f"Saved {model_name} prediction to {binary_output}")
            print(f"Saved {model_name} probability map to {model_output}")
        
        # Create final multiclass output by reading binary outputs in chunks
        print("\nCreating final multiclass prediction...")
        final_output = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.splitext(os.path.basename(output_path))[0]}_multiclass.tif"
        )
        
        # For now, just copy the building prediction as multiclass since we only have one model
        building_binary = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.splitext(os.path.basename(output_path))[0]}_building.tif"
        )
        
        if os.path.exists(building_binary):
            with rasterio.open(building_binary) as src, \
                 rasterio.open(
                     final_output,
                     'w',
                     driver='GTiff',
                     height=height,
                     width=width,
                     count=1,
                     dtype=np.uint8,
                     crs=crs,
                     transform=transform,
                     compress='lzw',
                     tiled=True,
                     blockxsize=512,
                     blockysize=512
                 ) as dst:
                
                # Copy in chunks
                chunk_windows = self.create_chunk_windows(width, height)
                total_pixels = 0
                building_pixels = 0
                
                for window in tqdm(chunk_windows, desc="Creating multiclass output"):
                    chunk = src.read(1, window=window)
                    dst.write(chunk, 1, window=window)
                    
                    # Calculate stats
                    total_pixels += chunk.size
                    building_pixels += np.sum(chunk == 1)
                    
                    del chunk
                
                dst.update_tags(
                    CLASS_VALUES="0,1",
                    CLASS_NAMES="Background,Building"
                )
        
        print(f"Saved final prediction to {final_output}")
        
        # Calculate statistics
        if total_pixels > 0:
            stats = {
                'background_percentage': ((total_pixels - building_pixels) / total_pixels) * 100,
                'building_percentage': (building_pixels / total_pixels) * 100,
                'vegetation_percentage': 0.0,
                'water_percentage': 0.0
            }
        else:
            stats = {
                'background_percentage': 100.0,
                'building_percentage': 0.0,
                'vegetation_percentage': 0.0,
                'water_percentage': 0.0
            }
        
        print("\nClass Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}%")
        
        # Cleanup temporary files
        print(f"\nCleaning up temporary files from {self.temp_dir}")
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize predictor with smaller chunk size for very large files
    predictor = EnsemblePredictor(chunk_size=1024)  # Process 1024x1024 chunks
    
    # Load models
    model_paths = {
        'building': ('path/to/building_model.h5', 0.5),
        # Add other models as needed
    }
    predictor.load_models(model_paths)
    
    # Run prediction
    input_image = "path/to/large_image.tif"
    output_path = "path/to/output.tif"
    
    stats = predictor.predict(input_image, output_path)
    print("Prediction completed successfully!")