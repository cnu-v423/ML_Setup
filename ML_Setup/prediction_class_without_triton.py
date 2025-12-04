# prediction_class_without_triton.py
import sys
from pathlib import Path
import os
from skimage import io
from osgeo import gdal
from scipy import ndimage
import rasterio
import numpy as np
from tqdm import tqdm
from model import create_unet_model, create_old_unet_model
import yaml
from skimage import exposure
import gc
from rasterio.warp import reproject, Resampling
import rasterio.windows


class EnsemblePredictor:
    """Predictor class for ensemble prediction with multiple semantic segmentation models"""
    
    def __init__(self, config_path='config/config_v1.yaml', class_priority=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tile_size = self.config['data']['input_size']
        overlap_percentage = self.config['data']['overlap_percentage']
        self.overlap = int(self.tile_size * overlap_percentage)
        self.channels = self.config['data']['channels']
        self.target_resolution = self.config['data']['resolution']
        self.models = {}
        self.thresholds = {}
        
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
        self.max_batch_size = 16
        
        # Validate class priority
        valid_classes = {self.CLASS_BUILDING, self.CLASS_VEGETATION , self.CLASS_WATER}
        if not set(self.class_priority) == valid_classes:
            raise ValueError("Class priority must contain exactly the classes: Building(1), Vegetation(2), and Water(3)")

    def calculate_zoom_factor(self, src_resolution):
        """
        Calculate zoom factor based on source and target resolutions
        """
        if isinstance(src_resolution, tuple):
            src_res = min(abs(src_resolution[0]), abs(src_resolution[1]))  # Use the finer resolution
        else:
            src_res = abs(src_resolution)
        
        target_res = abs(self.target_resolution)
        zoom_factor = src_res / target_res
        
        print(f"Source resolution: {src_res:.4f}m, Target resolution: {target_res:.4f}m")
        print(f"Zoom factor: {zoom_factor:.4f}")
        
        return zoom_factor

    def resample_image_to_target_resolution(self, image, src_transform, src_crs):
        """
        Resample the entire image to target resolution before tiling
        """
        # Calculate zoom factor from transform
        src_resolution = (abs(src_transform.a), abs(src_transform.e))
        zoom_factor = self.calculate_zoom_factor(src_resolution)
        
        if abs(zoom_factor - 1.0) < 0.01:  # No significant change needed
            print("Source resolution matches target resolution, no resampling needed")
            return image, src_transform
        
        # Calculate new dimensions
        original_height, original_width = image.shape[:2]
        new_height = int(original_height * zoom_factor)
        new_width = int(original_width * zoom_factor)
        
        print(f"Resampling from {original_width}x{original_height} to {new_width}x{new_height}")
        
        # Create new transform
        new_transform = rasterio.Affine(
            self.target_resolution,  # new pixel width
            src_transform.b,
            src_transform.c,  # same x origin
            src_transform.d,
            -self.target_resolution,  # new pixel height (negative for north-up)
            src_transform.f   # same y origin
        )
        
        # Handle multi-channel images
        if len(image.shape) == 3:
            bands, height, width = image.shape[2], image.shape[0], image.shape[1]
            # Reshape to (bands, height, width) for resampling
            image_bands = np.moveaxis(image, -1, 0)
        else:
            bands = 1
            height, width = image.shape
            image_bands = image[np.newaxis, :, :]
        
        # Create output array
        resampled_image = np.zeros((bands, new_height, new_width), dtype=image.dtype)
        
        # Resample each band
        for band_idx in range(bands):
            reproject(
                source=image_bands[band_idx],
                destination=resampled_image[band_idx],
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=new_transform,
                dst_crs=src_crs,
                resampling=Resampling.bilinear
            )
        
        # Convert back to (height, width, channels) format
        if bands > 1:
            resampled_image = np.moveaxis(resampled_image, 0, -1)
        else:
            resampled_image = resampled_image[0]
        
        print(f"Resampling completed. New shape: {resampled_image.shape}")
        return resampled_image, new_transform

    def load_models(self, model_paths):
        """
        Load multiple models with their respective thresholds
        """
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
    
    def change_projection(self, inputfile, referencefile):
        print("Changing Projection...\n")
        file = inputfile
        outfile = inputfile
        ds = gdal.Open(file)
        arr = ds.ReadAsArray()
        
        if len(arr.shape) == 3:
            num = arr.shape[0]
            ds_ref = gdal.Open(referencefile)
            geotrans = ds_ref.GetGeoTransform()
            proj = ds_ref.GetProjection()
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(outfile, arr.shape[2], arr.shape[1], arr.shape[0], gdal.GDT_Float32)
            outdata.SetGeoTransform(geotrans)
            outdata.SetProjection(proj)
            
            for i in range(1, int(num) + 1):
                outdata.GetRasterBand(i).WriteArray(arr[i - 1, :, :])
                outdata.GetRasterBand(i).SetNoDataValue(10000)
            outdata.FlushCache()
            outdata = None
            ds = None
        elif len(arr.shape) == 2:
            [cols, rows] = arr.shape
            ds_ref = gdal.Open(referencefile)
            geotrans = ds_ref.GetGeoTransform()
            proj = ds_ref.GetProjection()
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
            outdata.SetGeoTransform(geotrans)
            outdata.SetProjection(proj)
            outdata.GetRasterBand(1).WriteArray(arr)
            outdata.GetRasterBand(1).SetNoDataValue(10000)
            outdata.FlushCache()
            outdata = None
            ds = None

    def normalize_channel_veg(self, channel):
        """
        Advanced channel normalization with adaptive histogram equalization.
        """
        try:
            from skimage import exposure
            
            # Convert to float32
            channel = channel.astype(np.float32)
            
            # Skip if channel is all zeros
            if np.all(channel == 0):
                return channel
            
            # Remove extreme outliers while preserving the overall distribution
            try:
                # Use percentiles to determine min and max
                min_val = np.percentile(channel[channel > 0], 1)
                max_val = np.percentile(channel[channel > 0], 99)
            except Exception:
                # Fallback to min and max if percentiles fail
                min_val = np.min(channel)
                max_val = np.max(channel)
            
            # Clip to remove extreme values
            channel = np.clip(channel, min_val, max_val)
            
            # Normalize to [0, 1] range
            if max_val - min_val > 0:
                channel = (channel - min_val) / (max_val - min_val)
            else:
                channel = np.zeros_like(channel)
            
            # Apply Adaptive Histogram Equalization (CLAHE)
            channel = exposure.equalize_adapthist(channel)
            
            # Additional clipping to ensure values are strictly between 0 and 1
            channel = np.clip(channel, 0, 1)
            
            return channel
            
        except ImportError:
            print("scikit-image not available. Falling back to basic normalization.")
            # Fallback normalization if scikit-image is not installed
            if np.all(channel == 0):
                return channel
            
            # Normalize to [0, 1] range
            channel = channel.astype(np.float32)
            if np.max(channel) - np.min(channel) > 0:
                channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
            return channel
        
        except Exception as e:
            print(f"Normalization error: {str(e)}")
            return np.zeros_like(channel)

    def preprocess_sobel_image(self, image):
        # Create an empty array with the same shape as the input image
        processed = np.zeros_like(image, dtype=np.float32)
        
        # Process each channel individually
        for j in range(self.channels):
            # Extract the channel
            channel = image[:, :, j]
            
            # Apply the normalize_channel_veg function
            processed[:, :, j] = self.normalize_channel_veg(channel)
        
        return processed

    def preprocess_rgb_image(self, img):
        """Preprocess input image"""
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

    def filter_tiles(self, tiles):
        """
        Remove tiles that cause normalization errors (where min==max in percentiles)
        Returns filtered list of tiles and their positions
        """
        good_tiles = []
        good_positions = []
        
        for tile, position in tiles:
            try:
                has_valid_data = True
                
                # Check each channel for normalization issues
                for channel in range(tile.shape[-1]):
                    channel_data = tile[:, :, channel]
                    min_val = np.percentile(channel_data, 2)
                    max_val = np.percentile(channel_data, 98)
                    
                    if min_val >= max_val or np.isnan(min_val) or np.isnan(max_val):
                        has_valid_data = False
                        print(f"Removing problematic tile at position {position}")
                        break
                
                if has_valid_data:
                    if (tile.shape[0] != self.tile_size or tile.shape[1] != self.tile_size):
                        print(f"Removing tile with incorrect shape: {tile.shape}")
                        continue
                        
                    good_tiles.append(tile)
                    good_positions.append(position)
                    
            except Exception as e:
                print(f"Error processing tile at position {position}: {str(e)}")
                continue
                
        print(f"Removed {len(tiles) - len(good_tiles)} problematic tiles")
        print(f"Remaining tiles: {len(good_tiles)}")
        
        return good_tiles, good_positions

    def create_tiles_new(self, image):
        """Create tiles from image with overlap and filter problematic ones"""
        print("Creating tiles...")
        h, w, c = image.shape
        tiles = []
        positions = []
        
        stride = self.tile_size - self.overlap
        n_h = int(np.ceil((h - self.overlap) / stride))
        n_w = int(np.ceil((w - self.overlap) / stride))
        
        pad_h = stride * n_h + self.overlap - h
        pad_w = stride * n_w + self.overlap - w
        
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        # Create tiles and positions
        tile_position_pairs = []
        for i in range(n_h):
            for j in range(n_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + self.tile_size
                w_end = w_start + self.tile_size
                
                tile = image[h_start:h_end, w_start:w_end, :] ## only three bands.
                position = (h_start, h_end, w_start, w_end)
                tile_position_pairs.append((tile, position))
        
        # Filter problematic tiles
        filtered_tiles, filtered_positions = self.filter_tiles(tile_position_pairs)
        
        return np.array(filtered_tiles), filtered_positions, (h, w), (n_h, n_w)

    def apply_augmentation(self, tile):
        """Apply test-time augmentation to a tile including transpose operations"""
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
        """Merge predictions from augmented tiles including transposed versions"""
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

    def merge_predictions(self, predictions, positions, original_shape):
        """Merge predictions back into a single image"""
        h, w = original_shape
        merged = np.zeros((h, w, 1))
        counts = np.zeros((h, w, 1))
        
        weight = np.ones((self.tile_size, self.tile_size, 1))
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
        
        merged = np.divide(merged, counts, where=counts > 0)
        return merged

    def resolve_conflicts_new(self, predictions):
        """
        Resolve predictions using only the building model.
        """
        building_conf = predictions['building']
        building_mask = building_conf > self.thresholds['building']

        # Initialize final mask as background (0)
        final_mask = np.zeros_like(building_mask, dtype=np.uint8)

        # Assign building class where mask is True
        final_mask[building_mask] = self.CLASS_BUILDING

        return final_mask

    def normalize_tile(self, tile):
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

    def predict(self, image_path, output_path):
        """Run ensemble prediction with all models"""
        
        # Read image with full metadata
        with rasterio.open(image_path) as src:
            print(f"Input channels: {self.channels}")
            image = src.read(list(range(1, 4)))  # Read first 3 bands (RGB)
            transform = src.transform
            crs = src.crs
            if crs is None:
                # Use a default CRS (WGS 84) if none exists
                crs = rasterio.crs.CRS.from_epsg(4326)
            image = np.moveaxis(image, 0, -1)
            
            print(f"Original image shape: {image.shape}")
            print(f"Original transform: {transform}")
            
            # Resample image to target resolution
            # resampled_image, new_transform = self.resample_image_to_target_resolution(
            #     image, transform, crs
            # )

            resampled_image = image
            new_transform = transform
            
            print(f"Resampled image shape: {resampled_image.shape}")
            print(f"New transform: {new_transform}")
        
        # Create tiles from resampled image
        tiles, positions, original_shape, (n_h, n_w) = self.create_tiles_new(resampled_image)

        del resampled_image
        gc.collect()

        all_predictions = {}

        # Process each model separately to avoid memory accumulation
        for name, model in self.models.items():
            print(f"\nRunning prediction for {name}...")
            
            if name == "building":
                print("Processing building model...")
            elif name == "vegetation":
                print("Processing vegetation model... Skipping")
                continue  # Skip vegetation for now
            else:
                continue  # Skip other models
            
            # Process tiles in smaller batches to manage memory
            predictions = []
            total_tiles = len(tiles)
            batch_size = self.config['data']['batch_size']

            for chunk_start in range(0, total_tiles, batch_size):
                chunk_end = min(chunk_start + batch_size, total_tiles)
                chunk_tiles = tiles[chunk_start:chunk_end]
                
                print(f"Processing chunk {chunk_start//batch_size + 1}/{(total_tiles + batch_size - 1)//batch_size}")
                
                # Process tiles in this chunk
                processed_chunk_tiles = []
                for tile in chunk_tiles:
                    try:
                        # Apply RGB Normalization and preprocessing
                        rgb_tile = self.normalize_tile(tile)
                        processed_tile = self.preprocess_rgb_image(rgb_tile)
                        processed_chunk_tiles.append(processed_tile)
                    except Exception as e:
                        print(f"Error processing tile for rgb: {str(e)}")
                        # Use fallback processing or skip this tile
                        continue
                
                # Convert to numpy array for this chunk only
                if processed_chunk_tiles:
                    chunk_array = np.array(processed_chunk_tiles)
                    
                    # Process each tile in the chunk
                    for i, tile in enumerate(chunk_array):
                        try:
                            # Apply augmentations
                            augmented_tiles = self.apply_augmentation(tile)
                            aug_predictions = []
                            
                            # Process augmented tiles in smaller sub-batches
                            aug_batch_size = min(4, len(augmented_tiles))
                            for aug_start in range(0, len(augmented_tiles), aug_batch_size):
                                aug_end = min(aug_start + aug_batch_size, len(augmented_tiles))
                                aug_batch = augmented_tiles[aug_start:aug_end]
                                
                                # Get predictions for this augmentation batch
                                batch_preds = model.predict(aug_batch, verbose=0)
                                aug_predictions.extend(batch_preds)
                            
                            # Merge augmented predictions
                            merged_pred = self.merge_augmented_predictions(aug_predictions)
                            predictions.append(merged_pred)
                            
                            # Clear augmentation data from memory
                            del augmented_tiles, aug_predictions
                            
                        except Exception as e:
                            print(f"Error in prediction for tile {chunk_start + i}: {str(e)}")
                            continue
                    
                    # Clear chunk data from memory
                    del chunk_array, processed_chunk_tiles
                    gc.collect()
             
            # Merge all predictions for this model
            print(f"Merging {len(predictions)} predictions for {name}")
            merged = self.merge_predictions(predictions, positions, original_shape)
            all_predictions[name] = merged[:,:,0]  # Remove channel dimension
            
            # Clear predictions from memory before saving
            del predictions
            gc.collect()
        
            # Save individual model prediction before conflict resolution
            individual_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(output_path))[0]}_{name}_res{str(self.target_resolution).replace('.', 'p')}m.tif"
            )
        
            with rasterio.open(
                individual_output,
                'w',
                driver='GTiff',
                height=original_shape[0],
                width=original_shape[1],
                count=1,
                dtype=rasterio.uint8,
                crs=crs,
                transform=new_transform,  # Use the new transform from resampling
            ) as dst:
                # Save both binary mask and probabilities
                binary_mask = (merged[:,:,0] > self.thresholds[name]).astype(np.uint8)
                dst.write(binary_mask, 1)
                dst.update_tags(
                    MODEL_NAME=name,
                    THRESHOLD=str(self.thresholds[name]),
                    CLASS_NAME=name.upper(),
                    TARGET_RESOLUTION=str(self.target_resolution),
                    ZOOM_APPLIED="TRUE"
                )
            print(f"Saved {name} prediction to {individual_output}")
        
            # Also save probability map
            prob_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(output_path))[0]}_{name}_res{str(self.target_resolution).replace('.', 'p')}m_prob.tif"
            )
            with rasterio.open(
                prob_output,
                'w',
                driver='GTiff',
                height=original_shape[0],
                width=original_shape[1],
                count=1,
                dtype=rasterio.float32,
                crs=crs,
                transform=new_transform,  # Use the new transform from resampling
            ) as dst:
                dst.write(merged[:,:,0].astype(np.float32), 1)
                dst.update_tags(
                    MODEL_NAME=name,
                    CONTENT_TYPE="PROBABILITY_MAP",
                    TARGET_RESOLUTION=str(self.target_resolution),
                    ZOOM_APPLIED="TRUE"
                )
            print(f"Saved {name} probability map to {prob_output}")

        # Resolve conflicts and create final prediction
        print("\nResolving prediction conflicts...")
        final_mask = self.resolve_conflicts_new(all_predictions)
        
        final_output = os.path.join(
            output_path,
            f"{os.path.splitext(os.path.basename(output_path))[0]}_multiclass_res{str(self.target_resolution).replace('.', 'p')}m.tif"
        )

        # Save final prediction as GeoTIFF
        print(f"\nSaving final prediction to {final_output}")
        with rasterio.open(
            final_output,
            'w',
            driver='GTiff',
            height=original_shape[0],
            width=original_shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=new_transform,  # Use the new transform from resampling
        ) as dst:
            dst.write(final_mask, 1)
            
            # Add classification scheme to metadata
            dst.update_tags(
                CLASS_VALUES="0,1,2,3",
                CLASS_NAMES="Background,Building,Vegetation,Water",
                TARGET_RESOLUTION=str(self.target_resolution),
                ZOOM_APPLIED="TRUE"
            )
        
        # Calculate and return class statistics
        total_pixels = final_mask.size
        stats = {
            'background_percentage': np.sum(final_mask == self.CLASS_BACKGROUND) / total_pixels * 100,
            'building_percentage': np.sum(final_mask == self.CLASS_BUILDING) / total_pixels * 100,
            'vegetation_percentage': np.sum(final_mask == self.CLASS_VEGETATION) / total_pixels * 100,
            'water_percentage': np.sum(final_mask == self.CLASS_WATER) / total_pixels * 100,
            'target_resolution': self.target_resolution
        }

        return stats