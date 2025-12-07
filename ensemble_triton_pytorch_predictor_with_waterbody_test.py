import sys
from pathlib import Path
import os
from skimage import io
from osgeo import gdal
from scipy import ndimage
#import tensorflow as tf
import rasterio
import numpy as np
from tqdm import tqdm
#from model import create_unet_model
import yaml
import tritonclient.http as triton_client
from skimage import exposure


class EnsemblePredictor:
    """Predictor class for ensemble prediction with multiple semantic segmentation models"""
    
    def __init__(self, config_path='config/config.yaml', class_priority=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tile_size = self.config['data']['input_size']
        self.overlap = 32
        self.channels = self.config['data']['channels']
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
        self.max_batch_size = 3
        try:
            self.client = triton_client.InferenceServerClient(url="localhost:9002")
        except Exception as e:
            print(f'Unable to Connect to Triton {e}')
        self.input_name = "input_1"
        self.output_name = "conv2d_41" 
        
        # Validate class priority
        valid_classes = {self.CLASS_BUILDING, self.CLASS_VEGETATION , self.CLASS_WATER}
        if not set(self.class_priority) == valid_classes:
            raise ValueError("Class priority must contain exactly the classes: Building(1), Vegetation(2), and Water(3)")
    
    def load_models(self, model_paths):
        """
        Load multiple models with their respective thresholds
        
        Args:
            model_paths: dict with keys as model names and values as tuples of (model_path, threshold)
            e.g., {
                'building': ('path/to/building_model.h5', 0.444),
                'vegetation': ('path/to/vegetation_model.h5', 0.5),
                'water': ('path/to/water_model.h5', 0.6)
            }
        """
        for name, (model_name, threshold, input_name, output_name) in model_paths.items():
            # print(f"Loading {name} model from {model_path}...")
            # model = create_unet_model(self.config)
            # model.load_weights(model_path)
            self.thresholds[name] = threshold
            self.models[name] = {
                'model_name': model_name,
                'input_name': input_name,
                'output_name': output_name
            }

    def minMax(self, band):
        band = np.float32(band)       
        band = (band - band.min()) / (band.max() - band.min())  
        return band
    
    def change_projection(self, inputfile, referencefile):
        print("Changing Projection...\n")
        file = inputfile
        outfile = inputfile
        ds = gdal.Open(file)
        # band = ds.GetRasterBand(1)
        arr = ds.ReadAsArray()
        # print(arr.shape)
        if len(arr.shape) == 3:
            num = arr.shape[0]
            ds_ref = gdal.Open(referencefile)
            geotrans = ds_ref.GetGeoTransform()
            proj = ds_ref.GetProjection()
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(outfile, arr.shape[2], arr.shape[1], arr.shape[0], gdal.GDT_Float32)
            outdata.SetGeoTransform(geotrans)  ##sets same geotransform as input
            outdata.SetProjection(proj)  ##sets same projection as input
            # outdata.GetRasterBand(1).WriteArray(arr)
            for i in range(1, int(num) + 1):
                # print(arr.shape)
                outdata.GetRasterBand(i).WriteArray(arr[i - 1, :, :])
                outdata.GetRasterBand(i).SetNoDataValue(10000)
            # outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
            outdata.FlushCache()  ##saves to disk!!
            outdata = None
            band = None
            ds = None
        elif len(arr.shape) == 2:
            [cols, rows] = arr.shape
            ds_ref = gdal.Open(referencefile)
            geotrans = ds_ref.GetGeoTransform()
            proj = ds_ref.GetProjection()
            driver = gdal.GetDriverByName("GTiff")
            outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
            outdata.SetGeoTransform(geotrans)  ##sets same geotransform as input
            outdata.SetProjection(proj)  ##sets same projection as input
            outdata.GetRasterBand(1).WriteArray(arr)
            outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
            outdata.FlushCache()  ##saves to disk!!
            outdata = None
            band = None
            ds = None


    def add_sobel_filter(self, image):
        try:
            img = io.imread(image)
            img2 = gdal.Open(image)
            arr = img2.ReadAsArray()

            dra_img = []
            for band in range(img.shape[2] - 1):
                arr1 = img[:, :, 2 - band]
                arr2 = arr1.copy()
                arr2[arr2 > 0] = 1
                thr1 = round(np.percentile(arr1[arr1 > 0], 2.5))
                thr2 = round(np.percentile(arr1[arr1 > 0], 99))
                arr1[arr1 < thr1] = thr1
                arr1[arr1 > thr2] = thr2
                arr1 = (arr1 - thr1) / (thr2 - thr1)
                arr1 = arr1 * 255.
                arr1 = np.uint8(arr1)
                arr1[arr1 == 0] = 1.
                arr2 = np.uint8(arr2)
                arr1 = arr1 * arr2
                dra_img.append(arr1)
            dra_img = np.array(dra_img)
            dra_img = np.rollaxis(dra_img, 0, 3)
            dra_img = np.uint8(dra_img)
            b1 = dra_img[:, :, 0]
            b2 = dra_img[:, :, 1]
            b3 = dra_img[:, :, 2]
            b1 = self.minMax(b1)
            b2 = self.minMax(b2)
            b3 = self.minMax(b3)

            grey1 = (b1 + b2 + b3) / 3

            grey2 = grey1.copy()
            grey2[grey2 > 0] = 1
            grey = grey1.copy()
            grey = grey[grey != 0]
            thr1 = grey.mean() - 2 * grey.std()
            thr2 = grey.mean() + 2 * grey.std()
            grey1[grey1 < thr1] = thr1
            grey1[grey1 > thr2] = thr2
            grey1 = (grey1 - thr1) / (thr2 - thr1)
            sobelx = ndimage.sobel(grey1, 0)
            sobely = ndimage.sobel(grey1, 1)
            sobel = np.hypot(sobelx, sobely)
            sobelMinStd = (sobel - sobel.min()) / (sobel.std())
            sobelMinStd = sobelMinStd * grey2
            sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))
            base_name = os.path.splitext(os.path.basename(image))[0]
            output_dir = os.path.dirname(image)
            rgb_sobel_path = os.path.join(output_dir, f"{base_name}_rgb_sobel.tif")
            b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
            b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
            b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
            img = np.concatenate((b1, b2, b3), axis=-1)
            rgbsobelMinStd = np.concatenate((img, sobelMinStd), axis=-1)
            self.change_projection(rgb_sobel_path, image)
            io.imsave(rgb_sobel_path, rgbsobelMinStd)
            return rgb_sobel_path
        except Exception as e:
            print(f"Error in add_sobel_filter: {str(e)}")
            import traceback
            traceback.print_exc()
            return image


    def preprocess_sobel_image_old(self, image):
        # Normalize each channel using percentile-based normalization
        processed = np.zeros_like(image, dtype=np.float32)
        for j in range(self.channels):
            channel = image[:, :, j]
            min_val = np.percentile(channel, 2)
            max_val = np.percentile(channel, 98)
            processed[:, :, j] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
        return processed

    def normalize_channel_veg(self, channel):
        """
        Advanced channel normalization with adaptive histogram equalization.
        
        Args:
            channel (numpy.ndarray): Input channel to normalize
        
        Returns:
            numpy.ndarray: Normalized and enhanced channel
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
            channel = (channel - min_val) / (max_val - min_val)
            
            # Apply Adaptive Histogram Equalization (CLAHE)
            # This enhances local contrast and helps in feature visibility
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
            band = (band - np.min(band)) / (np.max(band) - np.min(band))
            img[:, :, i] = exposure.equalize_adapthist(band)
        return img


    # def add_sobel_filter_nir_sobel(self,tile):
    #     try:
    #         if tile.shape[2] != 4:
    #             raise ValueError("Input tile must have 4 channels (RGB + NIR).")

    #         img = tile.astype(np.float32)
            
    #         dra_img = []
            
    #         # Process RGB channels with percentile normalization
    #         for band in range(3):  # Only process first three bands (R, G, B)
    #             arr1 = img[:, :, band]
    #             arr2 = arr1.copy()
    #             arr2[arr2 > 0] = 1
                
    #             # Percentile normalization for RGB channels
    #             thr1 = round(np.percentile(arr1[arr1 > 0], 2.5))
    #             thr2 = round(np.percentile(arr1[arr1 > 0], 99))
    #             arr1[arr1 < thr1] = thr1
    #             arr1[arr1 > thr2] = thr2
    #             arr1 = (arr1 - thr1) / (thr2 - thr1) * 255.
    #             arr1 = np.uint8(arr1)
    #             arr2 = np.uint8(arr2)
    #             arr1[arr1 == 0] = 1.
    #             arr1 = arr1 * arr2
    #             dra_img.append(arr1)

    #         # Convert dra_img to array and roll axis
    #         dra_img = np.array(dra_img)
    #         dra_img = np.rollaxis(dra_img, 0, 3)  # Shape: (height, width, channels)

    #         # Extract RGB channels
    #         b1 = dra_img[:, :, 0]
    #         b2 = dra_img[:, :, 1]
    #         b3 = dra_img[:, :, 2]

    #         # Create grayscale image from RGB for Sobel filtering
    #         grey1 = (b1 + b2 + b3) / 3

    #         # Sobel filter application  
    #         sobelx = ndimage.sobel(grey1, axis=0)
    #         sobely = ndimage.sobel(grey1, axis=1)
    #         sobel = np.hypot(sobelx, sobely)

    #         # Normalize Sobel output to range [0, 1]
    #         sobelMinStd = (sobel - sobel.min()) / (sobel.max() - sobel.min())
            
    #         # Reshape Sobel output to match dimensions
    #         sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))

    #         # Process NIR channel with percentile normalization
    #         nir_channel = img[:, :, 3]
            
    #         # Percentile normalization for NIR channel
    #         nir_thr1 = round(np.percentile(nir_channel[nir_channel > 0], 2.5))
    #         nir_thr2 = round(np.percentile(nir_channel[nir_channel > 0], 99))
    #         nir_channel[nir_channel < nir_thr1] = nir_thr1
    #         nir_channel[nir_channel > nir_thr2] = nir_thr2
    #         nir_channel_normalized = (nir_channel - nir_thr1) / (nir_thr2 - nir_thr1)

    #         # Reshape RGB channels for concatenation
    #         b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
    #         b2 = np.reshape(b2, (b2.shape[0], b2.shape[1], 1))
    #         b3 = np.reshape(b3, (b3.shape[0], b3.shape[1], 1))
            
    #         # Concatenate RGB + NIR + Sobel to create a new image with all channels
    #         img_combined = np.concatenate((b1, b2, b3, nir_channel_normalized[:, :, np.newaxis], sobelMinStd), axis=-1)
    #         # print("$$$$$",img_combined.shape)

    #         return img_combined
        
    #     except Exception as e:
    #         print(f"Error in add_sobel_filter_nir_sobel: {e}")
    #         return None

    def apply_sobel_to_tile_waterbody(self,tile):
        """Apply Sobel filter to an individual tile with consistent normalization."""

        if tile.shape[2] != 4:
            raise ValueError("Input tile must have 4 channels (RGB + NIR).")

        # Extract channels
        r = tile[:, :, 0]
        g = tile[:, :, 1]
        b = tile[:, :, 2]
        nir_channel = tile[:, :, 3]

        # --- Normalize RGB with min-max (0-255) ---
        def minMax(x):
            x_min = x.min()
            x_max = x.max()
            if x_min == x_max:
                return np.zeros_like(x, dtype=np.float32)
            return ((x - x_min) / (x_max - x_min)).astype(np.float32)


        b1 = minMax(r)
        b2 = minMax(g)
        b3 = minMax(b)

        # --- Grayscale image for Sobel filtering ---
        grey1 = (b1 + b2 + b3) / 3.0

        sobelx = ndimage.sobel(grey1, axis=0)
        sobely = ndimage.sobel(grey1, axis=1)
        sobel = np.hypot(sobelx, sobely)

        # --- Normalize Sobel with percentile (2.5, 97.5) ---
        if np.any(sobel > 0):
            sobel_thr1 = np.percentile(sobel[sobel > 0], 2.5)
            sobel_thr2 = np.percentile(sobel[sobel > 0], 97.5)
            if sobel_thr1 == sobel_thr2:
                sobel_thr2 += 1e-5  # avoid division by zero
            sobel = np.clip(sobel, sobel_thr1, sobel_thr2)
            sobelMinStd = (sobel - sobel_thr1) / (sobel_thr2 - sobel_thr1)
        else:
            sobelMinStd = np.zeros_like(sobel)

        sobelMinStd = sobelMinStd[:, :, np.newaxis]

        # --- Normalize NIR with percentile (2.5, 97.5) ---
        if np.any(nir_channel > 0):
            nir_thr1 = np.percentile(nir_channel[nir_channel > 0], 2.5)
            nir_thr2 = np.percentile(nir_channel[nir_channel > 0], 97.5)
            if nir_thr1 == nir_thr2:
                nir_thr2 += 1e-5
            nir_channel = np.clip(nir_channel, nir_thr1, nir_thr2)
            nir_channel_normalized = (nir_channel - nir_thr1) / (nir_thr2 - nir_thr1)
        else:
            nir_channel_normalized = np.zeros_like(nir_channel)

        nir_channel_normalized = nir_channel_normalized[:, :, np.newaxis]

        # --- Reshape RGB ---
        b1 = b1[:, :, np.newaxis]
        b2 = b2[:, :, np.newaxis]
        b3 = b3[:, :, np.newaxis]

        # --- Final concat: RGB + NIR + Sobel ---
        tile_combined = np.concatenate((b1, b2, b3, nir_channel_normalized, sobelMinStd), axis=-1)

        tile_combined = tile_combined.astype(np.float32)
        return tile_combined



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
                for channel in range(tile.shape[-1]-1):
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
                print(f"Error processing tile {position}: {str(e)}")
                continue
                
        print(f"Removed {len(tiles) - len(good_tiles)} problematic tiles")
        print(f"Remaining tiles: {len(good_tiles)}")
        
        return good_tiles, good_positions

    def create_tiles_new(self, image):
        """Create tiles from image with overlap and filter problematic ones"""
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
                
                tile = image[h_start:h_end, w_start:w_end, :]
                position = (h_start, h_end, w_start, w_end)
                tile_position_pairs.append((tile, position))
        
        # Filter problematic tiles
        filtered_tiles, filtered_positions = self.filter_tiles(tile_position_pairs)
        
        return np.array(filtered_tiles), filtered_positions, (h, w), (n_h, n_w)


    def create_tiles(self, image):
        """Create tiles from image with overlap"""
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
        
        for i in range(n_h):
            for j in range(n_w):
                h_start = i * stride
                w_start = j * stride
                h_end = h_start + self.tile_size
                w_end = w_start + self.tile_size
                
                tile = image[h_start:h_end, w_start:w_end, :]
                tiles.append(tile)
                positions.append((h_start, h_end, w_start, w_end))
        
        return np.array(tiles), positions, (h, w), (n_h, n_w)

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
        """
        Merge predictions from augmented tiles.
        This version is definitively corrected for NCHW (Channels, Height, Width) format by
        explicitly specifying the axes for flip and rotation operations.
        """
        # predictions[i] has shape (C, H, W), e.g., (1, 512, 512)
        # The spatial axes are Height (1) and Width (2).

        # Reverse the standard augmentations
        pred_original = predictions[0]

        # --- CORE FIX: Use np.flip and specify the correct axis ---
        # Horizontal flip is along the Width axis (axis=2)
        pred_fliplr = np.flip(predictions[1], axis=2)
        # Vertical flip is along the Height axis (axis=1)
        pred_flipud = np.flip(predictions[2], axis=1)

        # --- CORE FIX: Specify the spatial axes (1, 2) for rotation ---
        pred_rot90_1 = np.rot90(predictions[3], k=-1, axes=(1, 2))
        pred_rot90_2 = np.rot90(predictions[4], k=-2, axes=(1, 2))
        pred_rot90_3 = np.rot90(predictions[5], k=-3, axes=(1, 2))

        # Reverse transpose operations (swapping Height and Width axes)
        pred_transpose = np.transpose(predictions[6], (0, 2, 1))
        pred_transpose_rot90_1 = np.rot90(np.transpose(predictions[7], (0, 2, 1)), k=-1, axes=(1, 2))
        pred_transpose_rot90_2 = np.rot90(np.transpose(predictions[8], (0, 2, 1)), k=-2, axes=(1, 2))
        pred_transpose_rot90_3 = np.rot90(np.transpose(predictions[9], (0, 2, 1)), k=-3, axes=(1, 2))

        # All reversed predictions now have the same shape, so np.mean will work.
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

    def resolve_conflicts(self, predictions, confidence_threshold=0.5):
        """
        Resolve conflicts between different model predictions
        
        Strategy:
        1. If only one model predicts a class with confidence > threshold, use that
        2. If multiple models predict with high confidence:
           - Take the prediction with highest confidence
           - If confidences are close (within 0.1), prefer: building > water > vegetation
        3. If no model predicts with high confidence, mark as background
        """
        building_pred = predictions['building'] > self.thresholds['building']
        vegetation_pred = predictions['vegetation'] > self.thresholds['vegetation']
        water_pred = predictions['water'] > self.thresholds['water']
        
        # Initialize final mask with background (0)
        final_mask = np.zeros_like(building_pred, dtype=np.uint8)
        
        # Get confidence scores
        building_conf = predictions['building']
        vegetation_conf = predictions['vegetation']
        water_conf = predictions['water']
        
        # Create masks for each class based on confidence
        building_mask = (building_conf > self.thresholds['building'])
        vegetation_mask = (vegetation_conf > self.thresholds['vegetation'])
        water_mask = (water_conf > self.thresholds['water'])
        
        # Handle conflicts
        for i in range(building_pred.shape[0]):
            for j in range(building_pred.shape[1]):
                # Count how many models predict this pixel
                predictions_count = (building_mask[i,j] + vegetation_mask[i,j] + water_mask[i,j])
                
                if predictions_count == 0:
                    final_mask[i,j] = self.CLASS_BACKGROUND
                
                elif predictions_count == 1:
                    # Only one model predicts - use that prediction
                    if building_mask[i,j]:
                        final_mask[i,j] = self.CLASS_BUILDING
                    elif vegetation_mask[i,j]:
                        final_mask[i,j] = self.CLASS_VEGETATION
                    elif water_mask[i,j]:
                        final_mask[i,j] = self.CLASS_WATER
                
                else:
                    # Multiple predictions - get confidences
                    confs = [
                        (self.CLASS_BUILDING, building_conf[i,j] if building_mask[i,j] else 0),
                        (self.CLASS_VEGETATION, vegetation_conf[i,j] if vegetation_mask[i,j] else 0),
                        (self.CLASS_WATER, water_conf[i,j] if water_mask[i,j] else 0)
                    ]
                    
                    # Sort by confidence
                    confs.sort(key=lambda x: x[1], reverse=True)
                    
                    # If highest confidence is significantly higher (>0.1 difference)
                    if confs[0][1] - confs[1][1] > 0.1:
                        final_mask[i,j] = confs[0][0]
                    else:
                        # If confidences are close, use configured class priority
                        classes = [c[0] for c in confs if c[1] > 0]
                        for priority_class in self.class_priority:
                            if priority_class in classes:
                                final_mask[i,j] = priority_class
                                break
        
        return final_mask

    def apply_sobel_to_tile(self, tile):
            try:
                img = tile.astype(np.float32)
                dra_img = []
                for band in range(img.shape[2] - 1):
                    arr1 = img[:, :, 2 - band]
                    arr2 = arr1.copy()
                    arr2[arr2 > 0] = 1
                    thr1 = round(np.percentile(arr1[arr1 > 0], 2.5))
                    thr2 = round(np.percentile(arr1[arr1 > 0], 99))
                    arr1[arr1 < thr1] = thr1
                    arr1[arr1 > thr2] = thr2
                    arr1 = (arr1 - thr1) / (thr2 - thr1)
                    arr1 = arr1 * 255.
                    arr1 = np.uint8(arr1)
                    arr1[arr1 == 0] = 1.
                    arr2 = np.uint8(arr2)
                    arr1 = arr1 * arr2
                    dra_img.append(arr1)
                dra_img = np.array(dra_img)
                dra_img = np.rollaxis(dra_img, 0, 3)
                dra_img = np.uint8(dra_img)
                b1 = dra_img[:, :, 0]
                b2 = dra_img[:, :, 1]
                b3 = dra_img[:, :, 2]
                b1 = self.minMax(b1)
                b2 = self.minMax(b2)
                b3 = self.minMax(b3)

                grey1 = (b1 + b2 + b3) / 3
                grey2 = grey1.copy()
                grey2[grey2 > 0] = 1
                grey = grey1.copy()
                grey = grey[grey != 0]
                thr1 = grey.mean() - 2 * grey.std()
                thr2 = grey.mean() + 2 * grey.std()
                grey1[grey1 < thr1] = thr1
                grey1[grey1 > thr2] = thr2
                grey1 = (grey1 - thr1) / (thr2 - thr1)
                sobelx = ndimage.sobel(grey1, 0)
                sobely = ndimage.sobel(grey1, 1)
                sobel = np.hypot(sobelx, sobely)

                # Add check for division by zero
                std = sobel.std()
                if std == 0:
                    sobelMinStd = np.zeros_like(sobel)
                else:
                    sobelMinStd = (sobel - sobel.min()) / std

                sobelMinStd = sobelMinStd * grey2
                sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))

                b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
                b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
                b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
                img = np.concatenate((b1, b2, b3), axis=-1)
                rgbsobelMinStd = np.concatenate((img, sobelMinStd), axis=-1)
                return rgbsobelMinStd
            except Exception as e:
                print(f"Error in add_sobel_filter: {str(e)}")
                import traceback
                traceback.print_exc()
                return tile  # Return original tile instead of undefined 'image'

    # def normalize_tile(self, tile):
    #     tile = np.transpose(tile, (2, 0, 1)) 
    #     rd_ras = tile.astype(np.float32)
    #     dra_img = []
    #     for band in range(rd_ras.shape[0]-1):
    #         arr = rd_ras[band]
    #         arr1 = arr.copy()
    #         thr1 = round(np.percentile(arr[arr > 0], 2.5))
    #         thr2 = round(np.percentile(arr[arr > 0], 99))
    #         arr1[arr1 < thr1] = thr1
    #         arr1[arr1 > thr2] = thr2
    #         arr1 = (arr1 - thr1) / (thr2 - thr1)
    #         arr1 = arr1 * 255.
    #         arr1 = np.uint8(arr1)
    #         arr1[arr1 == 0] = 1.
    #         dra_img.append(arr1)

    #     foo = np.stack(dra_img, axis=-1)
    #     foo = foo.astype(np.float32) / 255.0
    #     return foo

    def normalize_tile(self, tile):
        """
        Normalize tile using EXACT same method as training (Channel3_DataGenerator_old).
        
        Training used: image = image.astype(np.float32) / 255.0
        Inference must do the same to avoid distribution mismatch!
        
        tile shape: (H, W, C) from rasterio.open().read()
        """
        print("## Normalizing tile")
        # Convert to float32 and divide by 255
        # This matches exactly what Channel3_DataGenerator_old does
        tile = tile.astype(np.float32) / 255.0
        
        # Result: [0.004-1.0] range (because of zero→1 replacement in scale_tile)
        # This is EXACTLY what model was trained on!
        
        # No percentile recomputation!
        # No uint8 conversion!
        # No transposing needed!
        
        return tile


    def predict(self, image_path, output_path):
        """Run ensemble prediction with all models"""
        
        # Read image
        with rasterio.open(image_path) as src:
            print(self.channels)
            image = src.read(list(range(1, 5)))
            transform = src.transform
            crs = src.crs
            if crs is None:
                # Use a default CRS (WGS 84) if none exists
                crs = rasterio.crs.CRS.from_epsg(4326)
            image = np.moveaxis(image, 0, -1)
        
        # Create tiles first
        tiles, positions, original_shape, (n_h, n_w) = self.create_tiles_new(image)
        
        # Process tiles with Sobel filter and preprocessing
        processed_sobel_tiles = []
        processed_rgb_tiles = []
        processed_rgb_nir_sobel_tiles = []
        print("\nApplying Sobel filter and preprocessing to tiles...")
        
        for tile in tqdm(tiles, desc="Processing tiles"):
            try:
                continue
                # Apply Sobel filter and preprocessing directly in memory
                sobel_tile = self.apply_sobel_to_tile(tile)
                
                # Apply preprocessing
                processed_tile = self.preprocess_sobel_image_old(sobel_tile)
                    
                processed_sobel_tiles.append(processed_tile)
                
            except Exception as e:
                print(f"Error processing tile for sobel: {str(e)}")
                print("Using original tile with preprocessing only")
                
                
        for tile in tqdm(tiles, desc="Processing tiles"):
            try:
                continue
                # Apply Sobel filter and preprocessing directly in memory
                sobel_tile = self.apply_sobel_to_tile_waterbody(tile)
                
                # Apply preprocessing
                # processed_tile = self.preprocess_sobel_image_old(sobel_tile)
                processed_tile = sobel_tile
                    
                processed_rgb_nir_sobel_tiles.append(processed_tile)
                
            except Exception as e:
                print(f"Error processing tile for sobel: {str(e)}")
                print("Using original tile with preprocessing only")

        for tile in tqdm(tiles, desc="Processing tiles"):
            try:
                # Apply RGB Normalization and preprocessing directly in memory
                rgb_tile = self.normalize_tile(tile)

                # Apply preprocessing
                processed_tile = self.preprocess_rgb_image(rgb_tile)

                processed_rgb_tiles.append(processed_tile)

            except Exception as e:
                print(f"Error processing tile for rgb: {str(e)}")
                print("Using original tile with preprocessing only")

        
        # Convert back to numpy array
        sobel_tiles = np.array(processed_sobel_tiles)
        rgb_tiles = np.array(processed_rgb_tiles)
        rgb_nir_sobel_tiles = np.array(processed_rgb_nir_sobel_tiles)
        all_predictions = {}
        print(self.models)
        # Run prediction for each model
        for name, model in self.models.items():
            print(f"\nRunning prediction for {name}...")
            predictions = []

            if(name == "building"):
                #tiles = sobel_tiles
                tiles = rgb_tiles
            elif(name == "vegetation"):
                tiles = rgb_tiles
            else:
                tiles = rgb_nir_sobel_tiles

            for tile in tqdm(tiles, dynamic_ncols=True, desc=f"Processing {name} tiles"):
                # Apply augmentations for the current tile
                augmented_tiles = self.apply_augmentation(tile)
                num_patches = augmented_tiles.shape[0]
                
                # --------------------------------------------------------------------------
                # FIX: Initialize the list here, once per tile, before the batching loop.
                batch_predictions_list = []
                # --------------------------------------------------------------------------

                # This loop processes all augmentations of a SINGLE tile in batches
                for start_idx in range(0, num_patches, self.max_batch_size):
                    end_idx = min(start_idx + self.max_batch_size, num_patches)
                    batch_patches = augmented_tiles[start_idx:end_idx]

                    # Transpose from NHWC to NCHW format for the ONNX model
                    batch_patches_nchw = np.transpose(batch_patches, (0, 3, 1, 2)).astype(np.float32)
                    
                    # Prepare Triton inputs
                    inputs = [triton_client.InferInput(model["input_name"], batch_patches_nchw.shape, "FP32")]
                    inputs[0].set_data_from_numpy(batch_patches_nchw)

                    outputs = [triton_client.InferRequestedOutput(model["output_name"])]

                    try:
                        # Perform inference for the current batch
                        response = self.client.infer(model["model_name"], inputs, model_version="3", outputs=outputs)
                        batch_output = response.as_numpy(model["output_name"])

                        # Apply sigmoid here to convert logits → probabilities
                        # batch_output = 1 / (1 + np.exp(-batch_output))
                        
                        # Append the entire numpy array from the batch to the list
                        batch_predictions_list.append(batch_output)

                    except Exception as e:
                        print(f"Inference failed for batch {start_idx // self.max_batch_size + 1}: {e}")

                # AFTER the batching loop, concatenate all results for this tile's augmentations
                if batch_predictions_list:
                    aug_predictions = np.concatenate(batch_predictions_list, axis=0)
                else:
                    # This handles both the case of an empty input and the case where all batches failed
                    print(f"Warning: No successful predictions for one of the '{name}' tiles. Skipping.")
                    aug_predictions = np.array([]) 

                # We need to handle the case where aug_predictions is empty
                if aug_predictions.size > 0:
                    merged_pred = self.merge_augmented_predictions(aug_predictions)
                    merged_pred = np.transpose(merged_pred, (1, 2, 0))
                    predictions.append(merged_pred)

            merged = self.merge_predictions(predictions, positions, original_shape)

            all_predictions[name] = merged[:,:,0]  # Remove channel dimension
        
            # Save individual model prediction before conflict resolution
            individual_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_{name}.tif"
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
                transform=transform,
            ) as dst:
                # Save both binary mask and probabilities
                binary_mask = (merged[:,:,0] > self.thresholds[name]).astype(np.uint8)
                dst.write(binary_mask, 1)
                dst.update_tags(
                    MODEL_NAME=name,
                    THRESHOLD=str(self.thresholds[name]),
                    CLASS_NAME=name.upper()
                )
            print(f"Saved {name} prediction to {individual_output}")
        
            # Also save probability map
            prob_output = os.path.join(
                output_path,
                f"{os.path.splitext(os.path.basename(image_path))[0]}_{name}_prob.tif"
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
                transform=transform,
            ) as dst:
                dst.write(merged[:,:,0].astype(np.float32), 1)
                dst.update_tags(
                    MODEL_NAME=name,
                    CONTENT_TYPE="PROBABILITY_MAP"
                )
            print(f"Saved {name} probability map to {prob_output}")

        # Resolve conflicts and create final prediction
        print("\nResolving prediction conflicts...")
        final_mask = self.resolve_conflicts(all_predictions)
        
        final_output = os.path.join(
            os.path.dirname(output_path),
            f"{os.path.splitext(os.path.basename(output_path))[0]}_mutliclass.tif"
        )

        # Save final prediction as GeoTIFF
        print(f"\nSaving final prediction to {output_path}")
        with rasterio.open(
            final_output,
            'w',
            driver='GTiff',
            height=original_shape[0],
            width=original_shape[1],
            count=1,
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(final_mask, 1)
            
            # Add classification scheme to metadata
            dst.update_tags(
                CLASS_VALUES="0,1,2,3",
                CLASS_NAMES="Background,Building,Vegetation,Water"
            )
        
        # Calculate and return class statistics
        total_pixels = final_mask.size
        stats = {
            'background_percentage': np.sum(final_mask == self.CLASS_BACKGROUND) / total_pixels * 100,
            'building_percentage': np.sum(final_mask == self.CLASS_BUILDING) / total_pixels * 100,
            'vegetation_percentage': np.sum(final_mask == self.CLASS_VEGETATION) / total_pixels * 100,
            'water_percentage': np.sum(final_mask == self.CLASS_WATER) / total_pixels * 100
        }

        return stats
