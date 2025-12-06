import os
os.environ['SM_FRAMEWORK'] = 'tf.keras'
from skimage import exposure
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import rasterio
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K  
from skimage import exposure
import segmentation_models as sm

class Channel5_DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading satellite imagery without normalization."""
    
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, self.channels))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load image without normalization
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                # Move channels to last dimension
                image = np.moveaxis(image, 0, -1)
                batch_images[i] = image

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                batch_masks[i] = np.expand_dims(mask, axis=-1)

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

class Channel5_SobelNIR_DataGenerator(tf.keras.utils.Sequence):
    """Data generator specifically designed for 5-channel data with Sobel edge map."""
    
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']  # Should be 5
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, self.channels))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                # Move channels to last dimension
                image = np.moveaxis(image, 0, -1)
                
                # Normalize bands 1-4 to [0,1] if they aren't already
                # Note: Based on your stats, they're already in [0,1] range
                for j in range(5):# First 4 bands
                    if(j != 3):
                        if image[:,:,j].max() > 1.0:
                            image[:,:,j] = image[:,:,j] / 255.0
                
                # Handle the Sobel band (5th channel) specifically
                # Clip and normalize to [0,1] range
                sobel_band = image[:,:,3]
                if sobel_band.max() > 1.0:
                    # Clip to reasonable range based on your statistics
                    # (using 3 standard deviations from mean)
                    mean = 0.89  # From your stats
                    std = 0.97   # From your stats
                    upper_limit = mean + 3*std
                    sobel_band = np.clip(sobel_band, 0, upper_limit)
                    # Normalize to [0,1]
                    sobel_band = sobel_band / upper_limit
                    image[:,:,3] = sobel_band
                
                batch_images[i] = image

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)  # Binary mask
                batch_masks[i] = np.expand_dims(mask, axis=-1)

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class Channel3_DataGenerator_old(tf.keras.utils.Sequence):
    """Custom data generator adapted for preprocessed uint8 RGB images"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # This is the required implementation
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, 3))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load preprocessed RGB image (already scaled to uint8)
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                image = np.moveaxis(image, 0, -1)
                # Simple scaling to 0-1 since images are already preprocessed
                image = image.astype(np.float32) / 255.0
                batch_images[i] = image

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                batch_masks[i] = np.expand_dims(mask, axis=-1)

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

class Channel4_DataGenerator_new(tf.keras.utils.Sequence):
    """Custom data generator for loading and preprocessing satellite imagery"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        def preprocess_input(img):
            """
            Preprocess input image with channel-wise normalization and enhancement.

            Args:
                img (numpy.ndarray): Input image array

            Returns:
                numpy.ndarray: Preprocessed image
            """
            img = img.astype(np.float32)
            for i in range(img.shape[-1]):
                band = img[:, :, i]
                band = (band - np.min(band)) / (np.max(band) - np.min(band))
                img[:, :, i] = exposure.equalize_adapthist(band)
            return img

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        self.preprocess_input = preprocess_input

        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, self.channels))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                # Move channels to last dimension and preprocess
                image = np.moveaxis(image, 0, -1)
                image = self.preprocess_input(image)
                batch_images[i] = image

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                batch_masks[i] = np.expand_dims(mask, axis=-1)

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

class Channel4_DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading and preprocessing satellite imagery"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']
        self.classes = config['data']['num_classes']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        #print(self.channels)
        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, self.channels))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load and preprocess image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                # Normalize each channel
                for j in range(src.count):
                    if(True):
                        channel = image[j]
                        min_val = np.percentile(channel, 2)
                        max_val = np.percentile(channel, 98)
                        image[j] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
                image = np.moveaxis(image, 0, -1)
                batch_images[i] = image

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                batch_masks[i] = np.expand_dims(mask, axis=-1)

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


import cv2
import numpy as np
import rasterio
import tensorflow as tf
from scipy.ndimage import rotate

class Channel3_DataGenerator_aug_old(tf.keras.utils.Sequence):
    """Custom data generator adapted for preprocessed RGB images with CPU-only augmentations"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        
        # Only apply augmentations during training
        self.use_augmentation = is_training
        
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def resize_with_opencv(self, image, target_size, interpolation=cv2.INTER_LINEAR):
        """Resize image using OpenCV (CPU-only)"""
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    def resize_mask_with_opencv(self, mask, target_size):
        """Resize mask using OpenCV, handling different dimensions"""
        if len(mask.shape) == 3:
            # If mask has channel dimension, process without it
            mask_2d = mask.squeeze(-1) if mask.shape[-1] == 1 else mask[:,:,0]
            resized = cv2.resize(mask_2d, target_size, interpolation=cv2.INTER_NEAREST)
            return np.expand_dims(resized, axis=-1)
        else:
            # If mask is 2D, resize directly
            resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            return np.expand_dims(resized, axis=-1)

    def apply_augmentations(self, image, mask):
        """Apply various CPU-only augmentations to the image and mask"""
        
        # Ensure mask has the right shape
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        elif len(mask.shape) == 3 and mask.shape[-1] != 1:
            mask = mask[:, :, :1]
        
        # Random horizontal flip (50% probability)
        if np.random.rand() < 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        
        # Random vertical flip (25% probability)
        if np.random.rand() < 0.25:
            image = np.flipud(image)
            mask = np.flipud(mask)
        
        # Random 90-degree rotations (25% probability)
        if np.random.rand() < 0.25:
            k = np.random.randint(1, 4)  # 1: 90°, 2: 180°, 3: 270°
            image = np.rot90(image, k=k)
            mask = np.rot90(mask, k=k)
        
        # Random small rotation (20% probability) - for more variety
        if np.random.rand() < 0.2:
            try:
                angle = np.random.uniform(-15, 15)  # Random angle between -15 and 15 degrees
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                
                # Create rotation matrix
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                # Apply rotation
                image = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                
                # Handle mask rotation (ensure it's 2D for cv2.warpAffine)
                mask_2d = mask.squeeze(-1) if len(mask.shape) == 3 else mask
                mask_rotated = cv2.warpAffine(mask_2d, rotation_matrix, (w, h), 
                                            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                mask = np.expand_dims(mask_rotated, axis=-1)
            except Exception as e:
                print(f"Warning: Small rotation failed: {e}")
                pass
        
        # Random brightness adjustment (30% probability)
        if np.random.rand() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = image * brightness_factor
            image = np.clip(image, 0, 1)
        
        # Random contrast adjustment (30% probability)
        if np.random.rand() < 0.3:
            contrast_factor = np.random.uniform(0.8, 1.2)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = (image - mean) * contrast_factor + mean
            image = np.clip(image, 0, 1)
        
        # Random saturation adjustment for RGB (25% probability)
        if np.random.rand() < 0.25:
            try:
                saturation_factor = np.random.uniform(0.8, 1.2)
                # Convert to HSV, adjust saturation, convert back to RGB
                hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            except Exception as e:
                print(f"Warning: Saturation adjustment failed: {e}")
                pass
            
        # Random scale/crop for multi-scale robustness (30% probability)
        if np.random.rand() < 0.3:
            zoom_factor = np.random.uniform(0.8, 1.2)
            h, w = image.shape[:2]
            
            if zoom_factor < 1.0:  # Zoom out - resize smaller then pad
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                
                # Resize using OpenCV
                image_resized = self.resize_with_opencv(image, (new_w, new_h))
                mask_resized = self.resize_mask_with_opencv(mask, (new_w, new_h))
                
                # Create padded versions
                image_result = np.zeros_like(image)
                mask_result = np.zeros_like(mask)
                
                # Calculate padding
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                
                # Place resized image/mask in center
                image_result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image_resized
                mask_result[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = mask_resized
                
                return image_result, mask_result
                
            else:  # Zoom in - resize larger then crop
                new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
                
                # Resize using OpenCV
                image_resized = self.resize_with_opencv(image, (new_w, new_h))
                mask_resized = self.resize_mask_with_opencv(mask, (new_w, new_h))
                
                # Random crop back to original size
                max_h_start = max(0, new_h - h)
                max_w_start = max(0, new_w - w)
                
                h_start = np.random.randint(0, max_h_start + 1) if max_h_start > 0 else 0
                w_start = np.random.randint(0, max_w_start + 1) if max_w_start > 0 else 0
                
                image_result = image_resized[h_start:h_start+h, w_start:w_start+w]
                mask_result = mask_resized[h_start:h_start+h, w_start:w_start+w]
                
                return image_result, mask_result
        
        # Random Gaussian noise (15% probability)
        if np.random.rand() < 0.15:
            noise_std = np.random.uniform(0.01, 0.05)
            noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)
            image = image + noise
            image = np.clip(image, 0, 1)
        
        # Random gamma correction (20% probability)
        if np.random.rand() < 0.2:
            gamma = np.random.uniform(0.8, 1.2)
            image = np.power(image, gamma)
            image = np.clip(image, 0, 1)
        
        return image, mask

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, 3))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load preprocessed RGB image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                image = np.moveaxis(image, 0, -1)
                # Simple scaling to 0-1
                image = image.astype(np.float32) / 255.0

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1)

            # Apply augmentations during training
            if self.is_training and self.use_augmentation:
                image, mask = self.apply_augmentations(image, mask)

            batch_images[i] = image
            batch_masks[i] = mask

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class Channel4_MultiDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading and preprocessing satellite imagery"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']
        self.classes = config['data']['num_classes']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        #print(self.channels)
        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, self.channels))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, self.classes))

        for i, idx in enumerate(batch_indexes):
            # Load and preprocess image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                # Normalize each channel
                for j in range(src.count):
                    if(True):
                        channel = image[j]
                        min_val = np.percentile(channel, 2)
                        max_val = np.percentile(channel, 98)
                        image[j] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
                image = np.moveaxis(image, 0, -1)
                batch_images[i] = image

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                # for multiclass
                from tensorflow.keras.utils import to_categorical
                mask = src.read(1)
                mask = to_categorical(mask, num_classes=self.classes)  # (H, W, C)
                batch_masks[i] = mask

        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class Channel3_NewDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator that converts 4-channel RGBNIR images to 3-channel RGB images"""
    
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        # This is the required implementation
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, 3))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load 4-channel RGBNIR image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                image = np.moveaxis(image, 0, -1)  # Convert from (C,H,W) to (H,W,C)
                
                # Extract only the RGB channels (first 3 channels)
                rgb_image = image[:, :, :3]
                
                # Ensure image is properly scaled for RGB models
                if rgb_image.dtype == np.uint16:
                    rgb_image = (rgb_image / 65535.0 * 255.0).astype(np.uint8)
                elif rgb_image.dtype != np.uint8:
                    # Normalize to 0-255 range if not already uint8
                    rgb_image = ((rgb_image - rgb_image.min()) / 
                               (rgb_image.max() - rgb_image.min() + 1e-8) * 255.0).astype(np.uint8)
                
                batch_images[i] = rgb_image

            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                batch_masks[i] = np.expand_dims(mask, axis=-1)

        # Normalize RGB values to 0-1 range for model input
        batch_images = batch_images.astype(np.float32) / 255.0
        
        return batch_images, batch_masks

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

class Channel4_DataGenerator_old(tf.keras.utils.Sequence):
    """Custom data generator adapted for preprocessed uint8 RGB images"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        # This is the required implementation
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def preprocess_input(self, img):
        img = img.astype(np.float32)
        for i in range(img.shape[-1]):
            band = img[:, :, i]
            band = (band - np.min(band)) / (np.max(band) - np.min(band))
            img[:, :, i] = exposure.equalize_adapthist(band)
        return img
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, 4))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))
        
        for i, idx in enumerate(batch_indexes):
            # Load preprocessed RGB image 
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                image = np.moveaxis(image, 0, -1)
                # Apply the new preprocessing method
                image = self.preprocess_input(image)
                batch_images[i] = image
            
            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                batch_masks[i] = np.expand_dims(mask, axis=-1)
        
        return batch_images, batch_masks
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

class Channel3_dt_DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for RGB (3-channel) images with dual outputs"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.distance_transform_scale = config.get('distance_transform_scale', 3.0)  # Default scale factor
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        # RGB input
        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, 3))
        # Two outputs: binary mask and distance transform
        batch_binary_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))
        batch_dist_transforms = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))
        
        for i, idx in enumerate(batch_indexes):
            # Load RGB image
            with rasterio.open(self.image_paths[idx]) as src:
                # Read only the first 3 bands (RGB)
                image = src.read([1, 2, 3])
                image = np.moveaxis(image, 0, -1)
                # Scale to 0-1 range
                image = image.astype(np.float32) / 255.0
                batch_images[i] = image
            
            # Load and preprocess masks (2 bands: binary and distance transform)
            with rasterio.open(self.mask_paths[idx]) as src:
                binary_mask = src.read(1)
                dist_transform = src.read(2) if src.count > 1 else None
                
                # Process binary mask
                binary_mask = (binary_mask > 0).astype(np.float32)
                batch_binary_masks[i] = np.expand_dims(binary_mask, axis=-1)
                
                # Process distance transform or create it if not provided
                if dist_transform is None:
                    # Create improved distance transform using boundary-based approach
                    dist_transform = self._create_boundary_distance_transform(binary_mask)
                else:
                    # If distance transform is provided, ensure it's float32
                    dist_transform = dist_transform.astype(np.float32)
                
                batch_dist_transforms[i] = np.expand_dims(dist_transform, axis=-1)
        
        return batch_images, [batch_binary_masks, batch_dist_transforms]
    
    def _create_boundary_distance_transform(self, binary_mask):
        """
        Create boundary-based distance transform similar to the manual preprocessing method
        """
        # Find boundaries using morphological operations
        from scipy import ndimage
        from skimage.morphology import binary_erosion
        
        # Create boundary mask by finding pixels that are foreground but have background neighbors
        # This approximates polygon boundaries
        boundary_mask = binary_mask.astype(bool)
        eroded_mask = binary_erosion(boundary_mask)
        boundary_pixels = boundary_mask & ~eroded_mask
        
        # Convert boundary pixels to boundary array (1 at boundaries, 0 elsewhere)
        boundary_arr = boundary_pixels.astype(np.uint8)
        
        # Create inverted boundary mask (0 at boundaries, 1 elsewhere)
        inverted_boundary = 1 - boundary_arr
        
        # Calculate distance transform from boundaries
        distance_transform = ndimage.distance_transform_edt(inverted_boundary)
        
        # Normalize to 0-1 range
        max_dist = np.max(distance_transform)
        if max_dist > 0:
            normalized_distance = distance_transform / max_dist
        else:
            normalized_distance = distance_transform
        
        # Apply exponential decay for smooth boundary emphasis
        boundary_distance = np.exp(-normalized_distance * self.distance_transform_scale)
        
        return boundary_distance.astype(np.float32)
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class Channel3_DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator adapted for preprocessed uint8 RGB images"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        # This is the required implementation
        return int(np.floor(len(self.image_paths) / self.batch_size))
    
    def preprocess_input(self, img):
        img = img.astype(np.float32)
        for i in range(img.shape[-1]):
            band = img[:, :, i]
            band = (band - np.min(band)) / (np.max(band) - np.min(band))
            img[:, :, i] = exposure.equalize_adapthist(band)
        return img
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, 3))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))
        
        for i, idx in enumerate(batch_indexes):
            # Load preprocessed RGB image 
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                image = np.moveaxis(image, 0, -1)
                # Apply the new preprocessing method
                image = self.preprocess_input(image)
                batch_images[i] = image
            
            # Load and preprocess mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = (mask > 0).astype(np.float32)
                batch_masks[i] = np.expand_dims(mask, axis=-1)
        
        return batch_images, batch_masks
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)

def create_old_enhanced_unet(config):
    """Enhanced U-Net with memory-efficient attention"""
    input_size = config['data']['input_size']
    channels = config['data']['channels']
    
    # Multi-scale input processing
    inputs = layers.Input(shape=(input_size, input_size, channels))
    
    # Create different scales of input
    scale_1 = inputs
    scale_2 = layers.AveragePooling2D(2)(inputs)
    scale_4 = layers.AveragePooling2D(2)(scale_2)
    
    def efficient_attention(x, num_heads=4, key_dim=64):
        """Memory-efficient attention mechanism"""
        input_dim = x.shape[-1]
        
        # Reduce spatial dimensions before attention
        height, width = x.shape[1:3]
        reduction_factor = 4  # Reduce spatial dimensions by this factor
        
        # Apply spatial reduction
        x = layers.AveragePooling2D(pool_size=reduction_factor)(x)
        
        # Reshape for attention
        x = layers.Reshape((-1, input_dim))(x)
        
        # Self-attention with reduced sequence length
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim
        )(x, x)
        
        # Reshape back and upsample
        x = layers.Reshape((height // reduction_factor, width // reduction_factor, input_dim))(attention_output)
        x = layers.UpSampling2D(size=reduction_factor)(x)
        
        return x
    
    def memory_efficient_encoder_block(x, scale_features, filters):
        """Memory-efficient encoder block"""
        # Initial convolutions
        conv = layers.Conv2D(filters, 3, padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        
        # Apply efficient attention
        att = efficient_attention(conv, num_heads=4, key_dim=filters//4)
        
        # Residual connection
        conv = layers.Add()([conv, att])
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        
        # Second convolution
        conv = layers.Conv2D(filters, 3, padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        
        # Multi-scale feature fusion
        if scale_features:
            fused = multi_scale_feature_fusion([conv] + scale_features, filters)
        else:
            fused = conv
        
        return fused, layers.MaxPooling2D(2)(fused)
    
    def multi_scale_feature_fusion(features, filters):
        """Memory-efficient feature fusion"""
        resized_features = []
        target_size = features[0].shape[1:3]
        
        for feature in features:
            current_size = feature.shape[1:3]
            if current_size != target_size:
                # Calculate scaling factor safely
                scale_h = max(1, int(target_size[0] / current_size[0]))
                scale_w = max(1, int(target_size[1] / current_size[1]))
                
                if scale_h > 1 or scale_w > 1:
                    x = layers.UpSampling2D(size=(scale_h, scale_w))(feature)
                else:
                    pool_h = max(1, int(current_size[0] / target_size[0]))
                    pool_w = max(1, int(current_size[1] / target_size[1]))
                    x = layers.AveragePooling2D(pool_size=(pool_h, pool_w))(feature)
            else:
                x = feature
            resized_features.append(x)
        
        # Concatenate and reduce channels
        concat = layers.Concatenate()(resized_features)
        fused = layers.Conv2D(filters, 1, padding='same')(concat)  # 1x1 conv to reduce channels
        fused = layers.BatchNormalization()(fused)
        fused = layers.Activation('relu')(fused)
        fused = layers.Conv2D(filters, 3, padding='same')(fused)
        fused = layers.BatchNormalization()(fused)
        return layers.Activation('relu')(fused)
    
    # Encoder path
    conv1, pool1 = memory_efficient_encoder_block(scale_1, [scale_2, scale_4], 64)
    conv2, pool2 = memory_efficient_encoder_block(pool1, [scale_2, scale_4], 128)
    conv3, pool3 = memory_efficient_encoder_block(pool2, [scale_2, scale_4], 256)
    conv4, pool4 = memory_efficient_encoder_block(pool3, [scale_2, scale_4], 512)
    
    # Bridge
    bridge = layers.Conv2D(1024, 3, padding='same')(pool4)
    bridge = layers.BatchNormalization()(bridge)
    bridge = layers.Activation('relu')(bridge)
    bridge = efficient_attention(bridge, num_heads=8, key_dim=128)
    
    def memory_efficient_decoder_block(x, skip, filters):
        """Memory-efficient decoder block"""
        x = layers.Conv2DTranspose(filters, 2, strides=2)(x)
        
        # Efficient attention for skip connection
        skip_att = efficient_attention(skip, num_heads=2, key_dim=filters//4)
        
        # Combine features
        x = layers.Concatenate()([x, skip_att])
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    # Decoder path
    dec4 = memory_efficient_decoder_block(bridge, conv4, 512)
    dec3 = memory_efficient_decoder_block(dec4, conv3, 256)
    dec2 = memory_efficient_decoder_block(dec3, conv2, 128)
    dec1 = memory_efficient_decoder_block(dec2, conv1, 64)
    
    # Output
    outputs = layers.Conv2D(32, 3, padding='same', activation='relu')(dec1)
    outputs = layers.Conv2D(16, 3, padding='same', activation='relu')(outputs)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(outputs)
    
    return Model(inputs, outputs)


def create_old_unet_model(config):
    """Create U-Net model with attention mechanism"""
    input_size = config['data']['input_size']
    channels = config['data']['channels']
    
    inputs = layers.Input(shape=(input_size, input_size, channels))
    
    # Encoder
    def encoder_block(x, filters, kernel_size=3):
        conv = layers.Conv2D(filters, kernel_size, padding="same")(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("relu")(conv)
        conv = layers.Conv2D(filters, kernel_size, padding="same")(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("relu")(conv)
        pool = layers.MaxPooling2D((2, 2))(conv)
        return conv, pool

    # Attention mechanism
    def attention_block(x, skip_features):
        g1 = layers.Conv2D(x.shape[-1], 1)(x)
        x1 = layers.Conv2D(x.shape[-1], 1)(skip_features)
        psi = layers.Activation('relu')(g1 + x1)
        psi = layers.Conv2D(1, 1)(psi)
        psi = layers.Activation('sigmoid')(psi)
        return layers.multiply([skip_features, psi])

    # Encoder path
    conv1, pool1 = encoder_block(inputs, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)

    # Bridge
    conv5 = layers.Conv2D(1024, 3, padding="same")(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = layers.Conv2D(1024, 3, padding="same")(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)

    # Decoder
    def decoder_block(x, skip_features, filters, kernel_size=3):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
        attention = attention_block(x, skip_features)
        x = layers.Concatenate()([x, attention])
        x = layers.Conv2D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    # Decoder path
    dec4 = decoder_block(conv5, conv4, 512)
    dec3 = decoder_block(dec4, conv3, 256)
    dec2 = decoder_block(dec3, conv2, 128)
    dec1 = decoder_block(dec2, conv1, 64)

    # Output
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(dec1)
    
    model = Model(inputs, outputs)
    return model


# Custom attention mechanisms and modules
def spatial_attention(input_tensor):
    """Add spatial attention module after a feature map"""
    # Compute channel-wise average and max then concatenate
    avg_pool = tf.reduce_mean(input_tensor, axis=3, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=3, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=3)
    
    # Apply convolution to generate attention map
    attention_map = layers.Conv2D(1, kernel_size=7, padding='same')(concat)
    attention_map = layers.Activation('sigmoid')(attention_map)
    
    # Apply attention to input
    return input_tensor * attention_map

def channel_attention(input_tensor, ratio=16):
    """Squeeze-and-excitation channel attention"""
    channels = input_tensor.shape[-1]
    
    # Squeeze operation (global average pooling)
    squeeze = layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation operation (two fully-connected layers)
    excitation = layers.Dense(channels // ratio, activation='relu')(squeeze)
    excitation = layers.Dense(channels, activation='sigmoid')(excitation)
    
    # Reshape for multiplication
    excitation = layers.Reshape((1, 1, channels))(excitation)
    
    # Scale the input tensor
    scale = input_tensor * excitation
    
    return scale

def attention_gate(x, g, filters):
    """Attention gate for skip connections in U-Net"""
    g1 = layers.Conv2D(filters, kernel_size=1)(g)
    g1 = layers.BatchNormalization()(g1)
    
    x1 = layers.Conv2D(filters, kernel_size=1)(x)
    x1 = layers.BatchNormalization()(x1)
    
    psi = layers.Activation('relu')(g1 + x1)
    psi = layers.Conv2D(1, kernel_size=1)(psi)
    psi = layers.Activation('sigmoid')(psi)
    
    return x * psi

def pyramid_pooling_module(input_tensor):
    """Pyramid Pooling Module for multi-scale feature extraction"""
    channels = input_tensor.shape[-1]
    h, w = tf.shape(input_tensor)[1], tf.shape(input_tensor)[2]
    
    pool_sizes = [1, 2, 4, 8]
    pooled_features = []
    
    for pool_size in pool_sizes:
        # Pooling with appropriate size
        x = layers.AveragePooling2D(pool_size=pool_size)(input_tensor)
        # 1x1 conv to reduce channels
        x = layers.Conv2D(channels // 4, 1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        # Bilinear upsampling to original size
        x = tf.image.resize(x, (h, w))
        pooled_features.append(x)
    
    # Concatenate with the input tensor
    x = layers.Concatenate()(pooled_features + [input_tensor])
    x = layers.Conv2D(channels, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def edge_aware_module(input_tensor):
    """Extract edge features to enhance boundary detection"""
    # Create separate edge detection for multi-channel input
    channels = input_tensor.shape[-1]
    edge_features = []
    
    # Edge detection using separable convolution for efficiency
    depth_kernel = tf.constant([
        [1.0, 1.0, 1.0],
        [1.0, -8.0, 1.0],
        [1.0, 1.0, 1.0]
    ], dtype=tf.float32)
    depth_kernel = tf.reshape(depth_kernel, [3, 3, 1, 1])
    
    # Process each channel independently
    channel_tensors = tf.split(input_tensor, num_or_size_splits=channels, axis=-1)
    for channel in channel_tensors:
        edge = tf.nn.depthwise_conv2d(
            channel, 
            tf.tile(depth_kernel, [1, 1, 1, 1]), 
            strides=[1, 1, 1, 1], 
            padding='SAME'
        )
        edge_features.append(edge)
    
    edge_tensor = tf.concat(edge_features, axis=-1)
    
    # Combine edge features with original input
    combined = tf.concat([input_tensor, edge_tensor], axis=-1)
    output = layers.Conv2D(channels, 1, padding='same')(combined)
    output = layers.BatchNormalization()(output)
    output = layers.Activation('relu')(output)
    
    return output

# Enhanced adapter for 4-channel input
def build_enhanced_adapter(input_shape=(None, None, 4)):
    """Enhanced adapter that better preserves NIR information"""
    inp = layers.Input(shape=input_shape, name='adapter_input')
    
    # Split RGB and NIR
    rgb = layers.Lambda(lambda x: x[..., :3])(inp)
    nir = layers.Lambda(lambda x: x[..., 3:4])(inp)
    
    # Process NIR with more expressive layers
    nir_features = layers.Conv2D(16, (3,3), padding='same', activation='relu')(nir)
    nir_features = layers.BatchNormalization()(nir_features)
    nir_features = layers.Conv2D(16, (3,3), padding='same', activation='relu')(nir_features)
    nir_features = layers.BatchNormalization()(nir_features)
    
    # Create NIR-informed RGB channels
    nir_to_rgb = layers.Conv2D(3, (1,1), padding='same')(nir_features)
    nir_to_rgb = layers.BatchNormalization()(nir_to_rgb)
    nir_to_rgb = layers.Activation('sigmoid')(nir_to_rgb)
    
    # Combine RGB with NIR-derived features using gating
    combined = layers.Add()([rgb * nir_to_rgb, rgb])
    combined = layers.Conv2D(3, (1,1), padding='same')(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Activation('relu')(combined)
    
    return Model(inp, combined, name='enhanced_4to3_adapter')



# Improved U-Net implementation with attention and other enhancements
def build_enhanced_unet(input_size, num_classes=1, backbone_name='efficientnetb6',
                        use_adapter=True, adapter_type='enhanced', freeze_backbone=True):
    """
    Build an enhanced U-Net with attention mechanisms and advanced features

    Args:
        input_size: int, height/width of input patches
        num_classes: int, number of output classes (1 for binary segmentation)
        backbone_name: str, name of the backbone model
        use_adapter: bool, if True maps 4→3 channels before feeding encoder
        adapter_type: str, 'simple' or 'enhanced'
        freeze_backbone: bool, if True encoder weights are frozen for stage-1 training

    Returns: tf.keras.Model
    """
    if use_adapter:
        if adapter_type == 'enhanced':
            adapter = build_enhanced_adapter((input_size, input_size, 4))
        else:
            adapter = build_preprocess_adapter((input_size, input_size, 4))
        inp = layers.Input((input_size, input_size, 4), name='input_4ch')
        x = adapter(inp)
    else:
        inp = layers.Input((input_size, input_size, 3), name='input_3ch')
        x = inp

    # Initialize the backbone model with pretrained weights
    backbone = sm.Unet(
        backbone_name=backbone_name,
        input_shape=x.shape[1:],
        encoder_weights='imagenet',
        encoder_freeze=freeze_backbone,
        classes=num_classes,
        activation='sigmoid' if num_classes == 1 else 'softmax'
    )

    # Apply base U-Net to get features
    base_output = backbone(x)

    # Access the internal layers of the backbone for enhancement
    # This requires understanding the backbone model structure
    # Get the encoder and decoder features from the backbone
    encoder_features = []
    decoder_features = []

    # Extract encoder and decoder features
    # Note: The exact layer names will depend on the segmentation_models implementation
    # The following is based on common architectures but may need adjustment
    for layer in backbone.layers:
        if 'encoder' in layer.name and 'features' in layer.name:
            encoder_features.append(layer.output)
        elif 'decoder' in layer.name and 'stage' in layer.name:
            decoder_features.append(layer.output)

    # If we couldn't extract features via layer names, we'll need an alternative approach
    # For example, if the backbone has a different layer naming convention
    if len(encoder_features) == 0 or len(decoder_features) == 0:
        # Alternative: access through model's get_layer method with known layer names
        # This is just an example and would need to be adapted to actual model architecture
        try:
            # Try to get encoder blocks directly - names depend on backbone implementation
            encoder_names = [f'encoder_stage{i}' for i in range(1, 6)]  # Typical 5-stage encoder
            decoder_names = [f'decoder_stage{i}' for i in range(1, 6)]  # Typical 5-stage decoder

            for name in encoder_names:
                try:
                    layer = backbone.get_layer(name)
                    encoder_features.append(layer.output)
                except:
                    pass

            for name in decoder_names:
                try:
                    layer = backbone.get_layer(name)
                    decoder_features.append(layer.output)
                except:
                    pass
        except:
            # If we can't get encoder/decoder features, we'll enhance just the output
            enhanced_output = spatial_attention(base_output)
            enhanced_output = channel_attention(enhanced_output)

            # Create the final model with input and output
            model = Model(inputs=inp, outputs=enhanced_output, name=f'enhanced_{backbone_name}')
            return model

    # Apply spatial attention to selected encoder features
    enhanced_encoder_features = []
    for feature in encoder_features[-3:]:  # Apply to deeper layers
        enhanced = spatial_attention(feature)
        enhanced = channel_attention(enhanced)
        enhanced_encoder_features.append(enhanced)

    # Apply pyramid pooling to the bottleneck
    if len(encoder_features) > 0:
        bottleneck = encoder_features[-1]
        enhanced_bottleneck = pyramid_pooling_module(bottleneck)

    # Apply edge-aware module to intermediate decoder features
    enhanced_decoder_features = []
    for i, feature in enumerate(decoder_features[1:3] if len(decoder_features) >= 3 else decoder_features):
        enhanced = edge_aware_module(feature)
        enhanced_decoder_features.append(enhanced)

    # Apply final enhancements to the output
    enhanced_output = spatial_attention(base_output)
    enhanced_output = edge_aware_module(enhanced_output)

    # Create the final model with input and output
    model = Model(inputs=inp, outputs=enhanced_output, name=f'enhanced_{backbone_name}')

    return model


def build_preprocess_adapter(input_shape=(None, None, 4)):
    """
    1×1 conv adapter to map 4-channel input into 3 channels for a standard pretrained encoder.
    """
    inp = layers.Input(shape=input_shape, name='adapter_input')
    x = layers.Conv2D(3, (1,1), padding='same', name='adapter_conv')(inp)
    return Model(inp, x, name='5to3_adapter')


def create_ensemble_model(input_size, num_classes, backbones=['efficientnetb7', 'seresnext101'], freeze_backbone=True):
    """Create an ensemble of models with different backbones"""
    
    # Create input layer
    if num_classes == 4:
        inputs = layers.Input((input_size, input_size, 4))
        adapter = build_preprocess_adapter((input_size, input_size, 4))
        x = adapter(inputs)
    else:
        inputs = layers.Input((input_size, input_size, 3))
        x = inputs
    
    # Create individual models
    models_outputs = []
    for backbone in backbones:
        unet = sm.Unet(
            backbone_name=backbone,
            input_shape=x.shape[1:],
            encoder_weights='imagenet',
            encoder_freeze=freeze_backbone,  # Added this parameter
            classes=num_classes,
            activation='sigmoid' if num_classes == 1 else 'softmax'
        )
        
        if num_classes == 4:
            model_out = unet(x)
        else:
            model_out = unet(inputs)
        
        models_outputs.append(model_out)
    
    # Average predictions
    ensemble_output = tf.keras.layers.Average()(models_outputs)
    
    # Create ensemble model
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=ensemble_output)
    return ensemble_model

def build_unet_with_encoder(input_size,num_classes,
                        use_adapter=False,
                        freeze_backbone=True):
    """
    Build a U-Net with a encoder (ImageNet weights).
    Args:
      input_size: int, height/width of input patches
      use_adapter: bool, if True maps 4→3 channels before feeding encoder
      freeze_backbone: bool, if True encoder weights are frozen for stage-1 training
    Returns: tf.keras.Model
    """
    if use_adapter:
        adapter = build_preprocess_adapter((input_size, input_size, 5))
        inp = layers.Input((input_size, input_size, 5), name='input_5ch')
        x = adapter(inp)
    else:
        inp = layers.Input((input_size, input_size, 3), name='input_3ch')
        x = inp

    if num_classes == 1:
        unet = sm.Unet(
            backbone_name='senet154',
            #backbone_name='seresnext101',
            # backbone_name='efficientnetb1',
            input_shape=x.shape[1:],
            encoder_weights='imagenet',
            encoder_freeze=freeze_backbone,
            classes=1,
            activation='sigmoid'
        )
    else:
        unet = sm.Unet(
        backbone_name='efficientnetb4',
        input_shape=x.shape[1:],
        encoder_weights='imagenet',
        encoder_freeze=freeze_backbone,
        classes=num_classes,  # Changed from 1 to num_classes
        activation='softmax'  # Changed from 'sigmoid' to 'softmax' for multi-class
    )

    if use_adapter:
        out = unet(x)
        model = Model(inputs=inp, outputs=out, name='unet_resnet50_adapter')
    else:
        model = unet

    return model

def build_unet_dt_resnet50(input_size, 
                        dual_output=True,
                        use_adapter=False,
                        freeze_backbone=True,
                        backbone_name='senet154'):
    """
    Build a U-Net with a pre-trained encoder (ImageNet weights).
    
    Args:
      input_size: int, height/width of input patches
      dual_output: bool, if True produces both binary mask and distance transform outputs
      use_adapter: bool, if True maps 4→3 channels before feeding encoder
      freeze_backbone: bool, if True encoder weights are frozen for stage-1 training
      backbone_name: str, name of the backbone to use ('senet154', 'efficientnetb4', etc.)
    
    Returns: tf.keras.Model
    """
    import segmentation_models as sm
    from tensorflow.keras import layers, Model
    
    # Set up the input
    if use_adapter:
        adapter = build_preprocess_adapter((input_size, input_size, 5))
        inp = layers.Input((input_size, input_size, 5), name='input_5ch')
        x = adapter(inp)
    else:
        inp = layers.Input((input_size, input_size, 3), name='input_3ch')
        x = inp
    
    # Create the base U-Net model
    unet = sm.Unet(
        backbone_name=backbone_name,
        input_shape=x.shape[1:],
        encoder_weights='imagenet',
        encoder_freeze=freeze_backbone,
        classes=1,
        activation=None,  # No activation - we'll add it separately for each output
        decoder_filters=(256, 128, 64, 32, 16)  # Customize decoder size if needed
    )
    
    # Get the decoder output before activation
    decoder_output = unet.layers[-2].output  # Get the output before the final Conv2D
    
    if dual_output:
        # Create two separate output branches
        
        # Branch 1: Binary segmentation output
        binary_features = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_output)
        binary_output = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name='binary_output')(binary_features)
        
        # Branch 2: Distance transform output
        dist_features = layers.Conv2D(32, 3, padding='same', activation='relu')(decoder_output)
        dist_output = layers.Conv2D(1, 1, padding='same', activation='linear', name='distance_output')(dist_features)
        
        # Create model with dual outputs
        if use_adapter:
            model = Model(inputs=inp, outputs=[binary_output, dist_output], name='dual_output_unet')
        else:
            model = Model(inputs=unet.input, outputs=[binary_output, dist_output], name='dual_output_unet')
    else:
        # Single output case (just binary segmentation)
        binary_output = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(decoder_output)
        
        if use_adapter:
            model = Model(inputs=inp, outputs=binary_output, name='single_output_unet')
        else:
            model = Model(inputs=unet.input, outputs=binary_output, name='single_output_unet')
    
    return model

def build_unetplusplus_resnet50(input_size, num_classes,
                                use_adapter=False,
                                freeze_backbone=True):
    """
    Build a Unet++ model with a ResNet50 encoder (ImageNet weights).
    
    Args:
        input_size (int): Height and width of the input images.
        num_classes (int): Number of output classes.
        use_adapter (bool): If True, maps 5→3 channels before feeding encoder.
        freeze_backbone (bool): If True, encoder weights are frozen for initial training.
        
    Returns:
        tf.keras.Model: Compiled Unet++ model.
    """
    # Determine input channels
    input_channels = 5 if use_adapter else 3
    input_shape = (input_size, input_size, input_channels)
    inp = layers.Input(shape=input_shape, name='input')
    
    # Apply adapter if needed
    if use_adapter:
        x = build_preprocess_adapter(input_shape)(inp)
    else:
        x = inp
    
    # Define filter numbers for each level
    filter_num = [64, 128, 256, 512, 1024]
    
    # Set activation functions - keras_unet_collection expects activation
    # function names with capital letters as they refer to Layer classes
    activation = 'ReLU'  # Note the capitalization! Refers to the Layer class
    output_activation = 'Sigmoid' if num_classes == 1 else 'Softmax'
    
    # Build Unet++ model
    model = models.unet_plus_2d(
        input_size=x.shape[1:],  # Exclude batch size
        filter_num=filter_num,
        n_labels=num_classes,
        stack_num_down=2,
        stack_num_up=2,
        activation=activation,
        output_activation=output_activation,
        batch_norm=True,
        pool=True,
        unpool=True,
        backbone='EfficientNetB7',
        weights='imagenet',
        freeze_backbone=freeze_backbone,
        name='unetplusplus_resnet50'
    )
    
    # Create final model
    final_model = Model(inputs=inp, outputs=model(x), name='unetplusplus_resnet50_adapter' if use_adapter else 'unetplusplus_resnet50')
    return final_model


def create_unet_model(config):
    """Create enhanced U-Net model with additional attention mechanisms and preprocessing"""
    input_size = config['data']['input_size']
    channels = config['data']['channels']

    inputs = layers.Input(shape=(input_size, input_size, channels))

    # Add input normalization
    x = layers.Lambda(lambda x: x/255.0)(inputs)

    # Channel Attention Module (SE block)
    def channel_attention(inputs, ratio=8):
        channel = inputs.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
        avg_pool = layers.Dense(channel // ratio, activation='relu',
                              kernel_initializer='he_normal')(avg_pool)
        avg_pool = layers.Dense(channel, activation='sigmoid',
                              kernel_initializer='he_normal')(avg_pool)
        return layers.multiply([inputs, avg_pool])

    # Spatial Attention Module
    def spatial_attention(inputs):
        avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(inputs)
        max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(inputs)
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        spatial = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
        return layers.multiply([inputs, spatial])

    def se_resnext_block(x, filters):
        residual = x
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
    
        # SE block
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters//16, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        x = layers.Multiply()([x, se])
    
        # Residual connection
        if residual.shape[-1] != filters:
            residual = layers.Conv2D(filters, 1)(residual)
        x = layers.Add()([x, residual])
        return x

    # Enhanced encoder block with dual attention
    def encoder_block(x, filters, kernel_size=3):
        conv = layers.Conv2D(filters, kernel_size, padding="same")(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("relu")(conv)
        conv = layers.Conv2D(filters, kernel_size, padding="same")(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation("relu")(conv)

        # Add channel and spatial attention
        #conv = se_resnext_block(x, filters)
        conv = channel_attention(conv)
        conv = spatial_attention(conv)

        pool = layers.MaxPooling2D((2, 2))(conv)
        return conv, pool

    # Enhanced attention mechanism for skip connections
    def attention_gate(g, x):
        filters = x.shape[-1]
        g_conv = layers.Conv2D(filters, 1, padding='same')(g)
        x_conv = layers.Conv2D(filters, 1, padding='same')(x)
        act = layers.Activation('relu')(g_conv + x_conv)
        psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(act)
        return layers.multiply([x, psi])

    # Enhanced decoder block
    def decoder_block(x, skip_features, filters, kernel_size=3):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
        # Enhanced attention mechanism
        attention = attention_gate(x, skip_features)
        x = layers.Concatenate()([x, attention])

        x = layers.Conv2D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Add channel and spatial attention to decoder output
        x = channel_attention(x)
        x = spatial_attention(x)

        return x

    # Encoder path
    conv1, pool1 = encoder_block(x, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)

    # Bridge
    conv5 = layers.Conv2D(1024, 3, padding="same")(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)
    conv5 = layers.Conv2D(1024, 3, padding="same")(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation("relu")(conv5)

    # Apply attention to bridge
    conv5 = channel_attention(conv5)
    conv5 = spatial_attention(conv5)

    # Decoder path
    dec4 = decoder_block(conv5, conv4, 512)
    dec3 = decoder_block(dec4, conv3, 256)
    dec2 = decoder_block(dec3, conv2, 128)
    dec1 = decoder_block(dec2, conv1, 64)

    # Progressive refinement in output layers
    outputs = layers.Conv2D(32, 3, padding='same', activation='relu')(dec1)
    outputs = layers.Conv2D(16, 3, padding='same', activation='relu')(outputs)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(outputs)
    #num_classes = config['data']['channels']
    #outputs = layers.Conv2D(num_classes, 1, activation='softmax')(outputs)

    model = Model(inputs, outputs)
    return model



def get_callbacks(config):
    """Get training callbacks"""
    callbacks = [
        ModelCheckpoint(
            filepath=f"{config['paths']['model_save']}/best_model.h5",
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['training']['reduce_lr_patience'],
            min_lr=config['training']['min_lr'],
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config['paths']['logs'],
            histogram_freq=1
        )
    ]
    return callbacks



__all__ = [
    'Channel4_DataGenerator',
    'Channel3_DataGenerator',
    'build_preprocess_adapter',
    'build_unet_resnet50',
    'create_unet_model',
    'create_enhanced_unet',
    'create_old_unet_model',
]
