from skimage import exposure
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import rasterio
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K  
from skimage import exposure
from scipy import ndimage
from typing import Dict, Any


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
            
def minMax(arr):
    """Min-Max normalize the array to [0, 1]."""
    min_val = np.min(arr)
    max_val = np.max(arr)
    if max_val - min_val != 0:
        return (arr - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(arr, dtype=np.float32)

class Channel4_DataGenerator_latest(tf.keras.utils.Sequence):
    """Custom data generator for RGB + Sobel (4 channels) with manual grayscale conversion and Min-Max normalization."""

    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = config['data']['batch_size']
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']  # Fixed for RGB + Sobel
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def minMax(self,arr):
        """Min-Max normalize the array to [0, 1]."""
        min_val = np.min(arr)
        max_val = np.max(arr)
        if max_val - min_val != 0:
            return (arr - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(arr, dtype=np.float32)

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, self.channels), dtype=np.float32)
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1), dtype=np.float32)

        for i, idx in enumerate(batch_indexes):
            # Load R, G, B, NIR bands
            with rasterio.open(self.image_paths[idx]) as src:
                arr = src.read([1, 2, 3])  
                arr = np.moveaxis(arr, 0, -1)  # Shape: (H, W, 4)

            # Extract RGB channels and normalize them
            r = self.minMax(arr[:, :, 0])
            g = self.minMax(arr[:, :, 1])
            b = self.minMax(arr[:, :, 2])

            # Compute grayscale from normalized RGB
            grayscale = (r + g + b) / 3.0

            # Compute Sobel edge detection on grayscale
            sobelx = ndimage.sobel(grayscale, axis=0)
            sobely = ndimage.sobel(grayscale, axis=1)
            sobel = np.hypot(sobelx, sobely)
            sobel_norm = self.minMax(sobel)

            # Stack normalized R, G, B + Sobel as 4 channels
            combined = np.stack((r, g, b, sobel_norm), axis=-1)

            batch_images[i] = combined

            # Load and preprocess mask (binary mask)
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1)
                mask = np.expand_dims(mask, axis=-1).astype(np.float32)
                batch_masks[i] = mask

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

def create_enhanced_unet(config):
    input_size = config['data']['input_size']
    channels = config['data']['channels']

    inputs = layers.Input(shape=(input_size, input_size, channels))

    # Enhanced Spatial Attention Block
    def spatial_attention_block(x):
        # Channel-wise attention
        channel_avg = layers.GlobalAveragePooling2D()(x)
        channel_max = layers.GlobalMaxPooling2D()(x)
        channel_attention = layers.Dense(x.shape[-1] // 2, activation='relu')(channel_avg)
        channel_attention = layers.Dense(x.shape[-1], activation='sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, x.shape[-1]))(channel_attention)
        x = layers.Multiply()([x, channel_attention])

        # Spatial attention
        spatial_avg = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x)
        spatial_max = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x)
        spatial_concat = layers.Concatenate(axis=-1)([spatial_avg, spatial_max])
        spatial_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(spatial_concat)
        x = layers.Multiply()([x, spatial_attention])

        return x

    # Enhanced Encoder Block
    def encoder_block(x, filters, dropout_rate=0.25):
        # Deeper convolution with batch normalization
        conv = layers.Conv2D(filters, (3, 3), padding='same')(x)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters, (3, 3), padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)

        # Spatial attention
        conv = spatial_attention_block(conv)

        # Pooling with dropout
        pool = layers.MaxPooling2D((2, 2))(conv)
        pool = layers.Dropout(dropout_rate)(pool)

        return conv, pool

    # Encoder Path with Progressive Depth
    conv1, pool1 = encoder_block(inputs, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)
    conv5, pool5 = encoder_block(pool4, 1024)

    # Bridge
    bridge = layers.Conv2D(2048, (3, 3), padding='same')(pool5)
    bridge = layers.BatchNormalization()(bridge)
    bridge = layers.Activation('relu')(bridge)
    bridge = layers.Conv2D(2048, (3, 3), padding='same')(bridge)
    bridge = layers.BatchNormalization()(bridge)
    bridge = layers.Activation('relu')(bridge)

    # Decoder Block with Skip Connections
    def decoder_block(x, skip, filters, dropout_rate=0.25):
        # Upsampling
        up = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        
        # Concatenate with skip connection
        merge = layers.Concatenate()([up, skip])
        
        # Convolution
        conv = layers.Conv2D(filters, (3, 3), padding='same')(merge)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters, (3, 3), padding='same')(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation('relu')(conv)
        
        # Spatial attention
        conv = spatial_attention_block(conv)
        
        # Dropout
        conv = layers.Dropout(dropout_rate)(conv)
        
        return conv

    # Decoder Path
    dec5 = decoder_block(bridge, conv5, 1024)
    dec4 = decoder_block(dec5, conv4, 512)
    dec3 = decoder_block(dec4, conv3, 256)
    dec2 = decoder_block(dec3, conv2, 128)
    dec1 = decoder_block(dec2, conv1, 64)

    # Output Refinement
    outputs = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(dec1)
    outputs = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(outputs)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # Custom Boundary-Aware Loss
    def boundary_aware_loss(y_true, y_pred):
        # Binary Cross-Entropy
        bce = K.binary_crossentropy(y_true, y_pred)

        # Dice Loss
        intersection = K.sum(y_true * y_pred, axis=[1,2,3])
        union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
        dice_loss = 1 - K.mean((2. * intersection + 1) / (union + 1))

        # Edge Continuity Loss
        edges_true = tf.image.sobel_edges(y_true)
        edges_pred = tf.image.sobel_edges(y_pred)
        edge_loss = K.mean(K.abs(edges_true - edges_pred))

        # Weighted Combination
        return 0.4 * bce + 0.4 * dice_loss + 0.2 * edge_loss

    # Compile with custom loss
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=boundary_aware_loss,
        metrics=['accuracy']
    )

    return model

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
