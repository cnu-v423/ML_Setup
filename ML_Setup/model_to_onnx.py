import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import rasterio
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K 
from src.backbone_model import build_unet_resnet50
import tf2onnx
import onnx
import yaml

class Channel4_DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for loading and preprocessing satellite imagery"""
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
        batch_masks = npaimldl.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load and preprocess image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
                # Normalize each channel
                for j in range(self.channels):
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

class Channel3_DataGenerator(tf.keras.utils.Sequence):
    """Custom data generator adapted for preprocessed uint8 RGB images"""
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
        # This is the required implementation
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.image_paths))
        batch_indexes = self.indexes[start_idx:end_idx]

        batch_images = np.empty((len(batch_indexes), self.input_size, self.input_size, self.channels))
        batch_masks = np.empty((len(batch_indexes), self.input_size, self.input_size, 1))

        for i, idx in enumerate(batch_indexes):
            # Load preprocessed RGB image (already scaled to uint8)
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()
            
                # Specialized normalization
                for j in range(4):  # First 4 bands (RGB + additional)
                    channel = image[j]
                    min_val = np.percentile(channel, 2)
                    max_val = np.percentile(channel, 98)
                    image[j] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
            
                # NDVI band (5th band) is already in -1 to 1 range
                # Just ensure it's within the range and normalize if needed
                ndvi = image[4]
                image[4] = np.clip(ndvi, -1, 1)
            
                # Transpose to match expected shape
                image = np.moveaxis(image, 0, -1)
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

def create_unet_model():
    with open('config.yaml', 'r') as f:
        print("Config....")
        config = yaml.safe_load(f)
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
    
#model = create_unet_model()
model = build_unet_resnet50(
        num_classes = 1, #config['data']['num_classes'],
        input_size=256,
        freeze_backbone=False
    )

model.load_weights("building_pretraied_rgb_bgr_senet154_v1/best_model_1747031895.h5")
model.summary()
onnx_model, _ = tf2onnx.convert.from_keras(model)
onnx.save_model(onnx_model, "unet_senet154.onnx")
print("Converted")


