import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from skimage import exposure
import segmentation_models_pytorch as smp


class Channel3_DataGenerator_old(Dataset):
    """Custom data generator adapted for preprocessed uint8 RGB images"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        idx = self.indexes[index]
        
        # Load preprocessed RGB image (already scaled to uint8)
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()
            # Simple scaling to 0-1 since images are already preprocessed
            image = image.astype(np.float32) / 255.0
            
        # Load and preprocess mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)
            mask = (mask > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)
            
        return torch.from_numpy(image), torch.from_numpy(mask)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class Channel4_DataGenerator(Dataset):
    """Custom data generator for loading and preprocessing satellite imagery"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']
        self.classes = config['data']['num_classes']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        idx = self.indexes[index]
        
        # Load and preprocess image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()
            # Normalize each channel
            for j in range(src.count):
                channel = image[j]
                min_val = np.percentile(channel, 2)
                max_val = np.percentile(channel, 98)
                image[j] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
            image = image.astype(np.float32)

        # Load and preprocess mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)
            mask = mask.astype(np.float32)
            mask = np.expand_dims(mask, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class Channel4_MultiDataGenerator(Dataset):
    """Custom data generator for loading and preprocessing satellite imagery"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']
        self.classes = config['data']['num_classes']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        idx = self.indexes[index]
        
        # Load and preprocess image
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read()
            # Normalize each channel
            for j in range(src.count):
                channel = image[j]
                min_val = np.percentile(channel, 2)
                max_val = np.percentile(channel, 98)
                image[j] = np.clip((channel - min_val) / (max_val - min_val), 0, 1)
            image = image.astype(np.float32)

        # Load and preprocess mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)
            # for multiclass
            mask_one_hot = np.zeros((self.classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
            for c in range(self.classes):
                mask_one_hot[c] = (mask == c).astype(np.float32)

        return torch.from_numpy(image), torch.from_numpy(mask_one_hot)

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class PreprocessAdapter(nn.Module):
    """1×1 conv adapter to map 4-channel input into 3 channels for a standard pretrained encoder."""
    def __init__(self, in_channels=5, out_channels=3):
        super(PreprocessAdapter, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        return self.conv(x)


class UNetWithAdapter(nn.Module):
    """U-Net with optional channel adapter"""
    def __init__(self, unet_model, adapter=None):
        super(UNetWithAdapter, self).__init__()
        self.adapter = adapter
        self.unet = unet_model
        
    def forward(self, x):
        if self.adapter is not None:
            x = self.adapter(x)
        return self.unet(x)


def build_unet_resnet50(input_size, num_classes, use_adapter=False, freeze_backbone=True):
    """
    Build a U-Net with a ResNet50 encoder (ImageNet weights).
    Args:
      input_size: int, height/width of input patches
      use_adapter: bool, if True maps 4→3 channels before feeding encoder
      freeze_backbone: bool, if True encoder weights are frozen for stage-1 training
    Returns: PyTorch model
    """
    
    if num_classes == 1:
        # For binary segmentation
        # unet = smp.Unet(
        #     encoder_name='senet154',  # or 'seresnext101', 'efficientnet-b7'
        #     encoder_weights='imagenet',
        #     in_channels=3,
        #     classes=1,
        #     activation='sigmoid'  # We'll apply sigmoid in the loss or during inference
        # )

        unet = smp.UnetPlusPlus(
            encoder_name='senet154',  # or 'seresnext101', 'efficientnet-b7'
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            decoder_attention_type="scse",
            decoder_channels=(256, 128, 64, 32, 16),
            activation='sigmoid'  # We'll apply sigmoid in the loss or during inference
        )

    else:
        # For multi-class segmentation
        unet = smp.Unet(
            encoder_name='efficientnet-b4',
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes,
            activation=None  # We'll apply softmax in the loss or during inference
        )
    
    # Freeze encoder if requested
    if freeze_backbone:
        for param in unet.encoder.parameters():
            param.requires_grad = False
    
    if use_adapter:
        adapter = PreprocessAdapter(in_channels=5, out_channels=3)
        model = UNetWithAdapter(unet, adapter)
    else:
        model = unet
    
    return model


def get_callbacks(config, model_path, timestamp):
    """PyTorch doesn't have built-in callbacks like Keras, but we'll create equivalent functionality"""
    # This would be implemented in the training loop
    # Returning a dict of callback configurations for now
    return {
        'checkpoint': {
            'filepath': f"{model_path}/best_model_{timestamp}.pt",
            'monitor': 'val_loss',
            'mode': 'min',
            'save_best_only': True
        },
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': config['training']['early_stopping_patience'],
            'restore_best_weights': True
        },
        'reduce_lr': {
            'monitor': 'val_loss',
            'factor': 0.5,
            'patience': config['training']['reduce_lr_patience'],
            'min_lr': config['training']['min_lr']
        }
    }


# Placeholder functions for compatibility
def build_enhanced_unet(*args, **kwargs):
    """Placeholder for enhanced unet"""
    return build_unet_resnet50(*args, **kwargs)


def create_unet_model(*args, **kwargs):
    """Placeholder for create_unet_model"""
    return build_unet_resnet50(*args, **kwargs)


def create_ensemble_model(*args, **kwargs):
    """Placeholder for create_ensemble_model"""
    return build_unet_resnet50(*args, **kwargs)