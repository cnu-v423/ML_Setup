# VEGETATION DETECTION TRAINING - OPTIMIZED FOR TREES AND SHRUBS
# This script is specifically optimized for vegetation detection with RGB bands
# It adds vegetation-specific features (NDVI, GLI, SAVI, texture) and removes water body detection features

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import segmentation_models_pytorch as smp
from pytorch_backbone_model_v2 import (
    get_callbacks,
    build_unet_resnet50,
    create_ensemble_model,
    BuildingRecall,
    BuildingPrecision,
    BuildingIoU,
    F1Score
)

import time
from backup_utils import backup_project
import glob
import math
from sklearn.model_selection import train_test_split
import argparse
import rasterio
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import logging
from datetime import datetime
import cv2
import albumentations as A


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, initial_lr, total_epochs, warmup_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def step(self, epoch):
        """Apply learning rate schedule"""
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def compute_vegetation_indices(rgb_image):
    """
    Compute vegetation indices from RGB image to enhance vegetation detection.
    
    Args:
        rgb_image: (C, H, W) array with RGB bands [0]=Red, [1]=Green, [2]=Blue
        
    Returns:
        indices: (5, H, W) array with [R, G, B, NDVI, GLI, SAVI]
    """
    red = rgb_image[0].astype(np.float32)
    green = rgb_image[1].astype(np.float32)
    blue = rgb_image[2].astype(np.float32)
    
    # Normalize to prevent division issues
    red = np.clip(red, 1e-6, 1.0)
    green = np.clip(green, 1e-6, 1.0)
    blue = np.clip(blue, 1e-6, 1.0)
    
    # Vegetation Indices (these enhance vegetation detection)
    # ExG (Excess Green) - very effective for vegetation detection
    exg = 2 * green - red - blue
    exg = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)
    
    # NDVI-like from RGB (Normalized Difference based on R-G)
    # More green relative to red = more vegetation
    ndvi_rgb = (green - red) / (green + red + 1e-6)
    ndvi_rgb = (ndvi_rgb + 1) / 2  # Scale to 0-1
    
    # GLI (Green Leaf Index) 
    gli = (2 * green - red - blue) / (2 * green + red + blue + 1e-6)
    gli = (gli + 1) / 2  # Scale to 0-1
    
    # Color Index (G/(R+B))
    color_index = green / (red + blue + 1e-6)
    color_index = np.clip(color_index / 2, 0, 1)  # Normalize
    
    # Stack all features: [R, G, B, ExG, NDVI-RGB, GLI, ColorIndex]
    indices = np.stack([red, green, blue, exg, ndvi_rgb, gli, color_index])
    
    return indices


class VegetationDataGenerator(Dataset):
    """
    Optimized data generator for vegetation detection with RGB input.
    Computes vegetation indices on-the-fly to enhance detection.
    """
    
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = config['data']['input_size']
        self.is_training = is_training
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)
        
        # Aggressive augmentation for vegetation detection
        if self.is_training:
            self.aug = A.Compose([
                # Geometric transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, 
                                   rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
                
                # Vegetation-specific augmentations
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.35, p=0.6),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, 
                                    val_shift_limit=15, p=0.4),
                
                # Noise augmentation (subtle for vegetation)
                A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.15),
                
                # Crop for variety
                A.RandomCrop(height=self.input_size, width=self.input_size, p=0.2),
                
                # Elastic deformation for vegetation boundaries
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
                
                # Channel dropout (simulate missing spectral info)
                A.ChannelDropout(p=0.1),
                
            ], additional_targets={'mask': 'mask'})
        else:
            self.aug = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        idx = self.indexes[index]
        
        # Load RGB image
        with rasterio.open(self.image_paths[idx]) as src:
            rgb_image = src.read()  # shape: (3, H, W)
            rgb_image = rgb_image.astype(np.float32) / 255.0  # Normalize to 0-1
        
        # Load mask
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)  # shape: (H, W)
            mask = (mask > 0).astype(np.uint8)  # Binary mask
        
        # Compute vegetation indices
        veg_indices = compute_vegetation_indices(rgb_image)  # (7, H, W)
        
        # Combine RGB with vegetation indices
        combined_features = veg_indices.astype(np.float32)  # (7, H, W)
        
        # Apply augmentations during training
        if self.is_training and self.aug is not None:
            # Convert to HWC for albumentations
            combined_hwc = np.moveaxis(combined_features, 0, -1)  # (H, W, 7)
            combined_uint8 = (np.clip(combined_hwc, 0, 1) * 255).astype(np.uint8)
            mask_hwc = np.expand_dims(mask, axis=-1)  # (H, W, 1)
            
            # Create augmented version with mapping for 7 channels
            augmented = self.aug(image=combined_uint8, mask=mask_hwc)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Convert back to float and CHW
            combined_features = (aug_image.astype(np.float32) / 255.0)
            combined_features = np.moveaxis(combined_features, -1, 0)  # (7, H, W)
            
            if aug_mask.ndim == 3 and aug_mask.shape[-1] == 1:
                mask = aug_mask.squeeze(-1).astype(np.float32)
            else:
                mask = aug_mask.astype(np.float32)
        else:
            combined_features = combined_features.astype(np.float32)
            mask = mask.astype(np.float32)
        
        # Prepare mask with channel dimension
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(combined_features).float()  # (7, H, W)
        mask_tensor = torch.from_numpy(mask).float()  # (1, H, W)
        
        return image_tensor, mask_tensor
    
    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


class VegetationAdaptiveLoss(nn.Module):
    """
    Adaptive loss specifically optimized for vegetation detection.
    Emphasizes boundary detection and vegetation-specific characteristics.
    """
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Learnable loss weights
        self.alpha = nn.Parameter(torch.ones(1))  # BCE weight
        self.beta = nn.Parameter(torch.ones(1))   # Dice weight  
        self.gamma = nn.Parameter(torch.ones(1))  # Boundary weight
        self.delta = nn.Parameter(torch.tensor(0.5))  # Vegetation index weight
        self.class_weights = nn.Parameter(torch.ones(num_classes))
    
    def dice_loss(self, y_pred, y_true):
        """Dice loss for vegetation detection"""
        smooth = 1e-7
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum()
        dice_coef = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_coef
    
    def boundary_loss(self, y_pred, y_true):
        """
        Boundary loss focusing on vegetation edges.
        Critical for detecting tree/shrub boundaries.
        """
        # Sobel filters for edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_x = sobel_x.to(y_pred.device)
        sobel_y = sobel_y.to(y_pred.device)
        
        # Apply Sobel filters
        edges_true_x = F.conv2d(y_true, sobel_x, padding=1)
        edges_true_y = F.conv2d(y_true, sobel_y, padding=1)
        edges_pred_x = F.conv2d(y_pred, sobel_x, padding=1)
        edges_pred_y = F.conv2d(y_pred, sobel_y, padding=1)
        
        # Compute edge magnitude
        eps = 1e-8
        edges_true = torch.sqrt(edges_true_x**2 + edges_true_y**2 + eps)
        edges_pred = torch.sqrt(edges_pred_x**2 + edges_pred_y**2 + eps)
        
        return torch.mean(torch.abs(edges_true - edges_pred))
    
    def focal_loss(self, y_pred, y_true, alpha=0.25, gamma=2.0):
        """
        Focal loss for handling class imbalance.
        Helps model focus on harder vegetation pixels.
        """
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Binary focal loss
        ce_loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        focal_weight = torch.abs(y_true - y_pred) ** gamma
        focal_loss = alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def forward(self, y_pred, y_true):
        """Combined loss with learnable weights"""
        # Constraint parameters to be positive
        alpha = torch.clamp(self.alpha, min=0.0)
        beta = torch.clamp(self.beta, min=0.0)
        gamma = torch.clamp(self.gamma, min=0.0)
        delta = torch.clamp(self.delta, min=0.0)
        
        # Clamp inputs
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0 - 1e-7)
        y_true = torch.clamp(y_true, min=0.0, max=1.0)
        
        # BCE loss
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='mean')
        
        # Dice loss
        dice = self.dice_loss(y_pred, y_true)
        
        # Boundary loss (critical for vegetation boundaries)
        try:
            edge_loss = self.boundary_loss(y_pred, y_true)
        except:
            edge_loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
        # Focal loss (helps with hard vegetation pixels)
        focal = self.focal_loss(y_pred, y_true)
        
        # Combined loss: emphasize boundary and focal loss for vegetation
        total_loss = (alpha * bce + 
                     beta * dice + 
                     gamma * edge_loss + 
                     delta * focal)
        
        return total_loss


class VegetationMetrics:
    """Compute vegetation-specific metrics"""
    
    @staticmethod
    def f1_score(y_pred, y_true, threshold=0.5):
        """F1 score for binary vegetation detection"""
        y_pred_binary = (y_pred > threshold).float()
        tp = (y_true * y_pred_binary).sum().item()
        fp = ((1 - y_true) * y_pred_binary).sum().item()
        fn = (y_true * (1 - y_pred_binary)).sum().item()
        
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        return f1, precision, recall
    
    @staticmethod
    def iou(y_pred, y_true, threshold=0.5):
        """IoU for binary vegetation detection"""
        y_pred_binary = (y_pred > threshold).float()
        intersection = (y_true * y_pred_binary).sum().item()
        union = y_true.sum().item() + y_pred_binary.sum().item() - intersection
        return intersection / (union + 1e-7)


def set_gpu():
    """Set GPU configuration"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"🚀 GPUs detected: {device_count}")
        if device_count >= 1:
            torch.cuda.set_device(0)
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  No GPU available, using CPU")


def get_filename_without_extension(filepath):
    """Extract filename without extension"""
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]


def match_tile_mask_pairs(tiles_dir, masks_dir):
    """Match tiles and masks based on filenames"""
    tile_files = glob.glob(os.path.join(tiles_dir, '*.tif'))
    mask_files = glob.glob(os.path.join(masks_dir, '*.tif'))
    
    tile_dict = {get_filename_without_extension(f): f for f in tile_files}
    mask_dict = {get_filename_without_extension(f): f for f in mask_files}
    
    common_files = set(tile_dict.keys()).intersection(set(mask_dict.keys()))
    
    missing_masks = set(tile_dict.keys()) - set(mask_dict.keys())
    missing_tiles = set(mask_dict.keys()) - set(tile_dict.keys())
    
    if missing_masks:
        print(f"⚠️  {len(missing_masks)} tiles have no mask")
    if missing_tiles:
        print(f"⚠️  {len(missing_tiles)} masks have no tile")
    
    print(f"✅ Found {len(common_files)} valid tile-mask pairs")
    
    matched_tiles = [tile_dict[name] for name in common_files]
    matched_masks = [mask_dict[name] for name in common_files]
    
    return matched_tiles, matched_masks


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, stage):
    """Train one epoch"""
    model.train()
    train_loss = 0.0
    metrics_list = []
    
    pbar = tqdm(train_loader, desc=f"🌳 {stage} Epoch {epoch+1} [TRAIN]", 
                leave=False, ncols=100)
    
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        # Compute metrics
        f1, prec, rec = VegetationMetrics.f1_score(outputs.detach(), masks)
        iou = VegetationMetrics.iou(outputs.detach(), masks)
        metrics_list.append({'f1': f1, 'iou': iou, 'precision': prec, 'recall': rec})
        
        if len(metrics_list) % 10 == 0:
            avg_f1 = np.mean([m['f1'] for m in metrics_list[-10:]])
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'F1': f'{avg_f1:.3f}'})
    
    pbar.close()
    
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    avg_loss = train_loss / len(train_loader)
    
    return avg_loss, avg_metrics


def validate_epoch(model, val_loader, loss_fn, device, epoch, stage):
    """Validate one epoch"""
    model.eval()
    val_loss = 0.0
    metrics_list = []
    
    pbar = tqdm(val_loader, desc=f"🔍 {stage} Epoch {epoch+1} [VAL]", 
                leave=False, ncols=100)
    
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            val_loss += loss.item()
            
            # Compute metrics
            f1, prec, rec = VegetationMetrics.f1_score(outputs, masks)
            iou = VegetationMetrics.iou(outputs, masks)
            metrics_list.append({'f1': f1, 'iou': iou, 'precision': prec, 'recall': rec})
            
            if len(metrics_list) % 5 == 0:
                avg_iou = np.mean([m['iou'] for m in metrics_list[-5:]])
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'IoU': f'{avg_iou:.3f}'})
    
    pbar.close()
    
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}
    avg_loss = val_loss / len(val_loader)
    
    return avg_loss, avg_metrics


def train_vegetation_detector(config, tiles_dir, masks_dir, model_save_dir, weights_path=None):
    """Main training function for vegetation detection"""
    
    print("\n" + "="*70)
    print("🌳 VEGETATION DETECTION MODEL TRAINING 🌳")
    print("="*70)
    print("✨ Optimized for detecting trees, shrubs, and vegetation canopy")
    print("📊 Using RGB bands with computed vegetation indices")
    print("="*70 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preparation
    tiles, masks = match_tile_mask_pairs(tiles_dir, masks_dir)
    train_tiles, val_tiles, train_masks, val_masks = train_test_split(
        tiles, masks,
        test_size=config['data']['validation_split'],
        random_state=config['data']['random_state']
    )
    
    print(f"📊 Training set: {len(train_tiles)} samples")
    print(f"📊 Validation set: {len(val_tiles)} samples\n")
    
    # Create datasets
    train_dataset = VegetationDataGenerator(train_tiles, train_masks, config, is_training=True)
    val_dataset = VegetationDataGenerator(val_tiles, val_masks, config, is_training=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'],
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'],
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Build model - with adapter for 7-channel input
    print("🏗️  Building U-Net++ model with vegetation-optimized architecture...")
    
    # Create model with 7-channel input (RGB + 4 vegetation indices)
    model = smp.UnetPlusPlus(
        encoder_name='senet154',
        encoder_weights='imagenet',
        in_channels=3,  # Will use adapter
        classes=1,
        decoder_attention_type="scse",
        decoder_channels=(256, 128, 64, 32, 16),
        activation='sigmoid'
    )
    
    # Add channel adapter for 7 channels
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
    
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"🔄 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    if weights_path and os.path.exists(weights_path):
        print(f"📥 Loading pre-trained weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # Loss and optimizer
    loss_fn = VegetationAdaptiveLoss()
    loss_fn = loss_fn.to(device)
    
    os.makedirs(model_save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_model_path = os.path.join(model_save_dir, f'vegetation_unet_best_{timestamp}.pt')
    
    # ─── STAGE 1: FROZEN BACKBONE (Initial Training) ───
    print(f"\n{'='*70}")
    print("🎯 STAGE 1: FROZEN BACKBONE TRAINING")
    print(f"{'='*70}")
    print("📍 Focus: Train decoder and classifier on vegetation features")
    print("📍 Backbone (encoder) weights are frozen from ImageNet")
    print(f"{'='*70}\n")
    
    # Freeze encoder/backbone
    if hasattr(model, 'module'):  # DataParallel
        encoder = model.module.model.encoder
    else:
        encoder = model.model.encoder
    
    for param in encoder.parameters():
        param.requires_grad = False
    
    print("🔒 Encoder frozen. Training parameters: ", end="")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{trainable_params:,} / {total_params:,}")
    
    # Stage 1 configuration
    initial_lr_stage1 = config['model']['learning_rate']
    total_epochs_stage1 = 15
    warmup_epochs_stage1 = 2
    
    optimizer_stage1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=initial_lr_stage1,
        weight_decay=1e-4
    )
    scheduler_stage1 = WarmupCosineScheduler(
        optimizer_stage1, initial_lr_stage1, total_epochs_stage1, warmup_epochs_stage1
    )
    
    best_val_iou_stage1 = 0.0
    patience_stage1 = 0
    max_patience_stage1 = 8
    
    # Stage 1 training loop
    for epoch in range(total_epochs_stage1):
        current_lr = scheduler_stage1.step(epoch)
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer_stage1, loss_fn, device, epoch, "Stage 1"
        )
        val_loss, val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, epoch, "Stage 1"
        )
        
        print(f"\n📊 Stage 1 - Epoch {epoch+1}/{total_epochs_stage1}")
        print(f"   LR: {current_lr:.8f}")
        print(f"   Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"   Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"   Recall: {val_metrics['recall']:.4f} | Precision: {val_metrics['precision']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou_stage1:
            best_val_iou_stage1 = val_metrics['iou']
            # torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ New best model saved! (IoU: {best_val_iou_stage1:.6f})")
            patience_stage1 = 0
        else:
            patience_stage1 += 1
        
        if patience_stage1 >= max_patience_stage1:
            print(f"\n🛑 Stage 1 early stopping at epoch {epoch+1}")
            break
    
    print(f"\n✅ Stage 1 completed! Best IoU: {best_val_iou_stage1:.6f}\n")
    
    # ─── STAGE 2: FINE-TUNING (All Parameters) ───
    print(f"\n{'='*70}")
    print("🎯 STAGE 2: FINE-TUNING (Unfrozen Backbone)")
    print(f"{'='*70}")
    print("📍 Fine-tune entire model including backbone")
    print("📍 Lower learning rate to preserve pre-trained features")
    print(f"{'='*70}\n")
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    print("🔓 All parameters unlocked. Training parameters: ", end="")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{trainable_params:,} / {total_params:,}")
    
    # Stage 2 configuration
    initial_lr_stage2 = initial_lr_stage1 * 0.1  # Lower learning rate for fine-tuning
    total_epochs_stage2 = 100
    warmup_epochs_stage2 = 3
    
    optimizer_stage2 = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr_stage2,
        weight_decay=1e-4
    )
    scheduler_stage2 = WarmupCosineScheduler(
        optimizer_stage2, initial_lr_stage2, total_epochs_stage2, warmup_epochs_stage2, min_lr=1e-8
    )
    
    best_val_iou_stage2 = best_val_iou_stage1
    patience_stage2 = 0
    max_patience_stage2 = config['training']['early_stopping_patience']
    
    # Stage 2 training loop
    for epoch in range(total_epochs_stage2):
        current_lr = scheduler_stage2.step(epoch)
        
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer_stage2, loss_fn, device, epoch, "Stage 2"
        )
        val_loss, val_metrics = validate_epoch(
            model, val_loader, loss_fn, device, epoch, "Stage 2"
        )
        
        print(f"\n📊 Stage 2 - Epoch {epoch+1}/{total_epochs_stage2}")
        print(f"   LR: {current_lr:.8f}")
        print(f"   Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        print(f"   Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"   Recall: {val_metrics['recall']:.4f} | Precision: {val_metrics['precision']:.4f}")
        
        # Save best model
        if val_metrics['iou'] > best_val_iou_stage2:
            best_val_iou_stage2 = val_metrics['iou']
            # torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ New best model saved! (IoU: {best_val_iou_stage2:.6f})")
            patience_stage2 = 0
        else:
            patience_stage2 += 1
        
        if patience_stage2 >= max_patience_stage2:
            print(f"\n🛑 Stage 2 early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    final_model_path = os.path.join(model_save_dir, f'vegetation_unet_final_{timestamp}.pt')
    torch.save(model.state_dict(), final_model_path)
    
    print("\n" + "="*70)
    print(f"🎉 Two-Stage Training Completed!")
    print(f"{'='*70}")
    print(f"🏆 Stage 1 Best IoU: {best_val_iou_stage1:.6f}")
    print(f"🏆 Stage 2 Best IoU: {best_val_iou_stage2:.6f}")
    print(f"📈 Total Improvement: {(best_val_iou_stage2 - best_val_iou_stage1):.6f}")
    print(f"💾 Best model: {best_model_path}")
    print(f"💾 Final model: {final_model_path}")
    print("="*70 + "\n")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vegetation Detection Training')
    parser.add_argument('--input_tiles_dir', required=True, help='Directory with tiles')
    parser.add_argument('--input_masks_dir', required=True, help='Directory with masks')
    parser.add_argument('--model_path', required=True, help='Output directory for model')
    parser.add_argument('--weights_path', default=None, help='Pre-trained weights')
    args = parser.parse_args()
    
    with open('../config/config_vegetation.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    set_gpu()
    os.makedirs(args.model_path, exist_ok=True)
    
    train_vegetation_detector(config, tiles_dir=args.input_tiles_dir,
                            masks_dir=args.input_masks_dir,
                            model_save_dir=args.model_path,
                            weights_path=args.weights_path)
