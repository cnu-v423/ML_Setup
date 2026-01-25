# ADVANCED MULTI-CLASS SEGMENTATION TRAINING
# State-of-the-art techniques for 95%+ accuracy
# Road Type Classification: Background, Thar, CC, Mud/Gravel

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from pytorch_backbone_model_v2 import (
    build_unet_resnet50,
)

import time
import glob
import math
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import argparse
import rasterio
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from contextlib import nullcontext
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from datetime import datetime

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️ albumentations not installed. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'albumentations'])
    import albumentations as A
    from albumentations.pytorch import ToTensorV2



# ==================== ADVANCED LOSS FUNCTIONS ====================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance - critical for road types"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (N, C, H, W)
            targets: class indices (N, H, W)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Get probabilities
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LovaszSoftmaxLoss(nn.Module):
    """Lovász-Softmax Loss - directly optimizes IoU"""
    def __init__(self, classes=4):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (N, C, H, W)
            targets: class indices (N, H, W)
        """
        probas = F.softmax(inputs, dim=1)
        loss = 0
        for c in range(self.classes):
            target_c = (targets == c).float()
            proba_c = probas[:, c]
            loss += self._lovasz_softmax_flat(proba_c, target_c)
        return loss / self.classes

    @staticmethod
    def _lovasz_softmax_flat(probas, labels):
        """Lovász loss for binary case"""
        if probas.numel() == 0:
            return probas.sum()
        
        iou_gt = labels.mean()
        if iou_gt == 0:
            return probas[probas != probas].sum()
        
        sorted_probas, sorted_idx = torch.sort(probas.view(-1), descending=True)
        sorted_labels = labels.view(-1)[sorted_idx]
        
        intersection = sorted_labels * sorted_probas
        union = sorted_labels + sorted_probas - intersection
        
        jaccard = 1. - intersection.cumsum(0) / union.cumsum(0).clamp(min=1e-5)
        jaccard = torch.cat((jaccard[0:1], jaccard[1:] - jaccard[:-1]))
        
        return jaccard.dot(sorted_probas)


class DiceLoss(nn.Module):
    """Dice Loss - excellent for multi-class segmentation"""
    def __init__(self, num_classes, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits (N, C, H, W)
            targets: class indices (N, H, W)
        """
        probas = F.softmax(inputs, dim=1)
        
        dice_scores = []
        for c in range(self.num_classes):
            target_c = (targets == c).float()
            proba_c = probas[:, c]
            
            intersection = (proba_c * target_c).sum()
            union = proba_c.sum() + target_c.sum()
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(1.0 - dice)
        
        return torch.mean(torch.stack(dice_scores))


class ComboLoss(nn.Module):
    """Combination of multiple losses for optimal training"""
    def __init__(self, num_classes, class_weights=None, alpha=0.5, beta=0.3, gamma=0.2):
        super(ComboLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # Weight for CE + Focal
        self.beta = beta    # Weight for Dice
        self.gamma = gamma  # Weight for Lovász
        
        # Create weighted cross entropy
        if class_weights is None:
            class_weights = torch.ones(num_classes)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
        self.dice_loss = DiceLoss(num_classes=num_classes)
        self.lovasz_loss = LovaszSoftmaxLoss(classes=num_classes)

    def forward(self, inputs, targets):
        """Combined loss function"""
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        lovasz = self.lovasz_loss(inputs, targets)
        
        # Weighted combination (normalized)
        total_weight = self.alpha + self.beta + self.gamma
        loss = (self.alpha * (ce + focal) / 2.0 + self.beta * dice + self.gamma * lovasz) / total_weight
        
        return loss


# ==================== ADVANCED METRICS ====================

class MultiClassMetrics(nn.Module):
    """Comprehensive multi-class segmentation metrics"""
    def __init__(self, num_classes=4, class_names=None):
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.num_classes)
        self.fp = np.zeros(self.num_classes)
        self.fn = np.zeros(self.num_classes)
        self.tn = np.zeros(self.num_classes)

    def update(self, y_pred, y_true):
        """Update metrics with batch predictions"""
        # Convert logits to class indices
        y_pred_labels = torch.argmax(y_pred, dim=1)  # (N, H, W)
        y_true = y_true.long()

        with torch.no_grad():
            for c in range(self.num_classes):
                pred_c = (y_pred_labels == c).long()
                true_c = (y_true == c).long()
                
                self.tp[c] += (pred_c & true_c).sum().item()
                self.fp[c] += (pred_c & ~true_c).sum().item()
                self.fn[c] += (~pred_c & true_c).sum().item()
                self.tn[c] += (~pred_c & ~true_c).sum().item()

    def compute(self):
        """Compute all metrics"""
        epsilon = 1e-7
        
        # Per-class metrics
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        
        # Accuracy variants
        accuracy = self.tp / (self.tp + self.fn + epsilon)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy
        }

    def mean_metrics(self):
        """Get mean metrics across classes"""
        metrics = self.compute()
        return {k: v.mean() for k, v in metrics.items()}


class SegmentationMetrics:
    """Wrapper for comprehensive metric tracking"""
    def __init__(self, num_classes=4):
        self.num_classes = num_classes
        self.multiclass_metrics = MultiClassMetrics(num_classes)

    def reset(self):
        self.multiclass_metrics.reset()

    def update(self, y_pred, y_true):
        self.multiclass_metrics.update(y_pred, y_true)

    def compute_metrics(self):
        """Get detailed metrics"""
        mean_metrics = self.multiclass_metrics.mean_metrics()
        per_class = self.multiclass_metrics.compute()
        
        return {
            'mean_iou': mean_metrics['iou'],
            'mean_f1': mean_metrics['f1'],
            'mean_precision': mean_metrics['precision'],
            'mean_recall': mean_metrics['recall'],
            'mean_accuracy': mean_metrics['accuracy'],
            'per_class_iou': per_class['iou'],
            'per_class_f1': per_class['f1'],
        }


# ==================== ADVANCED DATA AUGMENTATION ====================

def get_advanced_augmentation(train=True, input_size=256):
    """Advanced data augmentation for improved generalization"""
    if train:
        return A.Compose([
            # Geometric augmentations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.7, border_mode=0),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ElasticTransform(p=0.3),
            A.GridDistortion(p=0.3),
            
            # Intensity augmentations
            A.GaussNoise(p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.3),
            A.ChannelShuffle(p=0.2),
            A.CLAHE(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.2),
            
            # Normalize
            A.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2, 0.2]),
            ToTensorV2()
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2, 0.2]),
            ToTensorV2()
        ], is_check_shapes=False)


# ==================== ADVANCED DATA GENERATOR ====================

class AdvancedChannel4_MultiDataGenerator:
    """Enhanced data generator with augmentation for multi-class"""
    def __init__(self, image_paths, mask_paths, config, is_training=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.input_size = config['data']['input_size']
        self.channels = config['data']['channels']
        self.num_classes = config['data']['num_classes']
        self.is_training = is_training
        self.transform = get_advanced_augmentation(train=is_training, input_size=self.input_size)
        self.indexes = np.arange(len(image_paths))
        if self.is_training:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        idx = self.indexes[index]
        
        try:
            # Load image
            with rasterio.open(self.image_paths[idx]) as src:
                image = src.read()  # (C, H, W)
                
                # Normalize each channel independently
                for j in range(src.count):
                    channel = image[j]
                    min_val = np.percentile(channel, 2)
                    max_val = np.percentile(channel, 98)
                    
                    # Avoid division by zero
                    if max_val > min_val:
                        image[j] = np.clip((channel - min_val) / (max_val - min_val + 1e-7), 0, 1)
                    else:
                        image[j] = np.zeros_like(channel)
                
                image = image.astype(np.float32)

            # Load mask
            with rasterio.open(self.mask_paths[idx]) as src:
                mask = src.read(1).astype(np.uint8)  # (H, W)

            # Convert image to (H, W, C) for augmentation
            image = np.transpose(image, (1, 2, 0))
            
            # Apply augmentation
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = torch.from_numpy(augmented['mask']).long()
            else:
                image = torch.from_numpy(image).float()
                mask = torch.from_numpy(mask).long()

            return image, mask

        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # Return dummy tensors to avoid breaking training
            dummy_image = torch.zeros((4, self.input_size, self.input_size), dtype=torch.float32)
            dummy_mask = torch.zeros((self.input_size, self.input_size), dtype=torch.long)
            return dummy_image, dummy_mask

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indexes)


# ==================== HELPER FUNCTIONS ====================

def get_filename_without_extension(filepath):
    """Extract filename without extension"""
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]


def match_tile_mask_pairs(tiles_dir, masks_dir):
    """Match tiles and masks"""
    tile_files = glob.glob(os.path.join(tiles_dir, '*.tif'))
    mask_files = glob.glob(os.path.join(masks_dir, '*.tif'))

    tile_dict = {get_filename_without_extension(f): f for f in tile_files}
    mask_dict = {get_filename_without_extension(f): f for f in mask_files}

    common_files = set(tile_dict.keys()).intersection(set(mask_dict.keys()))
    
    print(f"✓ Found {len(common_files)} valid tile-mask pairs")
    
    matched_tiles = [tile_dict[name] for name in common_files]
    matched_masks = [mask_dict[name] for name in common_files]

    return matched_tiles, matched_masks


def compute_class_weights_from_masks(mask_paths, num_classes=4):
    """Compute class weights for handling imbalance"""
    print("🔎 Computing class weights from masks...")
    
    class_counts = np.zeros(num_classes)
    total_pixels = 0
    
    sample_size = min(100, len(mask_paths))
    for mask_path in tqdm(mask_paths[:sample_size], desc="Analyzing masks"):
        try:
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                for c in range(num_classes):
                    class_counts[c] += (mask == c).sum()
                total_pixels += mask.size
        except Exception as e:
            continue
    
    # Compute weights inversely proportional to class frequency
    class_weights = total_pixels / (num_classes * (class_counts + 1e-7))
    class_weights = class_weights / class_weights.sum() * num_classes
    
    print("📊 Class Weights:")
    class_names = ['Background', 'Thar Road', 'CC Road', 'Mud/Gravel Road']
    for i, weight in enumerate(class_weights):
        print(f"  {class_names[i]}: {weight:.4f}")
    
    return torch.from_numpy(class_weights).float()


def set_gpu():
    """Set GPU configuration"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ Found {device_count} GPU(s)")
        if device_count >= 1:
            torch.cuda.set_device(0)
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU available, using CPU")


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, initial_lr, total_epochs, warmup_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def step(self, epoch):
        """Compute and apply learning rate"""
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ==================== TRAINING FUNCTIONS ====================

def train_epoch_with_progress(model, train_loader, optimizer, criterion, device, 
                             metrics, epoch, scaler=None):
    """Enhanced training loop with mixed precision"""
    model.train()
    train_loss = 0.0
    metrics.reset()
    
    train_pbar = tqdm(
        train_loader,
        desc=f"🚂 Epoch {epoch+1} [TRAIN]",
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}'
    )
    
    batch_losses = []
    
    for batch_idx, (images, masks) in enumerate(train_pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision
        ctx = autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
        with ctx:
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        batch_loss = loss.item()
        train_loss += batch_loss
        batch_losses.append(batch_loss)
        
        # Update metrics
        metrics.update(outputs.detach(), masks)
        
        if batch_idx % 10 == 0:
            avg_loss = np.mean(batch_losses[-50:])
            train_pbar.set_postfix_str(f"{avg_loss:.4f}")
    
    train_pbar.close()
    
    metrics_dict = metrics.compute_metrics()
    metrics_dict['loss'] = train_loss / len(train_loader)
    
    return metrics_dict


def validate_epoch_with_progress(model, val_loader, criterion, device, metrics, epoch):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    metrics.reset()
    
    val_pbar = tqdm(
        val_loader,
        desc=f"🔍 Epoch {epoch+1} [VAL]  ",
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] Loss: {postfix}'
    )
    
    batch_losses = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_pbar):
            images, masks = images.to(device), masks.to(device)
            
            ctx = autocast(device_type='cuda') if device.type == 'cuda' else nullcontext()
            with ctx:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            batch_loss = loss.item()
            val_loss += batch_loss
            batch_losses.append(batch_loss)
            
            metrics.update(outputs, masks)
            
            if batch_idx % 5 == 0:
                avg_loss = np.mean(batch_losses[-20:])
                val_pbar.set_postfix_str(f"{avg_loss:.4f}")
    
    val_pbar.close()
    
    metrics_dict = metrics.compute_metrics()
    metrics_dict['loss'] = val_loss / len(val_loader)
    
    return metrics_dict


def log_epoch_metrics(epoch, stage, train_metrics, val_metrics, learning_rate):
    """Log training metrics"""
    print(f"\n{'='*80}")
    print(f"📊 EPOCH {epoch+1} SUMMARY - {stage}")
    print(f"{'='*80}")
    
    print(f"📈 Training:")
    print(f"  • Loss: {train_metrics['loss']:.6f}")
    print(f"  • mIoU: {train_metrics['mean_iou']:.4f}")
    print(f"  • mF1:  {train_metrics['mean_f1']:.4f}")
    print(f"  • Acc:  {train_metrics['mean_accuracy']:.4f}")
    
    print(f"\n📉 Validation:")
    print(f"  • Loss: {val_metrics['loss']:.6f}")
    print(f"  • mIoU: {val_metrics['mean_iou']:.4f}")
    print(f"  • mF1:  {val_metrics['mean_f1']:.4f}")
    print(f"  • Acc:  {val_metrics['mean_accuracy']:.4f}")
    
    print(f"\n🔧 Learning Rate: {learning_rate:.8f}")
    
    # Per-class metrics
    class_names = ['Background', 'Thar Road', 'CC Road', 'Mud/Gravel']
    print(f"\n📋 Per-Class Validation IoU:")
    for i, name in enumerate(class_names):
        print(f"  • {name}: {val_metrics['per_class_iou'][i]:.4f}")


# ==================== MAIN TRAINING FUNCTION ====================

def train_multiclass_enhanced(config, tiles_dir, masks_dir, model_path, weights_path=None):
    """Advanced multi-class training"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Create output directory
    os.makedirs(model_path, exist_ok=True)
    log_dir = os.path.join(model_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Data preparation
    tiles, masks = match_tile_mask_pairs(tiles_dir, masks_dir)
    
    # Compute class weights
    num_classes = config['data']['num_classes']
    class_weights = compute_class_weights_from_masks(masks, num_classes)
    class_weights = class_weights.to(device)
    
    # Stratified split
    presence_flags = []
    for m in masks:
        try:
            with rasterio.open(m) as src:
                mask_arr = src.read(1)
                presence_flags.append(int(mask_arr.sum() > 0))
        except:
            presence_flags.append(0)
    
    try:
        train_tiles, val_tiles, train_masks, val_masks = train_test_split(
            tiles, masks,
            test_size=config['data']['validation_split'],
            random_state=42,
            stratify=presence_flags
        )
    except:
        train_tiles, val_tiles, train_masks, val_masks = train_test_split(
            tiles, masks,
            test_size=config['data']['validation_split'],
            random_state=42
        )
    
    print(f"\n📊 Dataset Split:")
    print(f"  • Training: {len(train_tiles)}")
    print(f"  • Validation: {len(val_tiles)}")
    
    # Create datasets with advanced augmentation
    train_dataset = AdvancedChannel4_MultiDataGenerator(
        train_tiles, train_masks, config, is_training=True
    )
    val_dataset = AdvancedChannel4_MultiDataGenerator(
        val_tiles, val_masks, config, is_training=False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    print(f"\n🧠 Building Model...")
    model = build_unet_resnet50(
        num_classes=num_classes,
        input_size=config['data']['input_size'],
        freeze_backbone=True
    )
    
    if torch.cuda.device_count() > 1:
        print(f"  • Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"  • Loaded weights from {weights_path}")
    
    # Loss function with class weights
    criterion = ComboLoss(
        num_classes=num_classes,
        class_weights=class_weights,
        alpha=0.4,   # CE + Focal
        beta=0.4,    # Dice
        gamma=0.2    # Lovász
    ).to(device)
    
    # Metrics
    train_metrics = SegmentationMetrics(num_classes=num_classes)
    val_metrics = SegmentationMetrics(num_classes=num_classes)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training configuration
    print(f"\n📋 Training Configuration:")
    print(f"  • Total Epochs: {config['training']['epochs']}")
    print(f"  • Batch Size: {config['data']['batch_size']}")
    print(f"  • Learning Rate: {config['model']['learning_rate']}")
    print(f"  • Early Stopping Patience: {config['training']['early_stopping_patience']}")
    
    # Stage 1: Frozen backbone
    print(f"\n{'='*80}")
    print(f"🔒 STAGE 1: Frozen Backbone - 10 Epochs")
    print(f"{'='*80}\n")
    
    initial_lr_stage1 = config['model']['learning_rate']
    optimizer_stage1 = torch.optim.AdamW(model.parameters(), lr=initial_lr_stage1, weight_decay=1e-4)
    scheduler_stage1 = WarmupCosineScheduler(optimizer_stage1, initial_lr_stage1, 10, 2)
    
    best_val_loss_stage1 = float('inf')
    
    for epoch in range(10):
        lr = scheduler_stage1.step(epoch)
        
        train_results = train_epoch_with_progress(
            model, train_loader, optimizer_stage1, criterion, device, train_metrics, epoch, scaler
        )
        
        val_results = validate_epoch_with_progress(
            model, val_loader, criterion, device, val_metrics, epoch
        )
        
        log_epoch_metrics(epoch, "Stage 1", train_results, val_results, lr)
        
        if val_results['loss'] < best_val_loss_stage1:
            best_val_loss_stage1 = val_results['loss']
            checkpoint_path = os.path.join(model_path, 'best_model_stage1.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model (Val Loss: {best_val_loss_stage1:.6f})")
    
    # Stage 2: Fine-tuning
    print(f"\n{'='*80}")
    print(f"🔓 STAGE 2: Fine-Tuning All Parameters - 50 Epochs")
    print(f"{'='*80}\n")
    
    for param in model.parameters():
        param.requires_grad = True
    
    initial_lr_stage2 = initial_lr_stage1 * 0.1
    optimizer_stage2 = torch.optim.AdamW(model.parameters(), lr=initial_lr_stage2, weight_decay=1e-4)
    scheduler_stage2 = WarmupCosineScheduler(optimizer_stage2, initial_lr_stage2, 50, 3, min_lr=1e-8)
    
    best_val_iou = 0.0
    patience_counter = 0
    
    for epoch in range(50):
        lr = scheduler_stage2.step(epoch)
        
        train_results = train_epoch_with_progress(
            model, train_loader, optimizer_stage2, criterion, device, train_metrics, epoch, scaler
        )
        
        val_results = validate_epoch_with_progress(
            model, val_loader, criterion, device, val_metrics, epoch
        )
        
        log_epoch_metrics(epoch + 10, "Stage 2", train_results, val_results, lr)
        
        val_iou = val_results['mean_iou']
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            checkpoint_path = os.path.join(model_path, 'best_model_stage2.pt')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model (Val mIoU: {best_val_iou:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"\n🛑 Early stopping triggered after {epoch + 10 + 1} epochs")
            break
    
    # Save final model
    final_model_path = os.path.join(model_path, 'multiclass_road_segmentation_final.pt')
    torch.save(model.state_dict(), final_model_path)
    
    print(f"\n{'='*80}")
    print(f"🎉 TRAINING COMPLETED!")
    print(f"{'='*80}")
    print(f"🏆 Best Validation mIoU: {best_val_iou:.6f}")
    print(f"💾 Final Model: {final_model_path}")
    print(f"💾 Best Stage 2: {os.path.join(model_path, 'best_model_stage2.pt')}")
    print(f"{'='*80}")
    
    return model


# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Multi-Class Road Segmentation')
    parser.add_argument('--input_tiles_dir', required=True, help='Directory with input tiles')
    parser.add_argument('--input_masks_dir', required=True, help='Directory with mask tiles')
    parser.add_argument('--model_path', required=True, help='Directory to save model')
    parser.add_argument('--weights_path', help='Path to pre-trained weights', default=None)
    args = parser.parse_args()

    # Load config
    with open('../config/config_v1.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure multi-class config
    config['data']['num_classes'] = 4
    
    set_gpu()
    os.makedirs(args.model_path, exist_ok=True)
    
    # Run training
    train_multiclass_enhanced(
        config,
        tiles_dir=args.input_tiles_dir,
        masks_dir=args.input_masks_dir,
        model_path=args.model_path,
        weights_path=args.weights_path
    )

