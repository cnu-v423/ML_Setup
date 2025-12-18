# ENHANCED TRAINING FUNCTIONS WITH LOGGING AND PROGRESS BARS

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from pytorch_backbone_model_v2 import (
    Channel3_DataGenerator_old,
    Channel4_DataGenerator,
    Channel4_MultiDataGenerator,
    get_callbacks,
    build_unet_resnet50,
    create_ensemble_model
)
#from my_new_model import build_model

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

def iou_metric(y_pred, y_true, smooth=1):
    """Intersection over Union (IoU) metric"""
    intersection = (y_true * y_pred).sum(dim=(2, 3))
    union = y_true.sum(dim=(2, 3)) + y_pred.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


def dice_coefficient(y_pred, y_true, smooth=1):
    """Dice coefficient metric"""
    intersection = (y_true * y_pred).sum(dim=(2, 3))
    union = y_true.sum(dim=(2, 3)) + y_pred.sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()


def dice_loss(y_pred, y_true):
    """Dice loss function"""
    return 1 - dice_coefficient(y_pred, y_true)


class BuildingRecall(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset state like TensorFlow metrics"""
        self.tp = 0.0
        self.fn = 0.0

    def update(self, y_pred, y_true):
        """Update state exactly like TensorFlow's update_state"""
        # Convert probabilities to binary predictions (matching TensorFlow)
        y_pred_binary = (y_pred > self.threshold).float()
        y_true = y_true.float()

        # Calculate true positives and false negatives (matching TensorFlow exactly)
        with torch.no_grad():
            tp = (y_true * y_pred_binary).sum().item()
            fn = (y_true * (1 - y_pred_binary)).sum().item()

            # Update state (matching TensorFlow's assign_add)
            self.tp += tp
            self.fn += fn

    def compute(self):
        """Calculate recall exactly like TensorFlow"""
        return self.tp / (self.tp + self.fn + 1e-7)


class BuildingPrecision(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset state like TensorFlow metrics"""
        self.tp = 0.0
        self.fp = 0.0

    def update(self, y_pred, y_true):
        """Update state exactly like TensorFlow's update_state"""
        # Convert probabilities to binary predictions (matching TensorFlow)
        y_pred_binary = (y_pred > self.threshold).float()
        y_true = y_true.float()

        # Calculate true positives and false positives (matching TensorFlow exactly)
        with torch.no_grad():
            tp = (y_true * y_pred_binary).sum().item()
            fp = ((1 - y_true) * y_pred_binary).sum().item()

            # Update state (matching TensorFlow's assign_add)
            self.tp += tp
            self.fp += fp

    def compute(self):
        """Calculate precision exactly like TensorFlow"""
        return self.tp / (self.tp + self.fp + 1e-7)


class BuildingIoU(nn.Module):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset state like TensorFlow metrics"""
        self.intersection = 0.0
        self.union = 0.0

    def update(self, y_pred, y_true):
        """Update state exactly like TensorFlow's update_state"""
        # Convert probabilities to binary predictions (matching TensorFlow)
        y_pred_binary = (y_pred > self.threshold).float()
        y_true = y_true.float()

        # Calculate intersection and union (matching TensorFlow exactly)
        with torch.no_grad():
            intersection = (y_true * y_pred_binary).sum().item()
            union = y_true.sum().item() + y_pred_binary.sum().item() - intersection

            # Update state (matching TensorFlow's assign_add)
            self.intersection += intersection
            self.union += union

    def compute(self):
        """Calculate IoU exactly like TensorFlow"""
        return self.intersection / (self.union + 1e-7)


class MultiClassIoU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.intersections = [0.0] * self.num_classes
        self.unions = [0.0] * self.num_classes

    def update(self, y_pred, y_true):
        # Assume y_pred is logits or probabilities with shape (N, C, H, W)
        # Convert to predicted class labels
        y_pred_labels = torch.argmax(y_pred, dim=1)
        y_true = y_true.long()

        with torch.no_grad():
            for c in range(self.num_classes):
                intersection = ((y_pred_labels == c) & (y_true == c)).sum().item()
                union = ((y_pred_labels == c) | (y_true == c)).sum().item()
                self.intersections[c] += intersection
                self.unions[c] += union

    def compute(self):
        ious = []
        for c in range(self.num_classes):
            iou = self.intersections[c] / (self.unions[c] + 1e-7)
            ious.append(iou)
        mean_iou = sum(ious) / self.num_classes
        return mean_iou, ious


class F1Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.precision = BuildingPrecision()
        self.recall = BuildingRecall()

    def reset(self):
        self.precision.reset()
        self.recall.reset()

    def update(self, y_pred, y_true):
        self.precision.update(y_pred, y_true)
        self.recall.update(y_pred, y_true)

    def compute(self):
        p = self.precision.compute()
        r = self.recall.compute()
        return 2 * (p * r) / (p + r + 1e-7)


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (probabilities)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, 1, H, W)
        """
        n, c, _, _ = pred.shape

        # pred is already probabilities from sigmoid activation
        # one-hot vector of ground truth
        one_hot_gt = F.one_hot(gt.squeeze(1).long(), c).permute(0, 3, 1, 2).float()

        # boundary map
        pad0 = (self.theta0 - 1) // 2
        pad = (self.theta - 1) // 2

        # boundary map calculations
        gt_b = F.max_pool2d(1 - one_hot_gt, self.theta0, stride=1, padding=pad0)
        gt_b = gt_b - (1 - one_hot_gt)

        pred_b = F.max_pool2d(1 - pred, self.theta0, stride=1, padding=pad0)
        pred_b = pred_b - (1 - pred)

        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, self.theta, stride=1, padding=pad)
        pred_b_ext = F.max_pool2d(pred_b, self.theta, stride=1, padding=pad)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = (pred_b * gt_b_ext).sum(dim=2) / (pred_b.sum(dim=2) + 1e-7)
        R = (pred_b_ext * gt_b).sum(dim=2) / (gt_b.sum(dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = (1 - BF1).mean()

        return loss


class AdaptiveLossLayer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # ✅ Learnable parameters (matching TensorFlow exactly)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.gamma = nn.Parameter(torch.ones(1))
        self.class_weights = nn.Parameter(torch.ones(num_classes))

    def dice_loss(self, y_pred, y_true):
        """Dice loss matching TensorFlow implementation exactly"""
        smooth = 1e-7
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum()
        dice_coef = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice_coef

    def boundary_loss(self, y_pred, y_true):
        """Boundary loss matching TensorFlow's sobel_edges implementation"""
        # Sobel edge detection (matching TensorFlow's sobel_edges)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # Move to same device as input
        sobel_x = sobel_x.to(y_pred.device)
        sobel_y = sobel_y.to(y_pred.device)

        # Apply Sobel filters
        edges_true_x = F.conv2d(y_true, sobel_x, padding=1)
        edges_true_y = F.conv2d(y_true, sobel_y, padding=1)
        edges_pred_x = F.conv2d(y_pred, sobel_x, padding=1)
        edges_pred_y = F.conv2d(y_pred, sobel_y, padding=1)

        # Compute edge magnitude with numerical stability
        eps = 1e-8
        edges_true_mag_sq = edges_true_x**2 + edges_true_y**2
        edges_pred_mag_sq = edges_pred_x**2 + edges_pred_y**2

        # Clamp to ensure non-negative values before sqrt
        edges_true_mag_sq = torch.clamp(edges_true_mag_sq, min=0.0)
        edges_pred_mag_sq = torch.clamp(edges_pred_mag_sq, min=0.0)

        edges_true = torch.sqrt(edges_true_mag_sq + eps)
        edges_pred = torch.sqrt(edges_pred_mag_sq + eps)

        return torch.mean(torch.abs(edges_true - edges_pred))

    def forward(self, y_pred, y_true):
        """
        ✅ FIXED: Handle probabilities correctly (both models output probabilities via sigmoid)
        Args:
            y_pred: Model output with sigmoid already applied (probabilities [0,1])
            y_true: Ground truth masks (values [0,1])
        """
        # ✅ Apply constraints like TensorFlow's NonNeg constraint
        alpha = torch.clamp(self.alpha, min=0.0)
        beta = torch.clamp(self.beta, min=0.0)
        gamma = torch.clamp(self.gamma, min=0.0)
        class_weights = torch.clamp(self.class_weights, min=0.0)

        # Clamp inputs to ensure valid ranges
        y_pred = torch.clamp(y_pred, min=1e-7, max=1.0-1e-7)
        y_true = torch.clamp(y_true, min=0.0, max=1.0)

        # ✅ FIXED: Use binary_cross_entropy (not with_logits) since model outputs probabilities
        # This matches TensorFlow's binary_crossentropy exactly
        bce = F.binary_cross_entropy(y_pred, y_true, reduction='none')

        # ✅ Apply class weighting exactly like TensorFlow
        # TensorFlow: class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        # TensorFlow: bce_loss = K.sum(class_loglosses * K.constant(weights))
        class_loglosses = bce.mean(dim=[2, 3])  # Mean over spatial dimensions like TensorFlow
        if len(class_weights) > 1:
            bce_loss = (class_loglosses * class_weights[1]).mean()  # Apply class weight
        else:
            bce_loss = class_loglosses.mean()

        # Dice loss (works with probabilities)
        dice = self.dice_loss(y_pred, y_true)

        # Boundary loss (works with probabilities)
        try:
            edge_loss = self.boundary_loss(y_pred, y_true)
        except Exception as e:
            print(f"Warning: Boundary loss calculation failed: {e}")
            edge_loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)

        # ✅ Combine losses with constrained weights (matching TensorFlow exactly)
        total_loss = alpha * bce_loss + beta * dice + gamma * edge_loss

        return total_loss


def get_filename_without_extension(filepath):
    """Extract filename without extension from a path"""
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]


def match_tile_mask_pairs(tiles_dir, masks_dir):
    """Match tiles and masks based on filenames and return only valid pairs"""
    tile_files = glob.glob(os.path.join(tiles_dir, '*.tif'))
    mask_files = glob.glob(os.path.join(masks_dir, '*.tif'))

    tile_dict = {get_filename_without_extension(f): f for f in tile_files}
    mask_dict = {get_filename_without_extension(f): f for f in mask_files}

    common_files = set(tile_dict.keys()).intersection(set(mask_dict.keys()))

    missing_masks = set(tile_dict.keys()) - set(mask_dict.keys())
    missing_tiles = set(mask_dict.keys()) - set(tile_dict.keys())

    if missing_masks:
        print(f"Warning: {len(missing_masks)} tiles have no corresponding mask. Examples: {list(missing_masks)[:3]}")

    if missing_tiles:
        print(f"Warning: {len(missing_tiles)} masks have no corresponding tile. Examples: {list(missing_tiles)[:3]}")

    print(f"Found {len(common_files)} valid tile-mask pairs")

    matched_tiles = [tile_dict[name] for name in common_files]
    matched_masks = [mask_dict[name] for name in common_files]

    return matched_tiles, matched_masks


def check_single_tile(args):
    """Check a single tile-mask pair"""
    tile_path, mask_path, expected_size = args
    
    try:
        # File existence
        if not (os.path.exists(tile_path) and os.path.exists(mask_path)):
            return None
            
        # File size check
        if os.path.getsize(tile_path) < 1000 or os.path.getsize(mask_path) < 1000:
            return None
            
        # Quick metadata check
        with rasterio.open(tile_path) as src:
            if src.width != expected_size or src.height != expected_size:
                return None
                
        with rasterio.open(mask_path) as mask_src:
            if mask_src.width != expected_size or mask_src.height != expected_size:
                return None
                
        return (tile_path, mask_path)
        
    except Exception:
        return None

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
import random
def filter_bad_tiles_parallel(tiles, masks, config, n_workers=None):
    """Parallel version - much faster for large datasets"""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 4)  # Don't overwhelm system
    
    expected_size = config['data']['input_size']
    args_list = [(tile, mask, expected_size) for tile, mask in zip(tiles, masks)]
    
    print(f"Starting parallel filtering with {n_workers} workers...")
    
    good_pairs = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(check_single_tile, args_list))
    
    # Filter out None results
    good_pairs = [pair for pair in results if pair is not None]
    
    if good_pairs:
        good_tiles, good_masks = zip(*good_pairs)
        good_tiles, good_masks = list(good_tiles), list(good_masks)
    else:
        good_tiles, good_masks = [], []
        
    print(f"Parallel filtering complete: {len(good_tiles)}/{len(tiles)} tiles retained")
    return good_tiles, good_masks

import numpy as np
import os
import rasterio
from concurrent.futures import ThreadPoolExecutor

def process_tile(tile_path, mask_path, config):
    try:
        if not os.path.exists(tile_path):
            print(f"Tile file does not exist: {tile_path}")
            return None, None

        if not os.path.exists(mask_path):
            print(f"Mask file does not exist: {mask_path}")
            return None, None

        with rasterio.open(tile_path) as src:
            image = src.read()

            if image.shape[1] != config['data']['input_size'] or image.shape[2] != config['data']['input_size']:
                print(f"Removing tile with incorrect shape: {tile_path}, shape: {image.shape}")
                return None, None

            # Check for normalization issues in all channels
            min_vals = np.percentile(image, 2, axis=(1, 2))
            max_vals = np.percentile(image, 98, axis=(1, 2))

            if np.any(min_vals >= max_vals) or np.any(np.isnan(min_vals)) or np.any(np.isnan(max_vals)):
                print(f"Removing problematic tile: {tile_path}")
                return None, None

            with rasterio.open(mask_path) as mask_src:
                mask = mask_src.read()
                if mask.shape[1] != config['data']['input_size'] or mask.shape[2] != config['data']['input_size']:
                    print(f"Removing tile due to incorrect mask shape: {mask_path}, shape: {mask.shape}")
                    return None, None

            return tile_path, mask_path

    except Exception as e:
        print(f"Error processing tile-mask pair ({tile_path}, {mask_path}): {str(e)}")
        return None, None


def filter_bad_tiles(tiles, masks, config):
    good_tiles = []
    good_masks = []

    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda p: process_tile(*p, config), zip(tiles, masks))
        
        for tile_path, mask_path in results:
            if tile_path and mask_path:
                good_tiles.append(tile_path)
                good_masks.append(mask_path)

    print(f"Removed {len(tiles) - len(good_tiles)} problematic tiles")
    print(f"Remaining tiles: {len(good_tiles)}")

    return good_tiles, good_masks


def filter_bad_tiles_old(tiles, masks, config):
    """Remove tiles that cause normalization errors"""
    good_tiles = []
    good_masks = []

    for tile_path, mask_path in zip(tiles, masks):
        try:
            if not os.path.exists(tile_path):
                print(f"Tile file does not exist: {tile_path}")
                continue

            if not os.path.exists(mask_path):
                print(f"Mask file does not exist: {mask_path}")
                continue

            with rasterio.open(tile_path) as src:
                image = src.read()

                if image.shape[1] != config['data']['input_size'] or image.shape[2] != config['data']['input_size']:
                    print(f"Removing tile with incorrect shape: {tile_path}, shape: {image.shape}")
                    continue

                has_valid_data = True
                for channel in image:
                    min_val = np.percentile(channel, 2)
                    max_val = np.percentile(channel, 98)

                    if min_val >= max_val or np.isnan(min_val) or np.isnan(max_val):
                        has_valid_data = False
                        print(f"Removing problematic tile: {tile_path}")
                        break

                if has_valid_data:
                    with rasterio.open(mask_path) as mask_src:
                        mask = mask_src.read()
                        if mask.shape[1] != config['data']['input_size'] or mask.shape[2] != config['data']['input_size']:
                            print(f"Removing tile due to incorrect mask shape: {mask_path}, shape: {mask.shape}")
                            continue
                    good_tiles.append(tile_path)
                    good_masks.append(mask_path)

        except Exception as e:
            print(f"Error processing tile-mask pair ({tile_path}, {mask_path}): {str(e)}")
            continue

    print(f"Removed {len(tiles) - len(good_tiles)} problematic tiles")
    print(f"Remaining tiles: {len(good_tiles)}")

    return good_tiles, good_masks


def set_gpu():
    """Set GPU configuration"""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("Number of devices : ", device_count)
        if device_count >= 1:
            torch.cuda.set_device(0)  # Use first GPU
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")


def warmup_cosine_decay(epoch, initial_lr, total_epochs, warmup_epochs, min_lr=1e-7):
    """Warmup cosine decay learning rate schedule - matches TensorFlow exactly"""
    # Warmup phase
    if epoch < warmup_epochs:
        return initial_lr * ((epoch + 1) / warmup_epochs)

    # Cosine decay phase
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, progress))  # Clip to [0, 1]
    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    return min_lr + (initial_lr - min_lr) * cosine_decay


class WarmupCosineScheduler:
    def __init__(self, optimizer, initial_lr, total_epochs, warmup_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr

    def step(self, epoch):
        """Apply learning rate schedule - matches TensorFlow's LearningRateScheduler exactly"""
        lr = warmup_cosine_decay(epoch, self.initial_lr, self.total_epochs, self.warmup_epochs, self.min_lr)

        # Apply to optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

class AdaptiveLossLogger:
    """Logger for tracking adaptive loss parameters over training"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.fieldnames = ['epoch', 'stage', 'alpha', 'beta', 'gamma', 'class_weight_0', 'class_weight_1', 
                          'train_loss', 'val_loss', 'val_iou', 'learning_rate']
        
        # Create CSV file with headers
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log_epoch(self, epoch, stage, adaptive_loss, train_loss, val_loss, val_iou, learning_rate):
        """Log adaptive loss parameters and metrics for current epoch"""
        
        # Extract parameters (detach from computation graph)
        alpha = adaptive_loss.alpha.detach().cpu().item()
        beta = adaptive_loss.beta.detach().cpu().item() 
        gamma = adaptive_loss.gamma.detach().cpu().item()
        
        # Handle class weights
        class_weights = adaptive_loss.class_weights.detach().cpu().numpy()
        class_weight_0 = class_weights[0] if len(class_weights) > 0 else 0.0
        class_weight_1 = class_weights[1] if len(class_weights) > 1 else 0.0
        
        # Create row data
        row_data = {
            'epoch': epoch + 1,
            'stage': stage,
            'alpha': f"{alpha:.6f}",
            'beta': f"{beta:.6f}", 
            'gamma': f"{gamma:.6f}",
            'class_weight_0': f"{class_weight_0:.6f}",
            'class_weight_1': f"{class_weight_1:.6f}",
            'train_loss': f"{train_loss:.6f}",
            'val_loss': f"{val_loss:.6f}",
            'val_iou': f"{val_iou:.6f}",
            'learning_rate': f"{learning_rate:.8f}"
        }
        
        # Write to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row_data)
        
        # Also print to console with nice formatting
        self._print_parameters(epoch, stage, alpha, beta, gamma, class_weight_0, class_weight_1)
    
    def _print_parameters(self, epoch, stage, alpha, beta, gamma, class_weight_0, class_weight_1):
        """Print parameters to console with nice formatting"""
        print(f"\n{'='*60}")
        print(f"📊 ADAPTIVE LOSS PARAMETERS - Epoch {epoch + 1} ({stage})")
        print(f"{'='*60}")
        print(f"🔵 Alpha (BCE weight):     {alpha:.6f}")
        print(f"🟢 Beta (Dice weight):     {beta:.6f}")
        print(f"🟡 Gamma (Boundary weight): {gamma:.6f}")
        print(f"⚪ Class Weight 0:         {class_weight_0:.6f}")
        print(f"🔴 Class Weight 1:         {class_weight_1:.6f}")
        print(f"{'='*60}")


def train_epoch_with_progress(model, train_loader, optimizer, adaptive_loss, device, metrics, epoch, stage):
    """Enhanced training function with progress bars and detailed logging"""
    model.train()
    train_loss = 0.0
    # Reset metrics at start of epoch
    for metric in metrics.values():
        metric.reset()
    
    # 🚀 CREATE PROGRESS BAR FOR TRAINING
    train_pbar = tqdm(
        train_loader, 
        desc=f"🚂 {stage} - Epoch {epoch+1} [TRAIN]",
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    batch_losses = []
    
    for batch_idx, (images, masks) in enumerate(train_pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = adaptive_loss(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        train_loss += batch_loss
        batch_losses.append(batch_loss)
        
        # Update metrics
        for metric in metrics.values():
            metric.update(outputs.detach(), masks)
        
        # 📊 UPDATE PROGRESS BAR WITH CURRENT STATS
        if batch_idx % 10 == 0:  # Update every 10 batches to avoid slowdown
            current_avg_loss = np.mean(batch_losses[-50:])  # Rolling average of last 50 batches
            
            # Get current metric values (partial epoch)
            current_iou = metrics['building_iou'].compute() if metrics['building_iou'].union > 0 else 0.0
            
            # Update progress bar postfix
            train_pbar.set_postfix({
                'Loss': f'{current_avg_loss:.4f}',
                'IoU': f'{current_iou:.3f}',
                'GPU': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'CPU'
            })
    
    train_pbar.close()
    
    # Compute final epoch metrics
    epoch_metrics = {name: metric.compute() for name, metric in metrics.items()}
    epoch_metrics['loss'] = train_loss / len(train_loader)
    
    return epoch_metrics


def validate_epoch_with_progress(model, val_loader, adaptive_loss, device, metrics, epoch, stage):
    """Enhanced validation function with progress bars"""
    model.eval()
    val_loss = 0.0
    
    # Reset metrics
    for metric in metrics.values():
        metric.reset()
    
    # 🚀 CREATE PROGRESS BAR FOR VALIDATION
    val_pbar = tqdm(
        val_loader,
        desc=f"🔍 {stage} - Epoch {epoch+1} [VAL]  ",
        leave=False,
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
    )
    
    batch_losses = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_pbar):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = adaptive_loss(outputs, masks)
            
            batch_loss = loss.item()
            val_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Update metrics
            for metric in metrics.values():
                metric.update(outputs, masks)
            
            # 📊 UPDATE PROGRESS BAR WITH CURRENT STATS
            if batch_idx % 5 == 0:  # Update every 5 batches for validation
                current_avg_loss = np.mean(batch_losses[-20:])  # Rolling average
                current_iou = metrics['building_iou'].compute() if metrics['building_iou'].union > 0 else 0.0
                
                val_pbar.set_postfix({
                    'Loss': f'{current_avg_loss:.4f}',
                    'IoU': f'{current_iou:.3f}',
                    'GPU': f'{torch.cuda.memory_allocated()/1e9:.1f}GB' if torch.cuda.is_available() else 'CPU'
                })
    
    val_pbar.close()
    
    # Compute final epoch metrics
    epoch_metrics = {name: metric.compute() for name, metric in metrics.items()}
    epoch_metrics['loss'] = val_loss / len(val_loader)
    
    return epoch_metrics


class EnhancedMetricsLogger:
    """Enhanced metrics logger with parameter tracking"""
    def __init__(self):
        pass
    
    def log_epoch_metrics(self, epoch, stage, train_metrics, val_metrics, adaptive_loss_params=None):
        """Enhanced logging with adaptive loss parameters"""
        print(f"\n{'🎯 EPOCH SUMMARY':<60}")
        print(f"{'─'*60}")
        print(f"📈 Training Metrics:")
        print(f"   • Loss:      {train_metrics['loss']:.6f}")
        print(f"   • Precision: {train_metrics.get('building_precision', 0):.4f}")
        print(f"   • Recall:    {train_metrics.get('building_recall', 0):.4f}")
        print(f"   • IoU:       {train_metrics.get('building_iou', 0):.4f}")
        
        print(f"\n📉 Validation Metrics:")
        print(f"   • Loss:      {val_metrics['loss']:.6f}")
        print(f"   • Precision: {val_metrics.get('building_precision', 0):.4f}")
        print(f"   • Recall:    {val_metrics.get('building_recall', 0):.4f}")
        print(f"   • IoU:       {val_metrics.get('building_iou', 0):.4f}")
        
        # Calculate and display F1 Score
        precision = val_metrics.get('building_precision', 0)
        recall = val_metrics.get('building_recall', 0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        print(f"   • F1 Score:  {f1:.4f}")
        
        # 🎨 Add visual indicators for performance
        if val_metrics.get('building_iou', 0) > 0.8:
            print(f"   🌟 Excellent IoU!")
        elif val_metrics.get('building_iou', 0) > 0.7:
            print(f"   ⭐ Good IoU!")
        elif val_metrics.get('building_iou', 0) > 0.5:
            print(f"   🔸 Moderate IoU")
        else:
            print(f"   🔻 Low IoU - needs improvement")



def train_enhanced(config, tiles_dir, masks_dir, model_path, run_name, weights_path=None):
    """Enhanced training function with comprehensive logging and progress tracking"""

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    
    # Create logging directory
    log_dir = os.path.join(model_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize adaptive loss parameter logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    adaptive_loss_logger = AdaptiveLossLogger(
        os.path.join(log_dir, f'adaptive_loss_params_{timestamp}.csv')
    )
    
    # Enhanced metrics logger
    metrics_logger = EnhancedMetricsLogger()
    
    # Data preparation (same as before)
    tiles, masks = match_tile_mask_pairs(tiles_dir, masks_dir)
    #tiles, masks = filter_bad_tiles(tiles, masks, config)
    train_tiles, val_tiles, train_masks, val_masks = train_test_split(
        tiles, masks,
        test_size=config['data']['validation_split'],
        random_state=config['data']['random_state']
    )

    # Create datasets and loaders (same as before)
    if config['data']['channels'] == 3:
        train_dataset = Channel3_DataGenerator_old(train_tiles, train_masks, config, is_training=True)
        val_dataset = Channel3_DataGenerator_old(val_tiles, val_masks, config, is_training=False)
    else:
        train_dataset = Channel4_DataGenerator(train_tiles, train_masks, config, is_training=True)
        val_dataset = Channel4_DataGenerator(val_tiles, val_masks, config, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)

    # Build model and loss
    model = build_unet_resnet50(
        num_classes=1, input_size=config['data']['input_size'], freeze_backbone=True
    )

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Detected {torch.cuda.device_count()} GPUs. Wrapping model in DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(device)
    
    #model = build_model(1, 3 ,encoder_name="senet154").to(device)

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))

    adaptive_loss = AdaptiveLossLayer().to(device)

    # Create metrics
    train_metrics = {
        'building_iou': BuildingIoU(),
        'building_recall': BuildingRecall(),
        'building_precision': BuildingPrecision()
    }
    val_metrics = {
        'building_iou': BuildingIoU(),
        'building_recall': BuildingRecall(),
        'building_precision': BuildingPrecision()
    }

    # Training configuration
    initial_lr_stage1 = config['model']['learning_rate']
    total_epochs_stage1 = 10
    warmup_epochs_stage1 = 2
    initial_lr_stage2 = initial_lr_stage1 * 0.1
    total_epochs_stage2 = 100
    warmup_epochs_stage2 = 3

    print(f"\n🎯 TRAINING CONFIGURATION")
    print(f"{'='*50}")
    print(f"📊 Dataset: {len(train_tiles)} train, {len(val_tiles)} val")
    print(f"⚙️  Batch size: {config['data']['batch_size']}")
    print(f"🧠 Model: U-Net with {model.__class__.__name__}")
    print(f"📈 Stage 1: {total_epochs_stage1} epochs (frozen backbone)")
    print(f"📈 Stage 2: {total_epochs_stage2} epochs (fine-tuning)")
    print(f"💾 Logs saved to: {log_dir}")
    print(f"{'='*50}")

    # ─── STAGE 1: FROZEN BACKBONE ───
    print(f"\n🚂 Starting Stage 1: Frozen Backbone Training")
    print(f"{'='*60}")
    
    optimizer_stage1 = torch.optim.Adam(model.parameters(), lr=initial_lr_stage1)
    scheduler_stage1 = WarmupCosineScheduler(optimizer_stage1, initial_lr_stage1, 
                                           total_epochs_stage1, warmup_epochs_stage1)
    
    best_val_loss_stage1 = float('inf')
    
    # 🎯 STAGE 1 TRAINING LOOP WITH PROGRESS TRACKING
    for epoch in range(total_epochs_stage1):
        current_lr = scheduler_stage1.step(epoch)
        
        # Train with progress bars
        train_results = train_epoch_with_progress(
            model, train_loader, optimizer_stage1, adaptive_loss, device, 
            train_metrics, epoch, "Stage 1"
        )
        
        # Validate with progress bars  
        val_results = validate_epoch_with_progress(
            model, val_loader, adaptive_loss, device, val_metrics, epoch, "Stage 1"
        )
        
        # 📊 LOG ADAPTIVE LOSS PARAMETERS
        adaptive_loss_logger.log_epoch(
            epoch=epoch,
            stage="Stage_1", 
            adaptive_loss=adaptive_loss,
            train_loss=train_results['loss'],
            val_loss=val_results['loss'],
            val_iou=val_results['building_iou'],
            learning_rate=current_lr
        )
        
        # Log metrics
        print(f"🔧 Learning Rate: {current_lr:.8f}")

        
        # Save best model
        if val_results['loss'] < best_val_loss_stage1:
            best_val_loss_stage1 = val_results['loss']
            #torch.save(model.state_dict(), f"{model_path}/best_model_stage1_{timestamp}.pt")
            print(f"💾 New best model saved! (Val Loss: {best_val_loss_stage1:.6f})")

    # ─── STAGE 2: FINE-TUNING ───
    print(f"\n🔧 Starting Stage 2: Fine-Tuning")
    print(f"{'='*60}")
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_stage2 = torch.optim.Adam(model.parameters(), lr=initial_lr_stage2)
    scheduler_stage2 = WarmupCosineScheduler(optimizer_stage2, initial_lr_stage2, 
                                           total_epochs_stage2, warmup_epochs_stage2, min_lr=1e-8)
    
    best_val_iou = 0.0
    patience_counter = 0
    
    # 🎯 STAGE 2 TRAINING LOOP WITH PROGRESS TRACKING
    for epoch in range(total_epochs_stage2):
        current_lr = scheduler_stage2.step(epoch)
        
        # Train with progress bars
        train_results = train_epoch_with_progress(
            model, train_loader, optimizer_stage2, adaptive_loss, device,
            train_metrics, epoch, "Stage 2"
        )
        
        # Validate with progress bars
        val_results = validate_epoch_with_progress(
            model, val_loader, adaptive_loss, device, val_metrics, epoch, "Stage 2"
        )
        
        # 📊 LOG ADAPTIVE LOSS PARAMETERS  
        adaptive_loss_logger.log_epoch(
            epoch=epoch,
            stage="Stage_2",
            adaptive_loss=adaptive_loss,
            train_loss=train_results['loss'],
            val_loss=val_results['loss'], 
            val_iou=val_results['building_iou'],
            learning_rate=current_lr
        )
        

        print(f"🔧 Learning Rate: {current_lr:.8f}")
        

        val_iou = val_results['building_iou']
        
        # Save best model and handle early stopping
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            #torch.save(model.state_dict(), f"{model_path}/best_model_{timestamp}.pt")
            print(f"💾 New best model saved! (Val IoU: {best_val_iou:.6f})")
            #mlflow.log_artifact(model_file, artifact_path="stage2_checkpoints")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
            break


    # Save final model
    final_model_file = os.path.join(model_path, 'unet_resnet50_final.pt')
    torch.save(model.state_dict(), final_model_file)


    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"🏆 Best Validation IoU: {best_val_iou:.6f}")
    print(f"📊 Parameter logs saved to: {adaptive_loss_logger.log_file}")
    print(f"💾 Final model: {os.path.join(model_path, 'unet_resnet50_final.pt')}")
    print(f"{'='*60}")


    # Infer and log a model signature (input→output schema).
    model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(val_loader))
        sample_images = sample_images.to(device)
        # Run a forward pass to get a dummy output:
        sample_outputs = model(sample_images)

    
    return model


# 🚀 USAGE EXAMPLE
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced U-Net training with parameter logging')
    parser.add_argument('--input_tiles_dir', required=True, help='Directory containing input tiles')
    parser.add_argument('--input_masks_dir', required=True, help='Directory containing input masks')
    parser.add_argument('--model_path', required=True, help='Directory for saving model')
    parser.add_argument('--weights_path', help='Path to pre-trained weights', default=None)
    parser.add_argument('--run_name', required=False, help='Name of the MLFlow Run')
    args = parser.parse_args()

    with open('../config/config_v1.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_gpu()
    os.makedirs(args.model_path, exist_ok=True)
    
    # 🚀 RUN ENHANCED TRAINING
    train_enhanced(config, tiles_dir=args.input_tiles_dir, masks_dir=args.input_masks_dir, 
                  model_path=args.model_path, run_name=args.run_name, weights_path=args.weights_path)