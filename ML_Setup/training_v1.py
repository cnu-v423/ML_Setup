
import os
import yaml
import tensorflow as tf
#from new_model import create_unet_model, Channel4_DataGenerator, Channel3_DataGenerator
from model import create_unet_model,create_old_unet_model, create_old_enhanced_unet, create_enhanced_unet, Channel4_DataGenerator,Channel4_DataGenerator_new, Channel3_DataGenerator, Channel3_DataGenerator_old, get_callbacks, Channel5_DataGenerator, Channel4_DataGenerator_latest
# from pytorch_backbone_model_v2 import create_unet_model, Channel3_DataGenerator_old
from backup_utils import backup_project
import glob
import math
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
#from optimizer import optimize_threshold
from sklearn.model_selection import train_test_split
import argparse
import rasterio
import numpy as np
import torch

def iou(y_true, y_pred, smooth=1):
    """
    Intersection over Union (IoU) metric
   
    Args:
        y_true (tensor): Ground truth tensor
        y_pred (tensor): Predicted tensor
        smooth (float): Smoothing factor to prevent division by zero
   
    Returns:
        tensor: IoU metric value
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    iou_score = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou_score

def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth))

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

class BuildingRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='building_recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Initialize state variables for true positives and false negatives
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probabilities to binary predictions
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate true positives and false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        # Update state variables
        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        # Calculate recall: TP / (TP + FN)
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        return recall

    def reset_state(self):
        # Reset state variables at the start of each epoch
        self.tp.assign(0.0)
        self.fn.assign(0.0)

class BuildingPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='building_precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Initialize state variables for true positives and false positives
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probabilities to binary predictions
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate true positives and false positives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        
        # Update state variables
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        # Calculate precision: TP / (TP + FP)
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        return precision

    def reset_state(self):
        # Reset state variables at the start of each epoch
        self.tp.assign(0.0)
        self.fp.assign(0.0)

class BuildingIoU(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='building_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        # Initialize state variables
        self.intersection = self.add_weight(name='intersection', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert probabilities to binary predictions
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate intersection and union
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        
        # Update state variables
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        # Calculate IoU: Intersection / Union
        iou = self.intersection / (self.union + K.epsilon())
        return iou

    def reset_state(self):
        # Reset state variables at the start of each epoch
        self.intersection.assign(0.0)
        self.union.assign(0.0)

class MetricsLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nDetailed Metrics for Epoch {epoch + 1}:")
        print(f"Training Metrics:")
        print(f"Building Precision: {logs.get('building_precision', 0):.4f}")
        print(f"Building Recall: {logs.get('building_recall', 0):.4f}")
        print(f"Building IoU: {logs.get('building_iou', 0):.4f}")
        print(f"\nValidation Metrics:")
        print(f"Val Building Precision: {logs.get('val_building_precision', 0):.4f}")
        print(f"Val Building Recall: {logs.get('val_building_recall', 0):.4f}")
        print(f"Val Building IoU: {logs.get('val_building_iou', 0):.4f}")
        
        # Calculate F1 Score
        precision = logs.get('val_building_precision', 0)
        recall = logs.get('val_building_recall', 0)
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        print(f"Val F1 Score: {f1:.4f}")

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()



class BoundaryLoss(tf.keras.layers.Layer):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """
    def __init__(self, theta0=3, theta=5, **kwargs):
        super().__init__(**kwargs)
        self.theta0 = theta0
        self.theta = theta
    
    def call(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, H, W, C)
            - gt: ground truth map
                    shape (N, H, W) or (N, H, W, 1)
        Return:
            - boundary loss, averaged over mini-batch
        """
        # Cast inputs to the same data type (float32)
        pred = tf.cast(pred, tf.float32)
        gt = tf.cast(gt, tf.float32)
        
        # Ensure gt is the right shape - remove last dimension if it exists
        if len(gt.shape) == 4 and gt.shape[-1] == 1:
            gt = tf.squeeze(gt, axis=-1)
        
        # TensorFlow uses channels-last format (N, H, W, C)
        n = tf.shape(pred)[0]
        c = tf.shape(pred)[-1]
        
        # softmax so that predicted map can be distributed in [0, 1]
        pred = tf.nn.softmax(pred, axis=-1)
        
        # one-hot vector of ground truth
        one_hot_gt = tf.one_hot(tf.cast(gt, tf.int32), c)
        
        # boundary map
        # Create the padding configuration
        pad0 = (self.theta0 - 1) // 2
        pad = (self.theta - 1) // 2
        
        # Define max pooling for boundary extraction
        def max_pool2d(x, kernel_size, padding):
            padded = tf.pad(x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            return tf.nn.max_pool2d(padded, ksize=[1, kernel_size, kernel_size, 1], 
                                   strides=[1, 1, 1, 1], padding='VALID')
        
        # boundary map calculations
        gt_b = max_pool2d(1 - one_hot_gt, self.theta0, pad0)
        gt_b = gt_b - (1 - one_hot_gt)
        
        pred_b = max_pool2d(1 - pred, self.theta0, pad0)
        pred_b = pred_b - (1 - pred)
        
        # extended boundary map
        gt_b_ext = max_pool2d(gt_b, self.theta, pad)
        pred_b_ext = max_pool2d(pred_b, self.theta, pad)
        
        # reshape
        gt_b = tf.reshape(gt_b, [n, -1, c])
        pred_b = tf.reshape(pred_b, [n, -1, c])
        gt_b_ext = tf.reshape(gt_b_ext, [n, -1, c])
        pred_b_ext = tf.reshape(pred_b_ext, [n, -1, c])
        
        # Transpose to match the original dimensions for calculations
        gt_b = tf.transpose(gt_b, [0, 2, 1])
        pred_b = tf.transpose(pred_b, [0, 2, 1])
        gt_b_ext = tf.transpose(gt_b_ext, [0, 2, 1])
        pred_b_ext = tf.transpose(pred_b_ext, [0, 2, 1])
        
        # Precision, Recall
        P = tf.reduce_sum(pred_b * gt_b_ext, axis=2) / (tf.reduce_sum(pred_b, axis=2) + 1e-7)
        R = tf.reduce_sum(pred_b_ext * gt_b, axis=2) / (tf.reduce_sum(gt_b, axis=2) + 1e-7)
        
        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)
        
        # summing BF1 Score for each class and average over mini-batch
        loss = tf.reduce_mean(1 - BF1)
        
        return loss



def get_filename_without_extension(filepath):
    """
    Extract filename without extension from a path
    Example: '/path/to/tile001.tif' -> 'tile001'
    """
    basename = os.path.basename(filepath)
    return os.path.splitext(basename)[0]

def match_tile_mask_pairs(tiles_dir, masks_dir):
    """
    Match tiles and masks based on filenames and return only valid pairs
    """
    # Get all tile and mask files only from one folder
    tile_files = glob.glob(os.path.join(tiles_dir, '*.tif'))
    mask_files = glob.glob(os.path.join(masks_dir, '*.tif'))
    

    # Get all the tile and mask files from different folders
    # tile_files = glob.glob(os.path.join(tiles_dir, "tiles*", "*.tif"))
    # mask_files = glob.glob(os.path.join(masks_dir, "masks*", "*.tif"))

    # Create dictionaries with filenames (without extension) as keys
    tile_dict = {get_filename_without_extension(f): f for f in tile_files}
    mask_dict = {get_filename_without_extension(f): f for f in mask_files}
    
    # Find common keys (filenames that exist in both directories)
    common_files = set(tile_dict.keys()).intersection(set(mask_dict.keys()))
    
    # Report statistics
    missing_masks = set(tile_dict.keys()) - set(mask_dict.keys())
    missing_tiles = set(mask_dict.keys()) - set(tile_dict.keys())
    
    if missing_masks:
        print(f"Warning: {len(missing_masks)} tiles have no corresponding mask. Examples: {list(missing_masks)[:3]}")
    
    if missing_tiles:
        print(f"Warning: {len(missing_tiles)} masks have no corresponding tile. Examples: {list(missing_tiles)[:3]}")
    
    print(f"Found {len(common_files)} valid tile-mask pairs")
    
    # Create matched pairs
    matched_tiles = [tile_dict[name] for name in common_files]
    matched_masks = [mask_dict[name] for name in common_files]
    
    return matched_tiles, matched_masks

def filter_bad_tiles(tiles, masks, config):
    """
    Remove tiles that cause normalization errors (where min==max in percentiles)
    Returns filtered lists of tiles and masks
    """
    good_tiles = []
    good_masks = []
    
    for tile_path, mask_path in zip(tiles, masks):
        try:
            # Check if both files exist
            if not os.path.exists(tile_path):
                print(f"Tile file does not exist: {tile_path}")
                continue
            
            if not os.path.exists(mask_path):
                print(f"Mask file does not exist: {mask_path}")
                continue
            
            with rasterio.open(tile_path) as src:
                image = src.read()

                # print("Image value :: ", image)
                
                if image.shape[1] != config['data']['input_size'] or image.shape[2] != config['data']['input_size']:
                    print(f"Removing tile with incorrect shape: {tile_path}, shape: {image.shape}")
                    continue

                # Check each channel for normalization issues
                has_valid_data = True
                for channel in image[:3]:
                    # print("Channel :: ", channel)
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
    # List available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    
    if len(physical_devices) > 1:
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[1], True)
        print("Using second GPU with memory growth")
    elif len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Only one GPU available, memory growth set")
    else:
        print("No GPU found")


class VisualizationCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_generator, save_dir='./visualizations', num_samples=2):
        super().__init__()
        self.val_generator = val_generator
        self.save_dir = save_dir
        self.num_samples = min(num_samples, val_generator.batch_size)  # Ensure we don't exceed batch size
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:  # Save every 5 epochs
            X, y_true = self.val_generator.__getitem__(0)
            y_pred = self.model.predict(X[:self.num_samples])

            for i in range(self.num_samples):
                plt.figure(figsize=(15, 5))

                # Show input image
                plt.imshow(np.clip(X[i], 0, 1))  # Show all bands or first 3 if RGB
                plt.title('Input Image')
                plt.axis('off')

                # Show ground truth
                plt.subplot(132)
                plt.imshow(y_true[i, ..., 0], cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')

                # Show prediction
                plt.subplot(133)
                plt.imshow(y_pred[i, ..., 0], cmap='gray')
                plt.title('Refined Prediction')
                plt.subplot(131)
                plt.axis('off')

                plt.savefig(os.path.join(self.save_dir, f'epoch_{epoch}_sample_{i}.png'))
                plt.close()


def train(config, tiles_dir, masks_dir, model_path,  weights_path=None):
    
    # Match tiles and masks by filename
    tiles, masks = match_tile_mask_pairs(tiles_dir, masks_dir)

    #tiles = sorted(glob.glob(os.path.join(tiles_dir, '*.tif')))
    #masks = sorted(glob.glob(os.path.join(masks_dir, '*.tif')))

    #tiles, masks = filter_bad_tiles(tiles, masks)
    tiles, masks = filter_bad_tiles(tiles, masks, config)

    if len(tiles) != len(masks):
        raise ValueError(f"Number of tiles ({len(tiles)}) does not match number of masks ({len(masks)})")
    
    if len(tiles) == 0:
        raise ValueError(f"No .tif files found in tiles directory: {tiles_dir}")
        
    train_tiles, val_tiles, train_masks, val_masks = train_test_split(
        tiles, masks, 
        test_size=config['data']['validation_split'], 
        random_state=config['data']['random_state'],
    ) 


    print(f"\nDataset Information:")
    print(f"Training samples: {len(train_tiles)}")
    print(f"Validation samples: {len(val_tiles)}")
    print(f"Batch size: {config['data']['batch_size']}")
    

    if(config['data']['channels'] == 3):
        train_generator = Channel3_DataGenerator_old(train_tiles, train_masks, config, is_training=True)
        val_generator = Channel3_DataGenerator_old(val_tiles, val_masks, config, is_training=False)
        
    elif(config['data']['channels'] == 5):
        print("Using channel 5 DataGenerator")
        train_generator = Channel5_DataGenerator(train_tiles, train_masks, config, is_training=True)
        val_generator = Channel5_DataGenerator(val_tiles, val_masks, config, is_training=False)
    else:
        train_generator = Channel4_DataGenerator_latest(train_tiles, train_masks, config, is_training=True)
        val_generator = Channel4_DataGenerator_latest(val_tiles, val_masks, config, is_training=False)
        
    images, masks = train_generator[0]

    print("Mask batch shape :", masks.shape)
    
    model = create_unet_model(config)
    # model = create_old_unet_model(config)
    # model = create_old_enhanced_unet(config)
    # model = create_enhanced_unet(config)
    # model = create_unet_model(config)

    # print(model.summary())
    # Add this block to load weights if provided
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading pre-trained weights from: {weights_path}")
        model.load_weights(weights_path)

    else: 
        print()
        print("The weight file is not provided.")

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    initial_lr = config['model']['learning_rate']
    print("Image batch shape:", images.shape)
    decay_rate = 0.1
    decay_steps = 10 * (len(train_tiles)//config['data']['batch_size'])
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    def combined_bce_dice_boundary_loss(y_true, y_pred):
        """Combined weighted BCE + Dice + Boundary-aware loss"""
        # --- BCE with class weights ---
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        
        # Adjusted class weights based on new distribution
        weight_background = 1.0
        weight_water = 1 / 0.063  # Inverse class ratio
        weights = K.constant([weight_background, weight_water])
        bce_loss = K.sum(class_loglosses * weights[1])

        # --- Dice loss ---
        dice = dice_loss(y_true, y_pred)

        # --- Boundary loss ---
        edges_true = tf.image.sobel_edges(y_true)
        edges_pred = tf.image.sobel_edges(y_pred)
        edge_loss = K.mean(K.abs(edges_true - edges_pred))

        # --- Combine losses ---
        alpha = 0.65  # BCE contribution
        beta = 0.25   # Dice contribution
        gamma = 0.10  # Boundary loss contribution

        total_loss = (alpha * bce_loss) + (beta * dice) + (gamma * edge_loss)
        return total_loss

    def normal_weighted_binary_crossentropy(y_true, y_pred):
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        weights = config['training']['class_weights']
        return K.sum(class_loglosses * K.constant(weights))


    def dice_coef(y_true, y_pred, smooth=1.0):
        """Dice coefficient for binary segmentation"""
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    def dice_loss_old(y_true, y_pred):
        """Dice loss to minimize. Same as (1-dice_coefficient)"""
        return 1 - dice_coef(y_true, y_pred)

    def boundary_loss(y_true, y_pred):
        """Boundary loss function wrapper that handles dimensionality and types"""
        # Cast to the same data type
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Ensure y_true is squeezed to remove any extra dimensions
        y_true_squeezed = tf.squeeze(y_true, axis=-1)
        return BoundaryLoss()(y_pred, y_true_squeezed)

    def combined_bce_dice_loss(y_true, y_pred):
        """Combined weighted binary crossentropy and dice loss

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Combined loss value
        """

        # y_pred = tf.cast(y_pred, tf.float32)
        # y_true = tf.cast(y_true, tf.float32)
        # Original weighted BCE
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        weights = 0.75  # Your original class weight

        bce_loss = K.sum(class_loglosses * K.constant(weights))

        # Dice loss
        dice = dice_loss_old(y_true, y_pred)
        # bloss = boundary_loss(y_pred, y_true)
        # Combine losses with weights
        # You can adjust alpha to control the contribution of each loss
        alpha = 0.75  # BCE contribution
        return (alpha * bce_loss) + ((1 - alpha) * dice)  
        # return bce_loss + dice + bloss
    def combined_bce_dice_bd_loss(y_true, y_pred):
        """Combined weighted binary crossentropy and dice loss

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Combined loss value
        """

        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        # Original weighted BCE
        class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
        weights = 0.97  # Your original class weight

        bce_loss = K.sum(class_loglosses * K.constant(weights))

        # Dice loss
        dice = dice_loss_old(y_true, y_pred)
        bloss = boundary_loss(y_pred, y_true)
        # Combine losses with weights
        # You can adjust alpha to control the contribution of each loss
        alpha = 0.85  # BCE contribution
        #return (alpha * bce_loss) + ((1 - alpha) * dice)  
        return bce_loss + dice + bloss

    def dice_coefficient(y_true, y_pred, smooth=1):
        """Dice coefficient metric"""
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
        return K.mean((2. * intersection + smooth) / (union + smooth))


    def dice_loss(y_true, y_pred):
        """Dice loss function"""
        return 1 - dice_coefficient(y_true, y_pred)


    def boundary_aware_loss(y_true, y_pred):
        """Combined boundary-aware loss function"""
        # Binary cross-entropy
        bce = K.binary_crossentropy(y_true, y_pred)

        # Dice loss component
        dice = dice_loss(y_true, y_pred)

        # Edge continuity loss
        edges_true = tf.image.sobel_edges(y_true)
        edges_pred = tf.image.sobel_edges(y_pred)
        edge_loss = K.mean(K.abs(edges_true - edges_pred))

        return 0.35 * bce + 0.35 * dice + 0.3 * edge_loss

    # Usage in model.compile():
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=combined_bce_dice_boundary_loss,
        # loss=combined_bce_dice_bd_loss,
        #loss=boundary_aware_loss,
        #loss=BoundaryLoss(),
        #loss=normal_weighted_binary_crossentropy,
        metrics=['accuracy',
            BuildingIoU(name='building_iou'),
            BuildingRecall(name='building_recall'),
            BuildingPrecision(name='building_precision'),
            #dice_coef  # Add Dice coefficient as a metric
        ]
    )


    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_path}/best_model.keras",
            monitor='val_loss',
            mode='min',
            save_best_only=True,
           verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config['training']['reduce_lr_patience'],
            min_lr=config['training']['min_lr'],
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config['paths']['logs'],
            histogram_freq=1,
            update_freq='epoch'
        )
    ]

    callbacks_new = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, 'combined_model_weights_dice_{val_dice_coefficient:.4f}.h5'),
            monitor='val_dice_coefficient',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            patience=15,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config['paths']['logs'],
            histogram_freq=1,
            update_freq='epoch'
        ),
        CSVLogger('combined_training_log.csv')

        #VisualizationCallback(val_generator)
    ]

    
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['training']['epochs'],
        steps_per_epoch=len(train_tiles)//config['data']['batch_size'],
        validation_steps=len(val_tiles)//config['data']['batch_size'],
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )

    print("\nOptimizing prediction threshold...")
    #threshold_results = optimize_threshold(model, config)

    results = {
        'training_history': history.history
        #'threshold_optimization': threshold_results
    }

    #print(threshold_results)


    return model, results
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train U-Net model for building detection')
    parser.add_argument('--input_tiles_dir', required=True, help='Directory containing input tiles')
    parser.add_argument('--input_masks_dir', required=True, help='Directory containing input masks')
    parser.add_argument('--model_path', required=True, help='Directory for saving model')
    parser.add_argument('--weights_path', help='Path to pre-trained weights file for fine-tuning', default=None)

    args = parser.parse_args()

    if not os.path.exists(args.input_tiles_dir):
        raise ValueError(f"Tiles directory does not exist: {args.input_tiles_dir}")
    if not os.path.exists(args.input_masks_dir):
        raise ValueError(f"Masks directory does not exist: {args.input_masks_dir}")

    with open('config/config_v1.yaml', 'r') as f:
        config = yaml.safe_load(f)

    set_gpu()
    model, results = train(config, tiles_dir=args.input_tiles_dir, masks_dir=args.input_masks_dir, model_path=args.model_path, weights_path=args.weights_path)
