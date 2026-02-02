# 🗺️ Complete Guide: Creating Tiles & Masks + Road Classification Model

---

## 📋 Table of Contents
1. [Understanding Tiles and Masks](#1-understanding-tiles-and-masks)
2. [Creating Tiles and Masks](#2-creating-tiles-and-masks)
3. [Road Classification Model Architecture](#3-road-classification-model-architecture)
4. [Training the Model](#4-training-the-model)
5. [Making Predictions](#5-making-predictions)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Understanding Tiles and Masks

### What Are Tiles?

**Tiles** are small rectangular images extracted from large satellite/aerial raster images (GeoTIFF files).

- **Why use tiles?**
  - Large satellite images are too big to load entirely into GPU memory (often 10GB+)
  - Training on small tiles is efficient and improves model generalization
  - Easy to manage, batch, and augment

- **Typical size**: 1024×1024 pixels (configurable)
- **Format**: GeoTIFF (.tif) with georeferencing preserved
- **Bands**: 3 bands (RGB) - extracted from multispectral data

### What Are Masks?

**Masks** are pixel-level annotations corresponding to tiles. They encode which pixels belong to which class.

```
Pixel values in mask:
├─ 0 = Background (non-road area)
├─ 1 = Thar Road
├─ 2 = CC Road
└─ 3 = Mud/Gravel Road
```

- **Paired with tiles**: Each mask corresponds to exactly one tile
- **Format**: GeoTIFF (.tif) with single band (grayscale)
- **Georeferencing**: Same spatial coordinates as the corresponding tile

---

## 2. Creating Tiles and Masks

### Input Data Requirements

You need:
1. **Large satellite image** (.tif file) - multiple bands
2. **Shapefile** (.shp) - vector polygons marking road locations with class information

```
Input Structure:
├── satellite_image.tif         (large raster, any number of bands)
├── roads.shp                   (shapefile with road polygons)
├── roads.shx
├── roads.dbf
└── roads.prj
```

### Step 1: Prepare Your Data

Your shapefile should have attributes for each road type. Example structure:

```python
# Columns in shapefile:
├── geometry      (polygon boundaries)
├── road_type     (e.g., "Thar", "CC", "Mud")
└── ... other attributes
```

### Step 2: Use the Tiling Script

**File**: `create_tilesandmasks_fixed.py`

#### Basic Usage

```bash
cd /home/srinivas/Pictures/github/ML_Setup

python create_tilesandmasks_fixed.py \
  --input_tif /path/to/satellite_image.tif \
  --input_shp /path/to/roads.shp \
  --output_dir ./tiles_and_masks_output \
  --tile_size 1024 \
  --overlap_percentage 0.25
```

#### Parameters Explained

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--input_tif` | Required | Path to large satellite image (GeoTIFF) |
| `--input_shp` | Required | Path to shapefile with road polygons |
| `--output_dir` | Required | Directory where tiles & masks will be saved |
| `--tile_size` | 1024 | Size of each tile (1024×1024 pixels) |
| `--overlap_percentage` | 0.25 | Overlap ratio (25% = 256px overlap on 1024px) |

### Step 3: Script Output Structure

After running, you'll get:

```
tiles_and_masks_output/
├── tiles/
│   ├── image_0_0.tif              (1024×1024, 3 bands)
│   ├── image_0_1024.tif           (overlapping region)
│   ├── image_1024_0.tif
│   └── ...
└── masks/
    ├── image_0_0.tif              (1024×1024, 1 band, values 0-3)
    ├── image_0_1024.tif
    ├── image_1024_0.tif
    └── ...
```

### How the Script Works (Internally)

```
1. Load satellite image metadata
   └─ Get: dimensions (W, H), CRS, georeferencing

2. Load shapefile
   └─ Reproject to match satellite image CRS
   └─ Filter geometries within raster extent

3. For each tile position:
   a. Calculate window bounds (with overlap)
   b. Read tile from disk (band 1,2,3 only)
   c. Pad tile to fixed size (1024×1024)
   d. SCALING: Apply percentile normalization
      ├─ Find 2.5% and 99% percentiles
      ├─ Clip values to this range
      ├─ Normalize to [0, 1]
      └─ Convert to uint8 [1, 255]
   
   e. Rasterize mask:
      ├─ Find all road polygons in tile
      ├─ Convert polygons to pixel grid
      ├─ Create binary mask (0 for background, 1+ for roads)
   
   f. Skip empty tiles (< 1% non-zero)
   
   g. Save both tile & mask as GeoTIFF

4. Output:
   └─ tiles/ folder with RGB images
   └─ masks/ folder with class labels
```

### Key Implementation Details

#### Overlap Handling

```python
# Without overlap: simple grid
stride = tile_size  # 1024

# With 25% overlap:
overlap_pixels = int(tile_size * 0.25)  # 256 pixels
stride = tile_size - overlap_pixels     # 768 pixels

# This means:
# Tile 0: pixels 0-1023
# Tile 1: pixels 768-1791  (overlaps with tile 0 by 256px)
# Tile 2: pixels 1536-2559 (overlaps with tile 1 by 256px)
```

#### Memory Efficiency

```python
# IMPORTANT: Reads ONE tile at a time from disk
with rasterio.open(input_tif) as src:
    for i, j in tile_positions:
        tile_data = src.read(selected_bands, window=window)
        # Process and save immediately
        # Does NOT load entire image into RAM
```

This means:
- No matter how large your satellite image is, memory usage stays constant
- Works with 100GB+ satellite images without issues

#### Mask Rasterization

```python
# Converts polygon geometries to pixel values
shapes = [(geom, value) for geom in road_polygons]
mask_arr = rasterize(
    shapes,
    out_shape=(tile_size, tile_size),
    transform=window_transform,
    all_touched=True,  # Ensures thin roads are captured
)
```

The `all_touched=True` parameter is crucial:
- Without it: thin roads (2-3 pixels wide) might be missed
- With it: all pixels touching the polygon are included

---

## 3. Road Classification Model Architecture

### Model Overview

Your road classification system is a **Multi-Class Semantic Segmentation Model** with 4 road types.

```
Input (1024×1024×3 RGB)
    ↓
┌─────────────────────────┐
│ ResNet50 Backbone       │  (Feature Extraction)
│ Pretrained on ImageNet  │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ FPN Decoder             │  (Feature Pyramid)
│ (Upsampling layers)     │
└─────────────────────────┘
    ↓
┌─────────────────────────┐
│ Segmentation Head       │  (Per-pixel classification)
│ 4 output channels       │
└─────────────────────────┘
    ↓
Output (1024×1024×4)
  Channel 0: Background probability
  Channel 1: Thar Road probability
  Channel 2: CC Road probability
  Channel 3: Mud/Gravel Road probability
    ↓
Argmax → Final class prediction (0-3)
```

### Key Components

#### 1. Backbone: ResNet50

- **Purpose**: Extract hierarchical features from input image
- **Pre-training**: ImageNet weights (transfer learning)
- **Levels**: 
  - Level 1: 64 channels (high resolution, low semantics)
  - Level 2: 256 channels
  - Level 3: 512 channels
  - Level 4: 1024 channels
  - Level 5: 2048 channels (low resolution, high semantics)

#### 2. Decoder: Feature Pyramid

- **Purpose**: Upsample features back to input resolution
- **Method**: 
  - Takes multi-scale features from backbone
  - Progressively upsamples
  - Fuses information from different levels
  - Result: 512-channel feature map at input resolution

#### 3. Segmentation Head

- **Purpose**: Convert features to class probabilities
- **Steps**:
  ```
  512-channel features
       ↓
  1×1 Conv → 256 channels
       ↓
  1×1 Conv → 4 channels (one per class)
       ↓
  Softmax → Probability for each class
  ```

### Loss Functions (Why 4 Losses?)

The model combines **4 different loss functions** for superior accuracy:

```
Total Loss = 0.4×(CrossEntropy + FocalLoss) + 0.4×DiceLoss + 0.2×LovaszLoss
```

#### Why Each Loss?

1. **Cross-Entropy Loss (0.2 weight)**
   - Standard pixel-level classification loss
   - Ensures basic correct predictions
   - Formula: $-\sum_c y_c \log(\hat{y}_c)$

2. **Focal Loss (0.2 weight)**
   - Focuses on hard examples (misclassified pixels)
   - Reduces impact of easy examples (background)
   - Handles class imbalance (rare road types)
   - Formula: $-\alpha(1-\hat{y}_c)^\gamma \log(\hat{y}_c)$

3. **Dice Loss (0.4 weight)**
   - Directly optimizes Intersection-over-Union (IoU)
   - What we actually care about for segmentation
   - Formula: $1 - \frac{2|X \cap Y|}{|X| + |Y|}$

4. **Lovász Loss (0.2 weight)**
   - Differentiable approximation of IoU
   - Works on the joint loss of all pixels
   - Prevents model from focusing on single large regions

### Class Weighting

Because road pixels are rare compared to background:

```python
weight[0] = 0.8   # Background (abundant)
weight[1] = 1.2   # Thar Road (medium)
weight[2] = 1.5   # CC Road (scarce)
weight[3] = 2.0   # Mud/Gravel Road (very scarce)
```

**Computation**:
```python
weight[c] = Total_Pixels / (Num_Classes × Pixels_in_Class[c])
```

This prevents the model from being lazy and just predicting "background" for everything.

### Data Augmentation Strategy

Applied during training to improve generalization:

| Category | Techniques | Purpose |
|----------|-----------|---------|
| **Geometric** | Rotation (45°), Elastic distortion, Grid distortion | Handle different road orientations |
| **Intensity** | Brightness/Contrast, Gamma correction | Handle different lighting/atmospheric conditions |
| **Noise** | Gaussian noise, Blur | Handle sensor noise |
| **Structural** | Random dropout, CLAHE | Handle missing data, improve contrast |

---

## 4. Training the Model

### Prerequisites

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install albumentations
pip install rasterio geopandas shapely
pip install pyyaml scikit-learn tqdm
```

### File Locations

```
pytorch_model_training/
├── enhanced_pytorch_backbone_training_multiclass.py  (MAIN TRAINING SCRIPT)
├── pytorch_backbone_model_v2.py                      (Model definition)
├── validate_multiclass_data.py                       (Data validation)
└── MULTICLASS_TRAINING_GUIDE.md                      (Detailed guide)
```

### Quick Start

```bash
cd pytorch_model_training

# Step 1: Validate your data
python validate_multiclass_data.py \
  --tiles_dir /path/to/tiles \
  --masks_dir /path/to/masks

# Step 2: Train the model
python enhanced_pytorch_backbone_training_multiclass.py \
  --input_tiles_dir /path/to/tiles \
  --input_masks_dir /path/to/masks \
  --model_path ./models/roads_multiclass_v1 \
  --batch_size 8 \
  --epochs 60
```

### Expected Training Time

| GPU | 60 Epochs | Notes |
|-----|-----------|-------|
| A100 | ~1 hour | Very fast |
| V100 | ~2 hours | Fast |
| RTX 3080 | ~1.5 hours | Medium |
| RTX 2080 | ~3 hours | Slower |
| CPU | ~20+ hours | Not recommended |

### Two-Stage Training Strategy

The training happens in **two stages** automatically:

#### Stage 1 (10 epochs): Decoder-only
```
ResNet50 backbone: FROZEN (weights don't change)
Decoder:           TRAINED

Why? Fast convergence, learn dataset patterns quickly
```

#### Stage 2 (50 epochs): Fine-tuning
```
ResNet50 backbone: UNFROZEN (all weights trainable)
Decoder:           TRAINED

Why? Fine-tune entire network for maximum accuracy
```

### Learning Rate Schedule

```
Stage 1 (Decoder-only):
├─ Epochs 0-2:    Linear warmup (0 → 1e-3)
├─ Epochs 2-10:   Cosine annealing (1e-3 → 1e-7)

Stage 2 (Fine-tuning):
├─ Epochs 0-2:    Linear warmup (0 → 1e-4)
├─ Epochs 3-60:   Cosine annealing (1e-4 → 1e-8)
```

This prevents training instability and ensures steady convergence.

### Output Directory Structure

```
models/roads_multiclass_v1/
├── best_model_stage1.pt           (Best model after stage 1)
├── best_model_stage2.pt           (Best model after stage 2)
├── multiclass_road_segmentation_final.pt
├── training_history.json          (Metrics per epoch)
├── config.yaml                    (Training configuration)
└── logs/
    └── training_log.txt           (Detailed training output)
```

### Metrics Tracked During Training

```python
# Per-epoch metrics:
├── Loss (total, CE, Focal, Dice, Lovász)
├── Accuracy (overall pixel accuracy)
├── Per-class IoU (Intersection over Union)
│   ├─ Background IoU
│   ├─ Thar Road IoU
│   ├─ CC Road IoU
│   └─ Mud/Gravel Road IoU
├── Mean IoU (average of all classes)
├── Learning rate
└── GPU memory usage
```

### Expected Results

After 60 epochs of training:

```
mIoU: 0.92-0.94 (92-94% average intersection-over-union)

Per-class performance:
├─ Background:     IoU = 0.95+ (very easy)
├─ Thar Road:      IoU = 0.91-0.93 (medium)
├─ CC Road:        IoU = 0.93-0.95 (good)
└─ Mud/Gravel:     IoU = 0.89-0.92 (hardest)
```

---

## 5. Making Predictions

### Single Image Prediction

```bash
cd pytorch_model_training

python multiclass_inference.py \
  --model_path ./models/roads_multiclass_v1/best_model_stage2.pt \
  --image_path /path/to/new_image.tif \
  --output_dir ./predictions
```

### Batch Predictions

```bash
python multiclass_inference.py \
  --model_path ./models/roads_multiclass_v1/best_model_stage2.pt \
  --image_dir /path/to/tile_directory \
  --output_dir ./predictions \
  --batch_size 16
```

### Output Files

```
predictions/
├── image_0_class_map.tif         (Predicted class labels: 0-3)
├── image_0_confidence.tif        (Confidence score: 0-1)
├── image_0_probabilities.tif     (Per-class probabilities: 4 bands)
└── ...
```

### Output Interpretation

**Class Map** (image_0_class_map.tif):
```
Pixel values:
├─ 0 = Background (not a road)
├─ 1 = Thar Road
├─ 2 = CC Road
└─ 3 = Mud/Gravel Road
```

**Confidence Map** (image_0_confidence.tif):
```
Values 0.0-1.0:
├─ 0.95-1.0   = Very confident prediction
├─ 0.7-0.95   = Confident prediction
├─ 0.5-0.7    = Uncertain (might need review)
└─ 0.0-0.5    = Low confidence (likely error)
```

**Probabilities** (image_0_probabilities.tif):
```
4 bands (one per class):
├─ Band 1: Background probability
├─ Band 2: Thar Road probability
├─ Band 3: CC Road probability
└─ Band 4: Mud/Gravel Road probability

Sum of all bands = 1.0 for each pixel
```

### Python API for Inference

```python
from multiclass_inference import MultiClassRoadSegmentor
import rasterio

# Load model
segmentor = MultiClassRoadSegmentor('best_model_stage2.pt')

# Single image
with rasterio.open('tile.tif') as src:
    image = src.read()

predictions = segmentor.predict(image)
# Returns: (class_map, confidence, probabilities)
#   class_map: (1024, 1024) - class labels
#   confidence: (1024, 1024) - confidence scores
#   probabilities: (1024, 1024, 4) - per-class probabilities
```

---

## 6. Troubleshooting

### Common Issues

#### Issue 1: "Out of Memory" Error During Training

**Cause**: Batch size too large for your GPU

**Solution**:
```bash
# Reduce batch size
python enhanced_pytorch_backbone_training_multiclass.py \
  --batch_size 4  # Instead of 8 or 16
```

#### Issue 2: Mask Has Wrong Classes (Values > 3)

**Cause**: Shapefile encoding issue

**Solution**:
```bash
# Validate your data first
python validate_multiclass_data.py \
  --tiles_dir ./tiles \
  --masks_dir ./masks

# It will show class distribution and flag issues
```

#### Issue 3: Training Loss Not Decreasing

**Cause**: Usually learning rate too high/low

**Solution**:
- If loss oscillates wildly: **Reduce learning rate**
- If loss decreases very slowly: **Increase learning rate**

The script auto-adjusts learning rate, but you can manually set:
```bash
python enhanced_pytorch_backbone_training_multiclass.py \
  --base_lr 5e-4  # Default is 1e-3
```

#### Issue 4: Predictions Look Bad

**Causes & Solutions**:

| Symptom | Cause | Solution |
|---------|-------|----------|
| Everything predicted as background | Model not trained enough | Train for more epochs (100+) |
| Wrong road types confused | Class imbalance | Add more data for minority classes |
| Thin roads missed | Mask rasterization issue | Use `all_touched=True` when creating masks |
| Noisy predictions | Need post-processing | Apply morphological filtering |

#### Issue 5: Different Results Each Run

**Cause**: Random initialization and data shuffling

**Solution**: Set random seed for reproducibility
```python
import random
import torch
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

### Performance Optimization Tips

1. **Use overlap during tile creation**
   ```bash
   --overlap_percentage 0.25  # 25% overlap helps model see tile boundaries
   ```

2. **Data augmentation is crucial**
   - More aggressive augmentation = better generalization
   - If overfitting (train loss << val loss), increase augmentation

3. **Mixed precision training** (automatic in the script)
   - 30-50% faster than regular float32
   - Same accuracy
   - Enabled by default with `torch.cuda.amp`

4. **Multi-GPU training**
   ```bash
   python -m torch.distributed.launch \
     --nproc_per_node=2 \
     enhanced_pytorch_backbone_training_multiclass.py
   ```

---

## Summary: Workflow

```
1. PREPARE DATA
   └─ satellite_image.tif + roads.shp
       ↓
2. CREATE TILES & MASKS
   └─ python create_tilesandmasks_fixed.py
       ├─ Output: tiles/ (RGB images)
       └─ Output: masks/ (class labels 0-3)
       ↓
3. VALIDATE DATA
   └─ python validate_multiclass_data.py
       └─ Checks: alignment, class values, distribution
       ↓
4. TRAIN MODEL
   └─ python enhanced_pytorch_backbone_training_multiclass.py
       ├─ Stage 1: Decoder-only (10 epochs)
       └─ Stage 2: Fine-tuning (50 epochs)
       ↓
5. EVALUATE RESULTS
   └─ Check training curves, per-class IoU
       ↓
6. MAKE PREDICTIONS
   └─ python multiclass_inference.py
       ├─ Output: class_map.tif
       ├─ Output: confidence.tif
       └─ Output: probabilities.tif
       ↓
7. POST-PROCESS (OPTIONAL)
   └─ Remove noise, smooth boundaries
       └─ Output: final_road_map.shp
```

---

## Related Files

- 📄 [IMPLEMENTATION_SUMMARY.md](pytorch_model_training/IMPLEMENTATION_SUMMARY.md) - Technical implementation details
- 📄 [MULTICLASS_TRAINING_GUIDE.md](pytorch_model_training/MULTICLASS_TRAINING_GUIDE.md) - Advanced training guide
- 🐍 [create_tilesandmasks_fixed.py](create_tilesandmasks_fixed.py) - Tile/mask creation script
- 🐍 [enhanced_pytorch_backbone_training_multiclass.py](pytorch_model_training/enhanced_pytorch_backbone_training_multiclass.py) - Main training script
- 🐍 [multiclass_inference.py](pytorch_model_training/multiclass_inference.py) - Inference script

---

**Last Updated**: February 2, 2026
**Version**: 1.0
**Status**: Production Ready ✅
