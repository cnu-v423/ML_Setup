# 🌳 VEGETATION DETECTION MODEL - COMPREHENSIVE GUIDE

## Overview
This guide covers the complete pipeline for vegetation detection (trees, shrubs, forest canopy) using RGB satellite imagery at 0.1m resolution.

## Key Improvements Made

### 1. **Removed Waterbody Detection Features**
   - ❌ Removed Sobel filter preprocessing (only relevant for water detection)
   - ❌ Removed unnecessary padding and grayscale conversions
   - ✅ Simplified preprocessing pipeline for vegetation

### 2. **Added Vegetation-Specific Features**
   The model now computes and uses these vegetation indices from RGB:

   - **ExG (Excess Green)**: `2*G - R - B`
     - Specifically designed for vegetation detection
     - Separates green vegetation from soil/buildings
   
   - **NDVI-like from RGB**: `(G - R) / (G + R)`
     - Normalized difference based on red-green channels
     - Captures vegetation vigor
   
   - **GLI (Green Leaf Index)**: `(2*G - R - B) / (2*G + R + B)`
     - Sensitive to leaf chlorophyll content
     - Better than NDVI for RGB-only data
   
   - **Color Index**: `G / (R + B)`
     - Simple greenness measure
     - Robust indicator of vegetation presence

### 3. **Optimized Model Architecture**
   - **Base Model**: UNet++ with SENet154 encoder
   - **Decoder**: SCSE (Spatial and Channel Squeeze-Excitation) attention
   - **Input**: 7 channels (RGB + 4 vegetation indices)
   - **Channel Adapter**: 1×1 convolution to adapt 7 channels to ImageNet pretrained encoder
   - **Output**: Binary vegetation probability map

### 4. **Improved Loss Function**
   Combined multiple loss terms specifically for vegetation:
   
   - **Adaptive BCE Loss**: Binary cross-entropy with learnable weights
   - **Dice Loss**: IoU-based loss for boundary accuracy
   - **Boundary Loss**: Sobel edge detection to emphasize vegetation edges (trees/shrubs)
   - **Focal Loss**: Focuses on hard-to-predict vegetation pixels
   
   Total Loss = α·BCE + β·Dice + γ·Boundary + δ·Focal

### 5. **Enhanced Data Augmentation**
   - Geometric: Flips, rotations, elastic transforms
   - Photometric: Brightness, contrast, hue/saturation changes
   - Spectral: Channel dropout (simulate missing spectral bands)
   - Spatial: Random crops for scale variety

### 6. **Tile Creation Optimization**
   - Adaptive scaling based on vegetation content
   - Filters out tiles with <1% vegetation content
   - Maintains 25% overlap for edge seamless prediction
   - Optimized for 512×512 tiles (51.2m × 51.2m at 0.1m resolution)

## File Structure

```
ML_Setup/
├── pytorch_model_training/
│   ├── vegetation_detection_training.py    ✨ NEW: Optimized training script
│   ├── enhanced_pytorch_backbone_training_advanced.py  (deprecated)
│   └── pytorch_backbone_model_v2.py
├── vegetation_inference.py                  ✨ NEW: Optimized inference
├── ensemble_triton_with_waterbody_test.py  (for other tasks)
├── create_vegetation_tiles.py               ✨ NEW: Tile creation for vegetation
├── create_tilesandmasks_fixed.py           (general purpose)
└── config/
    └── config_v1.yaml
```

## Workflow

### Step 1: Create Vegetation Tiles
```bash
python create_vegetation_tiles.py \
    --input_tif /path/to/rgb_image.tif \
    --input_shp /path/to/vegetation_mask.shp \
    --output_dir /path/to/tiles_masks \
    --tile_size 512 \
    --overlap 0.25
```

**Output**:
- `tiles_veg/`: RGB tiles at 512×512 pixels
- `masks_veg/`: Binary vegetation masks

**Expected**:
- Input: 0.1m resolution RGB TIFF
- Output: 512px tiles = 51.2m × 51.2m ground area
- Automatic filtering of low-vegetation tiles

### Step 2: Train Model
```bash
cd pytorch_model_training

python vegetation_detection_training.py \
    --input_tiles_dir /path/to/tiles_veg \
    --input_masks_dir /path/to/masks_veg \
    --model_path /path/to/save/model \
    --weights_path /path/to/pretrained/weights (optional)
```

**Configuration** (config_v1.yaml):
```yaml
data:
  input_size: 512
  channels: 3  # RGB (expanded to 7 with vegetation indices)
  batch_size: 8
  validation_split: 0.2
  random_state: 42

model:
  learning_rate: 0.001

training:
  epochs: 150
  early_stopping_patience: 15
  reduce_lr_patience: 5
  min_lr: 1e-8
```

**Training Features**:
- ✅ Two-stage training (frozen backbone → fine-tuning)
- ✅ Warmup cosine annealing learning rate
- ✅ Early stopping based on validation IoU
- ✅ Adaptive loss parameter tracking
- ✅ GPU acceleration with DataParallel support
- ✅ Real-time metrics (F1, IoU, Precision, Recall)

**Expected Results**:
- Best model saved as `vegetation_unet_best_[timestamp].pt`
- Final model saved as `vegetation_unet_final_[timestamp].pt`
- Target metrics: IoU > 0.75, F1 > 0.80

### Step 3: Run Inference
```bash
python vegetation_inference.py \
    --input_image /path/to/large_image.tif \
    --output_path /path/to/output/prediction.tif \
    --model_path /path/to/model/vegetation_unet_best.pt \
    --tile_size 512 \
    --overlap 64 \
    --threshold 0.5 \
    --device cuda
```

**Inference Features**:
- ✅ Memory-efficient tiling strategy
- ✅ Overlapping tiles with blending
- ✅ Generates:
  - Probability map (float32): `prediction.tif` (4 bands: RGB + probability)
  - Binary map (uint8): `prediction_binary.tif` (thresholded at 0.5)
- ✅ Vegetation coverage statistics
- ✅ Progress tracking

**Output**:
- `prediction.tif`: Original RGB + confidence probability [0-1]
- `prediction_binary.tif`: Binary vegetation mask [0-255]
- Console statistics: Coverage %, mean confidence, median confidence

## Key Parameters Explained

### Tile Size: 512 pixels
- At 0.1m resolution = 51.2m × 51.2m
- Good for capturing individual trees and small forest patches
- Balances memory usage and context

### Overlap: 64 pixels (12.5%)
- Ensures seamless predictions across tile boundaries
- Overlapping regions blended using weight averaging

### Threshold: 0.5
- Vegetation confidence threshold for binary classification
- Adjust based on:
  - 0.3-0.4: More liberal, captures all potential vegetation
  - 0.5: Balanced, recommended default
  - 0.6-0.7: Conservative, only confident vegetation

### Batch Size: 8
- Adjusted for GPU memory constraints
- Increase if GPU memory available (improves training speed)
- Decrease if out-of-memory errors

## Vegetation Detection Challenges & Solutions

### Challenge 1: Similar Colors (Trees vs. Buildings)
**Solution**: Using GLI and color indices which are sensitive to chlorophyll

### Challenge 2: Seasonal Variation
**Solution**: 
- Aggressive data augmentation (brightness, hue, saturation changes)
- Training on multi-seasonal data if available

### Challenge 3: Small Trees vs. Noise
**Solution**: 
- Boundary loss emphasizes tree edges
- Focal loss focuses on hard pixels

### Challenge 4: Forest Canopy Overlap
**Solution**:
- ExG index designed specifically for mixed spectral responses
- Model learns to handle overlapping tree crowns

## Expected Accuracy

Based on 0.1m resolution RGB data with proper vegetation digitization:

| Metric | Expected | Range |
|--------|----------|-------|
| IoU (vegetation) | 0.75-0.85 | 0.70-0.90 |
| F1 Score | 0.80-0.88 | 0.75-0.92 |
| Recall | 0.85-0.90 | 0.80-0.95 |
| Precision | 0.76-0.85 | 0.70-0.90 |

**Factors affecting accuracy**:
- ✅ Training data quality (digitization accuracy)
- ✅ Training data quantity (minimum 500-1000 tiles recommended)
- ✅ Image resolution (0.1m is excellent)
- ✅ Seasonal consistency
- ✅ Model fine-tuning

## Inference on Different Image Sizes

### Small Images (<2GB)
```bash
# Single pass, fast
python vegetation_inference.py \
    --input_image image_small.tif \
    --output_path output.tif \
    --tile_size 512 \
    --overlap 64
```

### Large Images (2-10GB)
```bash
# Efficient tiling with memory management
python vegetation_inference.py \
    --input_image image_large.tif \
    --output_path output.tif \
    --tile_size 512 \
    --overlap 64 \
    --device cuda  # Ensure GPU available
```

### Very Large Images (>10GB)
```bash
# Process in multiple passes if needed
# Or use multiprocessing wrapper
python vegetation_inference.py \
    --input_image image_huge.tif \
    --output_path output.tif \
    --tile_size 256  # Smaller tiles
    --overlap 32
    --device cuda
```

## Troubleshooting

### Issue: Low Accuracy on New Images
**Solution**:
- Check if images are in similar season as training data
- Verify image resolution matches (should be ~0.1m)
- Ensure RGB bands are in correct order

### Issue: Out of Memory During Inference
**Solution**:
- Reduce tile size (512 → 256)
- Reduce batch processing
- Use CPU mode (slower but less memory)

### Issue: Predictions Too Conservative (Confidence < 0.3)
**Solution**:
- Lower threshold (0.5 → 0.3-0.4)
- Check if training had enough diverse samples
- Verify augmentation parameters

### Issue: Predictions Include Non-vegetation (False Positives)
**Solution**:
- Increase threshold (0.5 → 0.6-0.7)
- Retrain model with harder examples
- Check training mask quality

## Advanced: Model Improvement

### To improve accuracy further:

1. **Add More Training Data**
   - Minimum: 500 tiles, Recommended: 1000+ tiles
   - Ensure diverse seasonal/lighting conditions

2. **Include Related Bands**
   - If available: NIR band for better vegetation detection
   - Thermal band for tree canopy vs. grass distinction

3. **Ensemble Multiple Models**
   - Train models on different seasons
   - Combine predictions for robustness

4. **Post-processing**
   - Morphological operations (closing, opening)
   - CRF (Conditional Random Field) smoothing
   - Connected component filtering

## References & Best Practices

### For RGB-based Vegetation Detection:
- **Excess Green Index (ExG)**: Meyer & Neto (2008)
- **Green Leaf Index (GLI)**: Louhaichi et al. (2001)
- **U-Net Architecture**: Ronneberger et al. (2015)
- **Focal Loss**: Lin et al. (2017)

### Hyperparameter Tuning:
- Learning rate: Start at 1e-3, decrease by 10x for fine-tuning
- Batch size: Larger (16, 32) if GPU memory allows
- Augmentation: Increase if validation gap is large
- Early stopping patience: 10-20 epochs typical

## Support & Questions

For issues or improvements:
1. Check the configuration file (config_v1.yaml)
2. Review training logs for convergence issues
3. Verify input data format and resolution
4. Test on a small subset before large-scale inference

---
**Last Updated**: 2024
**Model Version**: Vegetation Detection v1.0
**Input**: RGB Satellite Imagery (0.1m resolution)
**Output**: Binary Vegetation Mask + Confidence Map
