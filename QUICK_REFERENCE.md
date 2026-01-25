# 🌳 VEGETATION DETECTION - QUICK REFERENCE

## Installation

```bash
# Create environment
python -m venv veg_env
source veg_env/bin/activate  # On Windows: veg_env\Scripts\activate

# Install dependencies
pip install -r vegetation_requirements.txt
```

## Quick Start - 3 Commands

### Option A: Complete Pipeline (Train + Infer)
```bash
python vegetation_pipeline.py \
    --input_tiff ortho.tif \
    --vegetation_shp trees.shp \
    --output_dir ./output
```

### Option B: Step by Step

**Step 1: Create Tiles**
```bash
python create_vegetation_tiles.py \
    --input_tif data/ortho.tif \
    --input_shp data/trees.shp \
    --output_dir ./tiles_masks
```

**Step 2: Train Model**
```bash
cd pytorch_model_training
python vegetation_detection_training.py \
    --input_tiles_dir ../tiles_masks/tiles_veg \
    --input_masks_dir ../tiles_masks/masks_veg \
    --model_path ../models
cd ..
```

**Step 3: Inference**
```bash
python vegetation_inference.py \
    --input_image data/large_ortho.tif \
    --output_path predictions/veg_map.tif \
    --model_path models/vegetation_unet_best_*.pt \
    --threshold 0.5
```

---

## Key Features

### Input Requirements
- ✅ RGB TIFF (3 bands)
- ✅ 0.1m resolution (or adjust tile_size)
- ✅ Vegetation mask as Shapefile
- ✅ CRS must match between image and shapefile

### Outputs
- 📊 `prediction.tif` - RGB + confidence probability [0-1]
- 🔲 `prediction_binary.tif` - Thresholded vegetation mask
- 📈 Console statistics (coverage %, confidence)

### Automatically Computed Features
1. **ExG** - Excess Green: 2G - R - B
2. **NDVI-RGB** - (G-R)/(G+R)
3. **GLI** - Green Leaf Index: (2G-R-B)/(2G+R+B)
4. **Color Index** - G/(R+B)

---

## Common Parameters

### Tile Creation
```bash
--tile_size 512        # Pixels (51.2m at 0.1m res)
--overlap 0.25         # 25% overlap between tiles
```

### Training
```bash
--batch_size 8         # Adjust based on GPU memory
--epochs 150           # Total training epochs
--learning_rate 0.001  # Initial learning rate
```

### Inference
```bash
--tile_size 512        # Process tile size
--overlap 64           # Overlap in pixels
--threshold 0.5        # Vegetation confidence threshold
  0.3-0.4: Liberal (include potential vegetation)
  0.5:     Balanced (recommended)
  0.6-0.7: Conservative (only confident vegetation)
```

---

## Expected Performance

| Metric | Typical | Good | Excellent |
|--------|---------|------|-----------|
| IoU    | 0.70    | 0.75 | 0.85+     |
| F1     | 0.75    | 0.80 | 0.88+     |
| Recall | 0.80    | 0.85 | 0.90+     |

---

## Troubleshooting

### "No tiles created"
- Check shapefile and TIFF have same CRS
- Ensure shapefile has vegetation features
- Try reducing --overlap or adjusting --tile_size

### "Out of Memory"
- Reduce --tile_size (512 → 256)
- Reduce --batch_size (8 → 4)
- Use CPU: --device cpu

### "Low accuracy"
- Increase training samples (target: 1000+ tiles)
- Check image resolution matches training
- Verify mask quality/digitization

### "False positives"
- Increase --threshold (0.5 → 0.6-0.7)
- Retrain with hard negative examples
- Post-process with morphological operations

---

## GPU/CPU Performance

| Setting | Speed | Memory | Quality |
|---------|-------|--------|---------|
| GPU (CUDA) | 10-50x faster | 2-4GB | Same |
| CPU | Baseline | 1-2GB | Same |

### GPU Requirements
- NVIDIA GPU with 4GB+ VRAM
- CUDA 11.8+
- cuDNN compatible

### Enable GPU
```bash
--device cuda  # Automatic detection
```

---

## Model Architecture

```
Input (7 channels)
   ↓
Channel Adapter (7→3)
   ↓
UNet++ Encoder (SENet154 + ImageNet weights)
   ↓
Decoder with SCSE Attention
   ↓
Sigmoid Activation
   ↓
Output: Binary Vegetation Probability [0-1]
```

---

## Loss Function

Total Loss = α·BCE + β·Dice + γ·Boundary + δ·Focal

- **BCE**: Binary cross-entropy (basic classification)
- **Dice**: IoU-based loss (boundary accuracy)
- **Boundary**: Edge detection (tree/shrub edges)
- **Focal**: Hard pixel focus (difficult vegetation)

All weights (α, β, γ, δ) are learned during training.

---

## Post-Processing (Optional)

Improve binary predictions with morphological operations:

```python
import cv2
# Closing: fill small holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

# Opening: remove small noise
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
```

---

## Configuration File

Edit `config/config_v1.yaml`:

```yaml
data:
  input_size: 512          # Tile size
  channels: 3              # RGB input
  batch_size: 8            # Training batch
  validation_split: 0.2    # 80/20 train/val

model:
  learning_rate: 0.001
  
training:
  epochs: 150
  early_stopping_patience: 15
  min_lr: 1e-8
```

---

## Next Steps

1. **Prepare Data**: Digitize vegetation in shapefile
2. **Create Tiles**: Run `create_vegetation_tiles.py`
3. **Train Model**: Run `vegetation_detection_training.py`
4. **Validate**: Check accuracy on validation set
5. **Infer**: Run `vegetation_inference.py` on new images
6. **Post-process**: Apply morphological operations if needed
7. **Export**: Convert to your required format

---

## Advanced: Multi-Season Training

Train separate models for different seasons:

```bash
# Spring/Summer
python vegetation_pipeline.py --input_tiff spring.tif --vegetation_shp spring_veg.shp --output_dir spring_model

# Fall/Winter
python vegetation_pipeline.py --input_tiff fall.tif --vegetation_shp fall_veg.shp --output_dir fall_model
```

Ensemble predictions:
```python
pred_spring = load_tif('spring_pred.tif')
pred_fall = load_tif('fall_pred.tif')
ensemble = (pred_spring + pred_fall) / 2
```

---

## Resources

- **PyTorch Docs**: https://pytorch.org/docs
- **Segmentation Models**: https://github.com/qubvel/segmentation_models.pytorch
- **Rasterio Docs**: https://rasterio.readthedocs.io
- **Albumentations**: https://albumentations.ai

---

**Version**: 1.0  
**Last Updated**: 2024  
**Status**: Production Ready ✅
