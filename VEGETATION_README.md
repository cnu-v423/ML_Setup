# 🌳 VEGETATION DETECTION MODEL - RGB Satellite Imagery

> **Detect trees, shrubs, and forest canopy from high-resolution RGB satellite imagery**

## 📋 Overview

A complete deep learning pipeline for detecting vegetation (trees, shrubs, forest canopy) from RGB satellite imagery at 0.1m resolution. The system uses optimized U-Net++ architecture with vegetation-specific features and adaptive loss functions.

### Key Features
- ✅ **Vegetation Indices**: Automatically computed ExG, NDVI-RGB, GLI, Color Index
- ✅ **Advanced Architecture**: U-Net++ with SENet154 encoder + SCSE attention
- ✅ **Adaptive Loss**: Combined BCE + Dice + Boundary + Focal loss
- ✅ **Efficient Inference**: Tiled processing for images of any size
- ✅ **Production Ready**: Optimized for deployment with GPU/CPU support
- ✅ **Comprehensive Tools**: Training, inference, ensemble, comparison utilities

---

## 🚀 Quick Start

### 1. Installation
```bash
# Create environment
python -m venv veg_env
source veg_env/bin/activate

# Install dependencies
pip install -r vegetation_requirements.txt
```

### 2. Prepare Data
- RGB TIFF file (0.1m resolution)
- Shapefile with vegetation digitization (same CRS as TIFF)

### 3. Run Complete Pipeline
```bash
python vegetation_pipeline.py \
    --input_tiff ortho.tif \
    --vegetation_shp trees.shp \
    --output_dir ./output
```

That's it! The pipeline will:
1. Create tiles from your data
2. Train a vegetation detection model
3. Run inference on the input image
4. Generate probability and binary maps

---

## 📁 File Structure

```
ML_Setup/
├── 🌳 NEW VEGETATION FILES
│   ├── pytorch_model_training/
│   │   └── vegetation_detection_training.py      # Training script
│   ├── vegetation_inference.py                   # Inference
│   ├── create_vegetation_tiles.py                # Tile creation
│   ├── vegetation_pipeline.py                    # Orchestration
│   ├── vegetation_ensemble.py                    # Ensemble tools
│   ├── vegetation_requirements.txt               # Dependencies
│   ├── config/
│   │   └── config_vegetation.yaml               # Configuration
│   │
│   ├── 📚 DOCUMENTATION
│   ├── VEGETATION_DETECTION_GUIDE.md            # Complete guide
│   ├── IMPLEMENTATION_SUMMARY.md                 # Technical details
│   ├── QUICK_REFERENCE.md                       # Cheat sheet
│   └── README.md                                 # This file
│
├── 🔄 LEGACY FILES (still available)
│   ├── enhanced_pytorch_backbone_training_advanced.py
│   ├── ensemble_triton_with_waterbody_test.py
│   ├── create_tilesandmasks_fixed.py
│   └── ...others
```

---

## 🎯 Typical Workflow

### Step 1: Prepare Data
```bash
# Organize your files
data/
├── ortho.tif              # RGB satellite image (0.1m)
└── trees.shp            # Vegetation digitization
```

### Step 2: Create Training Tiles
```bash
python create_vegetation_tiles.py \
    --input_tif data/ortho.tif \
    --input_shp data/trees.shp \
    --output_dir ./training_data
```

### Step 3: Train Model
```bash
cd pytorch_model_training
python vegetation_detection_training.py \
    --input_tiles_dir ../training_data/tiles_veg \
    --input_masks_dir ../training_data/masks_veg \
    --model_path ../models
cd ..
```

### Step 4: Run Inference
```bash
python vegetation_inference.py \
    --input_image data/large_ortho.tif \
    --output_path predictions/veg_map.tif \
    --model_path models/vegetation_unet_best_*.pt \
    --threshold 0.5
```

### Step 5: Post-Process (Optional)
```python
# Apply morphological operations
import cv2
import rasterio

# Load binary map
with rasterio.open('predictions/veg_map_binary.tif') as src:
    binary = src.read(1)

# Clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
```

---

## 🔬 Technical Details

### Vegetation Indices
The model automatically computes 4 vegetation indices from RGB:

```python
# Formula implementations
ExG = 2*G - R - B                           # Excess Green
NDVI_RGB = (G - R) / (G + R + ε)           # RGB-NDVI approximation
GLI = (2*G - R - B) / (2*G + R + B + ε)   # Green Leaf Index
ColorIndex = G / (R + B + ε)               # Pure greenness
```

### Model Architecture
```
Input (7 channels)
    ↓
1×1 Conv Adapter (7→3)
    ↓
SENet154 Encoder (ImageNet pre-trained)
    ↓
U-Net++ Decoder with SCSE Attention
    ↓
Sigmoid Activation
    ↓
Output: Probability [0-1]
```

### Loss Function
```
Total Loss = α·BCE + β·Dice + γ·Boundary + δ·Focal

where:
- BCE: Binary cross-entropy
- Dice: 1 - |2*TP| / (|Pred| + |True|)
- Boundary: Difference in Sobel-detected edges
- Focal: (1-p)^γ * CE (focus on hard pixels)

All weights are learned during training
```

---

## 📊 Expected Performance

### Accuracy Metrics
| Metric | Conservative | Balanced | Liberal |
|--------|--------------|----------|---------|
| Precision | 0.90+ | 0.75-0.85 | 0.60-0.70 |
| Recall | 0.60-0.70 | 0.80-0.90 | 0.90+ |
| F1 Score | 0.70-0.75 | 0.80-0.88 | 0.70-0.75 |
| IoU | 0.70-0.75 | 0.75-0.85 | 0.65-0.75 |

*Threshold variations (Conservative: 0.6-0.7, Balanced: 0.5, Liberal: 0.3-0.4)*

### Computational Performance
| Task | GPU (RTX 3090) | CPU (16 cores) | Memory |
|------|----------------|----------------|---------|
| Training (1000 tiles) | 2-5 hours | 24-48 hours | 6-8GB |
| Inference (1 m²) | 5-30 min | 2-8 hours | 3-5GB |
| Tile Creation | 1-5 min | 5-20 min | 2GB |

---

## 🎛️ Key Parameters

### Training
```bash
--batch_size 8         # Increase if GPU memory available (16, 32)
--epochs 150           # Total training epochs
--learning_rate 0.001  # Initial learning rate
--tile_size 512        # Input tile size (pixels)
```

### Inference
```bash
--tile_size 512        # Processing tile size
--overlap 64           # Tile overlap (pixels)
--threshold 0.5        # Binary classification threshold
  0.3: Include potential vegetation (liberal)
  0.5: Balanced (recommended)
  0.7: Only confident vegetation (conservative)
```

### Data Augmentation (from config)
- Flips: 50% horizontal, 50% vertical
- Rotations: 0-360° with 90° chance
- Color: ±25% brightness, ±35% contrast, ±15% hue/saturation
- Noise: Gaussian noise, blur
- Spatial: Elastic transforms, random crops

---

## 📚 Documentation

### Complete Guides
1. **[VEGETATION_DETECTION_GUIDE.md](./VEGETATION_DETECTION_GUIDE.md)** - Comprehensive documentation
   - Full workflow explanation
   - Troubleshooting guide
   - Advanced techniques
   - Performance optimization

2. **[IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)** - Technical details
   - What was changed and why
   - Architecture specifications
   - Loss function mathematics
   - Performance metrics

3. **[QUICK_REFERENCE.md](./QUICK_REFERENCE.md)** - Quick commands
   - Common usage patterns
   - Parameter reference
   - Troubleshooting quick fixes
   - GPU/CPU performance

### Configuration
- **[config/config_vegetation.yaml](./config/config_vegetation.yaml)** - Detailed config with comments

---

## 🔧 Customization

### Adjust for Different Resolution
```bash
# For 0.2m resolution, use 256×256 tiles (51.2m)
python create_vegetation_tiles.py --tile_size 256

# For 0.05m resolution, use 1024×1024 tiles (51.2m)
python create_vegetation_tiles.py --tile_size 1024
```

### Change Detection Threshold
```bash
# Conservative (fewer false positives)
python vegetation_inference.py --threshold 0.7

# Liberal (fewer false negatives)
python vegetation_inference.py --threshold 0.3
```

### Increase Training Epochs
Edit `config/config_vegetation.yaml`:
```yaml
training:
  epochs: 300  # Default is 150
```

### Enable Mixed Precision Training
Edit `config/config_vegetation.yaml`:
```yaml
advanced:
  use_amp: true  # Faster training, less memory
```

---

## 🐛 Troubleshooting

### "No tiles created"
**Cause**: Shapefile and TIFF have different CRS or no overlap
```bash
# Check CRS match
gdalinfo ortho.tif | grep -i "coordinate"
ogrinfo -al trees.shp | grep -i "coordinate"
```
**Solution**: Reproject shapefile to match TIFF CRS

### "Out of Memory"
**Cause**: Insufficient GPU/CPU memory
```bash
# Reduce tile size
python vegetation_inference.py --tile_size 256

# Reduce batch size
python vegetation_detection_training.py --batch_size 4

# Use CPU mode
python vegetation_inference.py --device cpu
```

### "Low accuracy"
**Causes**: Insufficient training data, poor digitization, seasonal mismatch
**Solutions**:
1. Increase training samples (target 1000+ tiles)
2. Improve digitization accuracy
3. Train on multi-seasonal data
4. Adjust threshold based on use case

### "Predictions too conservative"
**Solution**: Lower threshold
```bash
python vegetation_inference.py --threshold 0.3
```

---

## 🎓 Learning Resources

### Getting Started
1. Read [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) (5 min)
2. Run `vegetation_pipeline.py` on test data (1-2 hours)
3. Review predictions in GIS software

### Deep Dive
1. Read [VEGETATION_DETECTION_GUIDE.md](./VEGETATION_DETECTION_GUIDE.md) (30 min)
2. Study [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) (45 min)
3. Review source code with comments

### Advanced Topics
- Multi-model ensemble ([vegetation_ensemble.py](./vegetation_ensemble.py))
- Custom loss functions (in training script)
- Post-processing techniques
- Seasonal model variations

---

## 📝 Citation & References

If you use this model, please cite:

```bibtex
@software{vegetation_detection_2024,
  title={Vegetation Detection Model: RGB Satellite Imagery},
  author={Your Name},
  year={2024},
  url={path/to/repository}
}
```

### Technical References
- **U-Net++**: Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (2020)
- **SENet**: Hu et al., "Squeeze-and-Excitation Networks" (2018)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- **RGB Vegetation Indices**: Meyer & Neto, "Verification of color vegetation indices for automated crop imaging applications" (2008)

---

## 📞 Support & Feedback

For issues, improvements, or questions:

1. **Check Documentation**: Review QUICK_REFERENCE.md and VEGETATION_DETECTION_GUIDE.md
2. **Check Logs**: Review training logs in `model_dir/logs/`
3. **Verify Data**: Ensure TIFF and shapefile have matching CRS
4. **Test Subset**: Try on smaller area first

---

## 📈 Future Enhancements

Potential improvements to consider:
- [ ] NIR band support (if available) for better vegetation detection
- [ ] Multi-temporal analysis (seasonal variation)
- [ ] Hierarchical post-processing for tree delineation
- [ ] Real-time inference optimization
- [ ] Web interface for easy inference
- [ ] Integration with GIS software

---

## 📄 License

[Specify your license here]

---

## 🙏 Acknowledgments

Built with:
- PyTorch for deep learning
- Segmentation Models for pre-built architectures
- Rasterio for geospatial I/O
- Geopandas for vector data handling
- Albumentations for data augmentation

---

**Version**: 1.0  
**Last Updated**: January 2026  
**Status**: ✅ Production Ready

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [VEGETATION_DETECTION_GUIDE.md](./VEGETATION_DETECTION_GUIDE.md) | Comprehensive guide with detailed explanations |
| [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) | Quick command reference and common tasks |
| [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) | Technical implementation details |
| [config_vegetation.yaml](./config/config_vegetation.yaml) | Configuration with detailed comments |

Start with the Quick Start section above, then dive into the full guides as needed!
