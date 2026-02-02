# 🚀 Quick Start: Multi-Class Mask Creation

## TL;DR - Just Run This

### Step 1: Prepare Your Data
```
your_data_folder/
├── satellite_image.tif    (your large satellite image)
├── roads.shp              (shapefile with road_type attribute)
├── roads.shx
├── roads.dbf
└── roads.prj
```

**Your shapefile must have:**
- `geometry` column (polygon boundaries)
- `road_type` column (values: 1, 2, or 3)

### Step 2: Run the Script
```bash
cd /home/srinivas/Pictures/github/ML_Setup

python create_tilesandmasks_fixed.py \
  --input_dir /path/to/your_data_folder \
  --output_dir /path/to/output_folder \
  --class_column road_type \
  --multiclass
```

### Step 3: Output
```
output_folder/
├── tiles/
│   ├── satellite_image_0_0.tif
│   ├── satellite_image_0_1024.tif
│   └── ...
└── masks/
    ├── satellite_image_0_0.tif    (Values: 0, 1, 2, or 3)
    ├── satellite_image_0_1024.tif
    └── ...
```

---

## What Each Mask Value Means

```
0 = Background (non-road)
1 = Thar Road
2 = CC Road
3 = Mud/Gravel Road
```

---

## Example Command

```bash
python create_tilesandmasks_fixed.py \
  --input_dir ./data/satellite_data \
  --output_dir ./data/training_tiles \
  --class_column road_type \
  --multiclass
```

---

## If Your Column Has Different Name

```bash
python create_tilesandmasks_fixed.py \
  --input_dir ./data \
  --output_dir ./output \
  --class_column road_classification \  # Instead of road_type
  --multiclass
```

---

## To Verify Your Masks Are Correct

```python
import rasterio
import numpy as np

# Open a mask file
with rasterio.open('output_dir/masks/satellite_image_0_0.tif') as src:
    mask = src.read(1)
    
# Check values (should be 0, 1, 2, 3)
print("Unique values:", np.unique(mask))
# Output: [0 1 2 3] or subset

# Check distribution
unique, counts = np.unique(mask, return_counts=True)
for val, count in zip(unique, counts):
    print(f"Class {val}: {count} pixels")
```

---

## Next Step: Train Your Model

```bash
cd pytorch_model_training

python enhanced_pytorch_backbone_training_multiclass.py \
  --input_tiles_dir ../data/training_tiles/tiles \
  --input_masks_dir ../data/training_tiles/masks \
  --model_path ./models/roads_multiclass_v1
```

---

## Still Have Questions?

📖 Read the full guide: [MULTICLASS_MASK_CREATION_GUIDE.md](MULTICLASS_MASK_CREATION_GUIDE.md)

---

**Status**: ✅ Ready to go!
