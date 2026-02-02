# 🛣️ Multi-Class Mask Creation Guide

## Overview

Your modified script now supports **multi-class mask creation** for road classification with 4 distinct road types:

```
Mask Pixel Values:
├─ 0 = Background (non-road area)
├─ 1 = Thar Road
├─ 2 = CC Road
└─ 3 = Mud/Gravel Road
```

---

## How It Works

### Before: Binary Masks
```python
# Old approach - all roads were 1
shapes = [(geom, 1) for geom in intersecting.geometry]
# Result: 0 (no road) or 1 (any road)
```

### After: Multi-Class Masks
```python
# New approach - uses road_type attribute
shapes = [(geom, int(road_class)) for geom, road_class in 
         zip(intersecting.geometry, intersecting[class_column])]
# Result: 0 (background), 1 (Thar), 2 (CC), 3 (Mud/Gravel)
```

---

## Shapefile Requirements

Your shapefile must have an attribute column containing the road class:

```
Shapefile Structure:
├── geometry       (polygon boundaries)
└── road_type      (values: 1, 2, or 3)
    ├─ 1 = Thar Road
    ├─ 2 = CC Road
    └─ 3 = Mud/Gravel Road
```

### Example Data:

| ID | geometry | road_type |
|----|----------|-----------|
| 1 | POLYGON(...) | 1 |
| 2 | POLYGON(...) | 2 |
| 3 | POLYGON(...) | 1 |
| 4 | POLYGON(...) | 3 |

---

## Usage

### Basic Command (Using Default 'road_type' Column)

```bash
cd /home/srinivas/Pictures/github/ML_Setup

python create_tilesandmasks_fixed.py \
  --input_dir /path/to/data \
  --output_dir /path/to/output \
  --multiclass
```

### If Your Column Has Different Name

```bash
python create_tilesandmasks_fixed.py \
  --input_dir /path/to/data \
  --output_dir /path/to/output \
  --class_column my_road_class \
  --multiclass
```

### Create Binary Masks Instead (Optional)

```bash
python create_tilesandmasks_fixed.py \
  --input_dir /path/to/data \
  --output_dir /path/to/output \
  --binary
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_dir` | Required | Directory with TIF and SHP files |
| `--output_dir` | Required | Output directory for tiles/masks |
| `--class_column` | `road_type` | Shapefile column name with class values |
| `--multiclass` | True | Create multiclass masks (1-3) |
| `--binary` | False | Create binary masks (0-1) only |

---

## Output Structure

```
output_dir/
├── tiles/
│   ├── image_0_0.tif              (1024×1024×3 RGB)
│   ├── image_0_1024.tif           (with overlap)
│   └── ...
└── masks/
    ├── image_0_0.tif              (1024×1024×1 multiclass)
    │                               Values: 0, 1, 2, or 3
    ├── image_0_1024.tif
    └── ...
```

### Reading Mask Values

```python
import rasterio
import numpy as np

with rasterio.open('masks/image_0_0.tif') as src:
    mask = src.read(1)  # shape: (1024, 1024)
    
# Get class distribution
unique_classes, counts = np.unique(mask, return_counts=True)
print(unique_classes)  # [0, 1, 2, 3]
print(counts)          # [500000, 150000, 200000, 174000]
```

---

## What Happens During Tile Creation

### Step 1: Load and Validate
```
Input:  satellite_image.tif + roads.shp (with road_type attribute)
            ↓
Read shapefile → Check 'road_type' column exists
                ↓
                Print: "Found classes: [1, 2, 3]"
```

### Step 2: Create Each Tile
```
For each tile position (i, j):
    ↓
    Read RGB data from satellite image
    ↓
    Find all road polygons in this tile
    ↓
    Rasterize mask:
    ├─ For each road polygon:
    │  └─ Get its road_type value (1, 2, or 3)
    │  └─ Set all pixels in polygon to that value
    ├─ Fill remaining pixels with 0 (background)
    ↓
    Save tile.tif + mask.tif (georeferenced)
```

### Step 3: Output Information
```
Example output line:
"Saved tile + mask → 0_0 | Class 0:500000, Class 1:150000, Class 2:200000, Class 3:74000"
                                ↑
                        Shows pixel counts per class
```

---

## Example Walkthrough

### Scenario: You have one satellite image with 100 roads

```
roads.shp:
├─ 40 roads with road_type=1 (Thar)
├─ 35 roads with road_type=2 (CC)
└─ 25 roads with road_type=3 (Mud/Gravel)
```

### Command:
```bash
python create_tilesandmasks_fixed.py \
  --input_dir ./data \
  --output_dir ./tiles_output \
  --class_column road_type \
  --multiclass
```

### Process:
```
1. ✅ Load shapefile → 100 geometries found
2. ✅ Validate 'road_type' column → Found classes: [1, 2, 3]
3. ✅ Create tile 0: 
   - Found 5 roads in this tile: 3×Class1, 1×Class2, 1×Class3
   - Mask pixels:
     * 0 (background): ~990,000 pixels
     * 1 (Thar):      ~5,000 pixels
     * 2 (CC):        ~3,000 pixels
     * 3 (Mud/Gravel):~2,000 pixels
4. ✅ Create tile 1: ...
   ...
N. ✅ Tiling complete! Created 64 tiles
```

---

## Verification: How to Check Your Masks

### Check 1: Mask Values Range

```python
import rasterio
import numpy as np
from pathlib import Path

# Check all masks
masks_dir = Path('output_dir/masks')
for mask_file in sorted(masks_dir.glob('*.tif')):
    with rasterio.open(mask_file) as src:
        mask = src.read(1)
        unique = np.unique(mask)
        print(f"{mask_file.name}: {unique}")
        # Should print: [0 1 2 3] (if all classes present)
        # Or subset like: [0 1 3] (if class 2 not in this tile)
```

### Check 2: Class Distribution

```python
import rasterio
import numpy as np
from pathlib import Path
from collections import defaultdict

total_counts = defaultdict(int)

masks_dir = Path('output_dir/masks')
for mask_file in sorted(masks_dir.glob('*.tif')):
    with rasterio.open(mask_file) as src:
        mask = src.read(1)
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            total_counts[int(u)] += c

print("📊 Total pixel distribution across all masks:")
for class_id in sorted(total_counts.keys()):
    pixels = total_counts[class_id]
    percentage = (pixels / sum(total_counts.values())) * 100
    
    class_names = {
        0: "Background",
        1: "Thar Road",
        2: "CC Road",
        3: "Mud/Gravel Road"
    }
    
    print(f"  Class {class_id} ({class_names[class_id]}): {pixels:,} pixels ({percentage:.1f}%)")
```

**Expected output:**
```
📊 Total pixel distribution across all masks:
  Class 0 (Background):       150,000,000 pixels (88.2%)
  Class 1 (Thar Road):        12,000,000 pixels (7.0%)
  Class 2 (CC Road):          7,500,000 pixels (4.4%)
  Class 3 (Mud/Gravel Road):  500,000 pixels (0.4%)
```

### Check 3: Tile-Mask Alignment

```python
import rasterio
from pathlib import Path

tiles_dir = Path('output_dir/tiles')
masks_dir = Path('output_dir/masks')

tiles = sorted(tiles_dir.glob('*.tif'))
masks = sorted(masks_dir.glob('*.tif'))

assert len(tiles) == len(masks), f"Mismatch! {len(tiles)} tiles vs {len(masks)} masks"

for tile_file, mask_file in zip(tiles, masks):
    with rasterio.open(tile_file) as src_tile:
        with rasterio.open(mask_file) as src_mask:
            assert src_tile.height == src_mask.height, f"Height mismatch in {tile_file}"
            assert src_tile.width == src_mask.width, f"Width mismatch in {tile_file}"
            assert src_tile.bounds == src_mask.bounds, f"Bounds mismatch in {tile_file}"

print("✅ All tiles and masks are perfectly aligned!")
```

---

## Training Your Model

Once you have multiclass masks, use them with the training script:

```bash
cd pytorch_model_training

python enhanced_pytorch_backbone_training_multiclass.py \
  --input_tiles_dir ../tiles_output/tiles \
  --input_masks_dir ../tiles_output/masks \
  --model_path ./models/roads_multiclass_v1
```

The training will automatically:
- Detect 4 classes (0-3)
- Calculate class weights
- Train with multiclass loss functions
- Track per-class IoU metrics

---

## Troubleshooting

### ❌ Error: "Column 'road_type' not found"

**Solution:**
```bash
# Check available columns
python -c "
import geopandas as gpd
gdf = gpd.read_file('your_shapefile.shp')
print('Columns:', list(gdf.columns))
"

# Use correct column name
python create_tilesandmasks_fixed.py \
  --input_dir ./data \
  --output_dir ./output \
  --class_column actual_column_name
```

### ❌ Error: "Unexpected class values"

**Example message:**
```
⚠️ WARNING: Found unexpected class values: {4, 5}
   Expected values: 1 (Thar), 2 (CC), 3 (Mud/Gravel)
```

**Solution:** Fix shapefile values or map them:

```python
import geopandas as gpd

gdf = gpd.read_file('roads.shp')
# Map 4→3, 5→1, etc.
gdf['road_type'] = gdf['road_type'].replace({4: 3, 5: 1})
gdf.to_file('roads_fixed.shp')
```

### ❌ Masks are all zeros

**Cause:** Road polygons not intersecting with tiles (CRS mismatch)

**Solution:**
```bash
# Check CRS alignment
python -c "
import rasterio
import geopandas as gpd

with rasterio.open('satellite.tif') as src:
    print('Raster CRS:', src.crs)

gdf = gpd.read_file('roads.shp')
print('Shapefile CRS:', gdf.crs)
"
```

Ensure both have same CRS before running tile creation.

---

## Summary

✅ **Your modified script now:**
- Reads `road_type` attribute from shapefile
- Creates masks with class values 0, 1, 2, 3
- Validates class values during execution
- Shows pixel counts per class for each tile
- Supports different column names
- Has optional binary mask fallback

✅ **To use it:**
```bash
python create_tilesandmasks_fixed.py \
  --input_dir /path/to/data \
  --output_dir /path/to/output \
  --class_column road_type \
  --multiclass
```

✅ **Your shapefile needs:**
- `geometry` column with road polygons
- `road_type` column with values: 1, 2, or 3

---

**Status**: ✅ Ready to create multiclass training data!
