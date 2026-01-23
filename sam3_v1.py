import math
import torch
import rasterio
import numpy as np
import cv2
from rasterio.windows import Window
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from ultralytics.models.sam import SAM3SemanticPredictor
import argparse

def run_predictions(input_path, output_path):

    inp_path = input_path
    out_mask_path = output_path
    tile_size = 1024
    overlap = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = SAM3SemanticPredictor(
        overrides={
            "task": "segment",
            "mode": "predict",
            "model": "/workspace/input/ML_training/sam3/sam3_vegitation_finetuned.pt",
            "device": device
        }
    )
    predictor.setup_model()

    ds = rasterio.open(inp_path)
    meta = ds.meta.copy()
    meta.update(count=1, dtype="uint8", compress="lzw")

    H, W = ds.height, ds.width
    
    # ✅ FIX 1: Use float32 for soft blending
    final_mask = np.zeros((H, W), dtype=np.float32)
    count_mask = np.zeros((H, W), dtype=np.uint8)

    # Global normalization
    sample = ds.read(
        [1, 2, 3],
        out_shape=(3, ds.height // 10, ds.width // 10)
    )

    valid = sample[sample > 0]
    p2, p98 = np.percentile(valid, [2, 98])

    print(f"Global normalization values → p2={p2:.2f}, p98={p98:.2f}")

    xs = math.ceil((W - overlap) / (tile_size - overlap))
    ys = math.ceil((H - overlap) / (tile_size - overlap))

    print(f"Processing {xs * ys} tiles ({xs}x{ys})...")

    @torch.no_grad()
    def run_tile(ix, iy):
        x = ix * (tile_size - overlap)
        y = iy * (tile_size - overlap)
        w = min(tile_size, W - x)
        h = min(tile_size, H - y)

        win = Window(x, y, w, h)
        img = ds.read([1, 2, 3], window=win)
        img = np.transpose(img, (1, 2, 0))

        # Global normalization
        img = np.clip(img, p2, p98)
        img = ((img - p2) / (p98 - p2) * 255).astype(np.uint8)
        img = np.ascontiguousarray(img)

        # POSITIVE_TEXT = [
        #     "tree canopy",
        #     "tree crown",
        #     "tall trees",
        #     "woody vegetation",
        #     "trees with trunks and branches",
        #     "forest canopy",
        #     "arboreal vegetation"
        # ]

        # NEGATIVE_TEXT = [
        #     "grass",
        #     "lawn",
        #     "short grass",
        #     "crop field",
        #     "herbaceous plants",
        #     "ground vegetation",
        #     "pasture",
        #     "low vegetation"
        # ]

        POSITIVE_TEXT = [
            "isolated tree",
            "single tree",
            "scattered trees",
            "tree with visible crown",
            "tree surrounded by grass",
            "savanna tree",
            "pasture tree",
            "small tree crown",
            "individual tree",
            "tree canopy",
            "tree crown",
            "tall trees",
            "woody vegetation",
            "trees with trunks and branches",
            "forest canopy",
            "arboreal vegetation"
        ]

        NEGATIVE_TEXT = [
            "crop field",
            "agricultural crops",
            "row crops",
            "plantation rows",
            "bush clusters"
        ]



        preds = predictor(
            img,
            text=POSITIVE_TEXT,
            negative_text=NEGATIVE_TEXT,
            conf=0.2
        )

        out = np.zeros((h, w), dtype=np.float32)

        # ✅ FIX 2: REMOVED valid mask cropping - use ALL predictions
        for p in preds:
            if p.masks is None:
                continue

            for m in p.masks.data.cpu().numpy().astype(np.float32):
                # ✅ FIX 3: NO area filtering here - keep everything
                # out = np.maximum(out, m)
                out += m

        return x, y, out

    # Process all tiles
    for iy in range(ys):
        for ix in range(xs):
            x, y, m = run_tile(ix, iy)
            h, w = m.shape
            
            # ✅ FIX 4: Accumulate predictions with soft blending
            final_mask[y:y+h, x:x+w] += m
            # count_mask[y:y+h, x:x+w] += 1
            count_mask[y:y+h, x:x+w] += (m > 0).astype(np.uint8)

        
        print(f"Processed row {iy + 1}/{ys}")

    # ✅ FIX 5: Average overlapping predictions
    print("Averaging overlapping predictions...")
    final_mask = final_mask / np.maximum(count_mask, 1)
    
    # Threshold to binary
    final_mask_binary = (final_mask > 0.25).astype(np.uint8)

    # ✅ FIX 6: Apply area filtering AFTER full mosaic (optional post-processing)
    # This is now applied globally, not per-tile
    print("Applying global post-processing...")
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(final_mask_binary, connectivity=8)
    
    # Optional: Remove small components globally (uncomment if needed)
    # min_area = 400
    # for label in range(1, num_labels):
    #     mask = (labels == label)
    #     area = mask.sum()
    #     if area < min_area:
    #         final_mask_binary[mask] = 0
    
    print(f"Writing output to {out_mask_path}...")
    with rasterio.open(out_mask_path, "w", **meta) as dst:
        dst.write(final_mask_binary, 1)

    ds.close()
    print("✅ Prediction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM3 tiled prediction with proper overlap handling')

    parser.add_argument('--input_path', required=True,
                      help='Path to input image file')
    parser.add_argument('--output_path', required=True,
                      help='Path for output classification TIF')

    args = parser.parse_args()

    run_predictions(args.input_path, args.output_path)