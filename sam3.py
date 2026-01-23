import math
import torch
import rasterio
import numpy as np
import cv2
from rasterio.windows import Window
from ultralytics.models.sam import SAM3SemanticPredictor
import argparse

def run_predictions(input_path, output_path):

    inp_path = input_path #"/workspace/input/resamed_10cm/Mangalagiri_10.tif"
    out_mask_path = output_path #"/workspace/input/Predictions/Vegitation_predictions/sam3_predictions/Mangalagiri_1024_overlap_600.tif"
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
    final_mask = np.zeros((H, W), dtype=np.uint8)


    sample = ds.read(
        [1, 2, 3],
        out_shape=(3, ds.height // 10, ds.width // 10)
    )

    valid = sample[sample > 0]

    # Percentile-based global normalization values
    p2, p98 = np.percentile(valid, [2, 98])

    print(f"Global normalization values → p2={p2:.2f}, p98={p98:.2f}")


    xs = math.ceil((W - overlap) / (tile_size - overlap))
    ys = math.ceil((H - overlap) / (tile_size - overlap))

    @torch.no_grad()
    def run_tile(ix, iy):
        x = ix * (tile_size - overlap)
        y = iy * (tile_size - overlap)
        w = min(tile_size, W - x)
        h = min(tile_size, H - y)

        win = Window(x, y, w, h)
        img = ds.read([1, 2, 3], window=win)
        img = np.transpose(img, (1, 2, 0))

        # GLOBAL normalization
        img = np.clip(img, p2, p98)
        img = ((img - p2) / (p98 - p2) * 255).astype(np.uint8)

        img = np.ascontiguousarray(img)

        POSITIVE_TEXT = [
            "tree canopy",
            "tree crown",
            "tall trees",
            "woody vegetation",
            "trees with trunks and branches",
            "forest canopy",
            "arboreal vegetation"
        ]

        NEGATIVE_TEXT = [
            "grass",
            "lawn",
            "short grass",
            "crop field",
            "herbaceous plants",
            "ground vegetation",
            "pasture",
            "low vegetation"
        ]



        preds = predictor(
            img,
            text=POSITIVE_TEXT,
            negative_text=NEGATIVE_TEXT,
            conf=0.35
        )


        out = np.zeros((h, w), dtype=np.uint8)

        margin = overlap // 2
        valid = np.zeros_like(out, dtype=bool)
        valid[margin:h-margin, margin:w-margin] = True

        for p in preds:
            if p.masks is None:
                continue

            for m in p.masks.data.cpu().numpy().astype(bool):
                area = m.sum()
                if area < 400:
                    continue
                out[m & valid] = 1

        return x, y, out


    for iy in range(ys):
        for ix in range(xs):
            x, y, m = run_tile(ix, iy)
            h, w = m.shape
            final_mask[y:y+h, x:x+w] = np.maximum(final_mask[y:y+h, x:x+w], m)

    with rasterio.open(out_mask_path, "w", **meta) as dst:
        dst.write(final_mask, 1)

    ds.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble prediction for multi-class segmentation')

    parser.add_argument('--input_path', required=True,
                      help='Path to input image file')
    parser.add_argument('--output_path', required=True,
                      help='Path for output classification TIF')


    args = parser.parse_args()

    run_predictions(args.input_path, args.output_path)
