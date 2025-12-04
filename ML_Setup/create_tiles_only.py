from rasterio.windows import Window
import os
import rasterio
import numpy as np
from pathlib import Path
import yaml


def create_tiles(input_tif, output_dir, tile_size=256):
    """
    Create tiles from TIF file (no masks).
    """
    # Create output directory
    basename = os.path.splitext(os.path.basename(input_tif))[0]
    tiles_dir = os.path.join(output_dir, f"tiles_{tile_size}_{basename}_0p3")

    os.makedirs(tiles_dir, exist_ok=True)

    # Read the TIF file with full metadata
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()

        height = src.height
        width = src.width
        num_bands = src.count

        # Calculate tiles
        n_tiles_height = int(np.ceil(height / tile_size))
        n_tiles_width = int(np.ceil(width / tile_size))

        # Read entire raster (if possible)
        try:
            full_img = src.read()
            use_full_img = True
            print(f"Read full image into memory: {full_img.shape}")
        except MemoryError:
            use_full_img = False
            print("Image too large, reading tile by tile")

        # Process each tile
        for i in range(n_tiles_height):
            for j in range(n_tiles_width):
                col_off = j * tile_size
                row_off = i * tile_size
                width_window = min(tile_size, width - col_off)
                height_window = min(tile_size, height - row_off)

                window = Window(col_off, row_off, width_window, height_window)

                if use_full_img:
                    raw_tile = full_img[:, row_off:row_off + height_window,
                                           col_off:col_off + width_window]
                else:
                    raw_tile = src.read(window=window)

                # Pad if smaller than tile_size
                tile = np.zeros((num_bands, tile_size, tile_size), dtype=src.dtypes[0])
                tile[:, :height_window, :width_window] = raw_tile

                # Skip empty tiles (almost all zeros)
                if np.mean(tile) < 0.01:
                    continue

                # Update metadata
                window_transform = rasterio.windows.transform(window, src.transform)
                tile_profile = profile.copy()
                tile_profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': window_transform,
                })

                # Save tile
                tile_path = os.path.join(tiles_dir, f"{basename}_{i * tile_size}_{j * tile_size}.tif")
                with rasterio.open(tile_path, 'w', **tile_profile) as dst:
                    dst.write(tile)


def process_all_files(input_dir, output_dir):
    """
    Process all TIF files in input directory (no SHP needed).
    """
    tif_files = list(Path(input_dir).glob("*.tif"))
    print("Checking files...")

    with open('config/config_v1.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for tif_file in tif_files:
        basename = tif_file.stem
        print(f"Processing {tif_file.name}")
        try:
            create_tiles(str(tif_file), output_dir, tile_size=config['data']['input_size'])
            print("Created all tiles successfully")
        except Exception as e:
            print(f"Error processing {tif_file.name}: {str(e)}")


if __name__ == "__main__":
    input_folder = input("Enter input folder path: ").strip()
    output_folder = input("Enter output folder path: ").strip()

    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    print(f"\nInput folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print("Processing..")
    process_all_files(input_folder, output_folder)

