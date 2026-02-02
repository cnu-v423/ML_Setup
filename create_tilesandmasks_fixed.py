from rasterio.features import rasterize 
import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
from rasterio.windows import Window
from pathlib import Path
from shapely.geometry import box
import yaml
import argparse

def scale_tile(tile):
    """
    Applies percentile normalization + scaling to uint8
    tile shape: (bands, H, W)
    returns: scaled uint8 tile (same shape)
    """
    print("Scaling tile")
    scaled_bands = []

    for b in range(tile.shape[0]):
        arr = tile[b].copy()

        if np.any(arr > 0):
            # Compute 2.5% and 99% percentiles only from positive values
            thr1 = round(np.percentile(arr[arr > 0], 2.5))
            thr2 = round(np.percentile(arr[arr > 0], 99))

            # Clip
            arr[arr < thr1] = thr1
            arr[arr > thr2] = thr2

            # Normalize
            if (thr2 - thr1) > 0:
                arr = (arr - thr1) / (thr2 - thr1)
            else:
                arr = np.zeros_like(arr)

            # Convert to uint8
            arr = (arr * 255).astype(np.uint8)

            # Avoid pure zeros
            arr[arr == 0] = 1
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

        scaled_bands.append(arr)

    return np.stack(scaled_bands)

def create_tiles(input_tif, input_shp, output_dir, tile_size=1024, overlap_percentage=256, class_column='road_type', is_multiclass=True):
    """
    Create tiles from TIF file and corresponding masks from SHP file.
    This version never loads the entire raster into RAM.
    It reads only each tile window directly from disk.
    
    Args:
        input_tif: Path to input satellite image
        input_shp: Path to input shapefile
        output_dir: Output directory
        tile_size: Size of tiles (default 1024x1024)
        overlap_percentage: Overlap percentage (default 256 pixels on 1024)
        class_column: Column name in shapefile containing class values (default 'road_type')
        is_multiclass: If True, use class values from shapefile; if False, create binary masks
    
    Mask classes:
        0 = Background (no road)
        1 = Thar Road
        2 = CC Road
        3 = Mud/Gravel Road
    """

    basename = os.path.splitext(os.path.basename(input_tif))[0]

    # tiles_dir = os.path.join(output_dir, f"tiles_{tile_size}_{basename}_new")
    # masks_dir = os.path.join(output_dir, f"masks_{tile_size}_{basename}_new")
    tiles_dir = os.path.join(output_dir, "tiles")
    masks_dir = os.path.join(output_dir, "masks")

    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # --------------------------
    # Read raster + metadata
    # --------------------------
    with rasterio.open(input_tif) as src:

        overlap_pixels = int(tile_size * overlap_percentage)
        stride = tile_size - overlap_pixels

        if stride <= 0:
            raise ValueError("overlap_percent must be < 100")

        profile = src.profile.copy()
        width = src.width
        height = src.height
        transform = src.transform
        selected_bands = [1, 2, 3]       # Only 3 bands for output
        num_bands = len(selected_bands)

        # --------------------------
        # Load and fix shapefile CRS
        # --------------------------
        gdf = gpd.read_file(input_shp)

        if gdf.crs is None:
            print(f"Setting shapefile CRS to: {src.crs}")
            gdf.set_crs(src.crs, inplace=True)
        elif gdf.crs != src.crs:
            print(f"Reprojecting shapefile from {gdf.crs} → {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # --------------------------
        # Validate multiclass column if needed
        # --------------------------
        if is_multiclass:
            if class_column not in gdf.columns:
                raise ValueError(f"Column '{class_column}' not found in shapefile. Available columns: {list(gdf.columns)}")
            
            # Check class values
            unique_classes = sorted(gdf[class_column].unique())
            print(f"📊 Found classes in '{class_column}': {unique_classes}")
            
            # Validate class values are in expected range (0-3 or 1-3)
            valid_classes = set(unique_classes)
            expected_classes = {1, 2, 3}  # Thar, CC, Mud/Gravel
            
            if not valid_classes.issubset(expected_classes):
                print(f"⚠️ WARNING: Found unexpected class values: {valid_classes - expected_classes}")
                print(f"   Expected values: 1 (Thar), 2 (CC), 3 (Mud/Gravel)")
            
            print(f"✅ MultiClass mode: Using '{class_column}' attribute for mask values")

        # Filter shapes outside raster extent
        raster_bounds = box(*src.bounds)
        gdf = gdf[gdf.geometry.intersects(raster_bounds)]

        if len(gdf) == 0:
            print(f"⚠️ Warning: No geometries intersect with {basename}")
            return

        # --------------------------
        # Tile count
        # --------------------------
        # n_tiles_h = int(np.ceil(height / tile_size))
        # n_tiles_w = int(np.ceil(width / tile_size))

        ## With Overlap
        n_tiles_h = int(np.ceil((height - tile_size) / stride)) + 1
        n_tiles_w = int(np.ceil((width - tile_size) / stride)) + 1


        print(f"Raster size: {width}×{height}, generating {n_tiles_h}×{n_tiles_w} tiles")

        # --------------------------
        # Process tiles one by one
        # --------------------------
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):

                # col_off = j * tile_size
                # row_off = i * tile_size
                ## With Overlap
                col_off = j * stride
                row_off = i * stride

                win_w = min(tile_size, width - col_off)
                win_h = min(tile_size, height - row_off)

                window = Window(col_off, row_off, win_w, win_h)
                window_transform = rasterio.windows.transform(window, transform)

                # Read only this tile from disk
                tile_data = src.read(selected_bands, window=window)

                # Pad to full tile size (only for edges)
                tile = np.zeros((num_bands, tile_size, tile_size), dtype=src.dtypes[0])
                tile[:, :win_h, :win_w] = tile_data

                # Skip empty tiles
                if np.mean(tile) < 0.01:
                    continue

                # Tile bounds
                tile_bounds = rasterio.windows.bounds(window, transform)
                tile_box = box(*tile_bounds)

                # Intersecting geoms
                intersecting = gdf[gdf.geometry.intersects(tile_box)]

                # Filter only building class (building_i == 1)  
                # intersecting = intersecting[intersecting["building_i"] == 1]

                if len(intersecting) == 0:
                    continue

                # Rasterize mask with class values
                if is_multiclass:
                    # Use class values from the specified column
                    shapes = [(geom, int(road_class)) for geom, road_class in 
                             zip(intersecting.geometry, intersecting[class_column])]
                else:
                    # Binary mask: all roads get value 1
                    shapes = [(geom, 1) for geom in intersecting.geometry]

                mask_arr = rasterize(
                    shapes,
                    out_shape=(tile_size, tile_size),
                    transform=window_transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
                )

                if np.sum(mask_arr) == 0:
                    continue

                # --------------------------
                # Save tile
                # --------------------------

                scaled_tile = scale_tile(tile)

                tile_profile = profile.copy()
                tile_profile.update({
                    'count': num_bands,
                    'height': tile_size,
                    'width': tile_size,
                    'transform': window_transform,
                })

                # tile_path = os.path.join(tiles_dir, f"{basename}_{i*tile_size}_{j*tile_size}.tif")
                tile_path = os.path.join(tiles_dir, f"{basename}_{row_off}_{col_off}.tif")


                with rasterio.open(tile_path, 'w', **tile_profile) as dst:
                    dst.write(scaled_tile)

                # --------------------------
                # Save mask
                # --------------------------
                mask_profile = profile.copy()
                mask_profile.update({
                    'count': 1,
                    'dtype': np.uint8,
                    'height': tile_size,
                    'width': tile_size,
                    'transform': window_transform
                })
                mask_profile.pop("nodata", None)

                # mask_path = os.path.join(masks_dir, f"{basename}_{i*tile_size}_{j*tile_size}.tif")
                #with overlap
                mask_path = os.path.join(masks_dir, f"{basename}_{row_off}_{col_off}.tif")
            

                with rasterio.open(mask_path, "w", **mask_profile) as dst_mask:
                    dst_mask.write(mask_arr[np.newaxis, :, :])

                # Print class distribution for this tile
                unique_vals, counts = np.unique(mask_arr, return_counts=True)
                class_info = ", ".join([f"Class {int(v)}:{c}" for v, c in zip(unique_vals, counts)])
                print(f"Saved tile + mask → {i}_{j} | {class_info}")

    print("✅ Tiling complete!")




def create_full_mask(input_tif, input_shp, output_mask_path):
    """
    Create a full-size binary mask from a shapefile that matches the input TIF.
    """
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
 
        # Read shapefile
        gdf = gpd.read_file(input_shp)
 
        # Align CRS
        if gdf.crs is None:
            print("Setting CRS of shapefile to match raster.")
            gdf.set_crs(src.crs, inplace=True)
        elif gdf.crs != src.crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
 
        # Filter geometries within raster bounds
        raster_bounds = box(*src.bounds)
        gdf = gdf[gdf.geometry.intersects(raster_bounds)]
 
        if len(gdf) == 0:
            print("No geometries intersect with raster.")
            return
 
        # Create binary mask from geometries
        shapes = [(geom, 1) for geom in gdf.geometry]
        mask_arr = rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
 
        # Write the full mask to a GeoTIFF
        mask_profile = profile.copy()
        mask_profile.update({
            'count': 1,
            'dtype': np.uint8,
            'nodata': 0
        })

        mask_profile.pop('nodata', None)
    
        with rasterio.open(output_mask_path, 'w', **mask_profile) as dst:
            dst.write(mask_arr[np.newaxis, :, :])
 
        print(f"Full mask saved to: {output_mask_path}")

def process_all_files(input_dir, output_dir, class_column='road_type', is_multiclass=True):
    """
    Process all TIF and SHP files in input directory
    
    Args:
        input_dir: Directory with input TIF and SHP files
        output_dir: Output directory for tiles and masks
        class_column: Column name in shapefile for class values
        is_multiclass: Whether to create multiclass or binary masks
    """
    # Get all TIF files
    tif_files = list(Path(input_dir).glob("*.tif"))
    print("Checking pairs...")

    with open('config/config_v1.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    for tif_file in tif_files:
        # Find corresponding SHP file
        shp_file = tif_file.with_suffix(".shp")
        basename = tif_file.stem
        print(f"shape file :: {shp_file}")
        if shp_file.exists():
            print(f"Processing {tif_file.name} and {shp_file.name}")
            try:
                # full_mask_path = os.path.join(output_dir, f"{basename}_full_mask.tif")
                # create_full_mask(str(tif_file), str(shp_file), full_mask_path)

                create_tiles(
                    str(tif_file), 
                    str(shp_file), 
                    output_dir, 
                    tile_size=config['data']['input_size'], 
                    overlap_percentage=config['data']['overlap_percentage'],
                    class_column=class_column,
                    is_multiclass=is_multiclass
                )
                print("✅ Created all the tiles and respective masks")
            except Exception as e:
                print(f"❌ Error processing {tif_file.name}: {str(e)}")
        else:
            print(f"❌ No corresponding SHP file found for {tif_file.name}")

if __name__ == "__main__":
    # Example usage
    # input_directory = "/home/vinay/Downloads/buildingwithUC_Rectified_data"
    # output_directory = "/home/vinay/Downloads/buildingwithUC_Rectified_train_1024"

    parser = argparse.ArgumentParser(description='Create tiles and masks for road classification')
    parser.add_argument('--input_dir', required=True, help='Directory containing input TIF and SHP files')
    parser.add_argument('--output_dir', required=True, help='Output directory for tiles and masks')
    parser.add_argument('--class_column', default='road_type', help='Column name in shapefile containing class values (default: road_type)')
    parser.add_argument('--multiclass', action='store_true', default=True, help='Create multiclass masks (default: True)')
    parser.add_argument('--binary', action='store_true', help='Create binary masks instead of multiclass')
    args = parser.parse_args()
    
    # Determine if multiclass or binary
    is_multiclass = not args.binary

    input_folder = os.path.abspath(args.input_dir)
    output_folder = os.path.abspath(args.output_dir)

    # input_folder = os.path.abspath(input_folder)
    # output_folder = os.path.abspath(output_folder)
    
    # Confirm with user
    print(f"\n{'='*60}")
    print(f"🗺️  ROAD CLASSIFICATION TILE & MASK CREATION")
    print(f"{'='*60}")
    print(f"Input folder:    {input_folder}")
    print(f"Output folder:   {output_folder}")
    print(f"Class column:    {args.class_column}")
    print(f"Mode:            {'MULTICLASS' if is_multiclass else 'BINARY'}")
    print(f"{'='*60}")
    print("Processing..\n")
    
    process_all_files(input_folder, output_folder, class_column=args.class_column, is_multiclass=is_multiclass)
