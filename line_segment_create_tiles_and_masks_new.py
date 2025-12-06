from rasterio.features import rasterize 
import os
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
from rasterio.windows import Window
from pathlib import Path
from shapely.geometry import box
import yaml
import rasterio.warp

def calculate_zoom_factor(src_resolution, target_resolution):
    """
    Calculate zoom factor based on source and target resolutions
    """
    if isinstance(src_resolution, tuple):
        src_res = min(src_resolution)  # Use the finer resolution
    else:
        src_res = abs(src_resolution)
    
    target_res = abs(target_resolution)
    zoom_factor = src_res / target_res
    
    print(f"Source resolution: {src_res:.4f}m, Target resolution: {target_res:.4f}m")
    print(f"Zoom factor: {zoom_factor:.4f}")
    
    return zoom_factor

def process_tile_with_zoom(src, window, zoom_factor, target_resolution, tile_size, config):
    """
    Process a single tile with zoom/resampling, always using first 3 bands
    """
    # Read only the first three bands of the original tile
    original_tile = src.read([1, 2, 3], window=window)
    original_height, original_width = window.height, window.width
    
    # Calculate new dimensions based on zoom factor
    new_height = int(original_height * zoom_factor)
    new_width = int(original_width * zoom_factor)
    
    # Get the window transform
    window_transform = rasterio.windows.transform(window, src.transform)
    
    # Create new transform for the resampled tile
    # Adjust pixel size according to zoom factor
    new_transform = rasterio.Affine(
        window_transform.a / zoom_factor,  # new pixel width
        window_transform.b,
        window_transform.c,  # same x origin
        window_transform.d,
        window_transform.e / zoom_factor,  # new pixel height  
        window_transform.f   # same y origin
    )
    
    # Create output array for exactly 3 bands
    resampled_tile = np.zeros((config['data']['tiff_bands'], new_height, new_width), dtype=src.dtypes[0])
    
    # Resample each of the three bands
    for band_idx in range(config['data']['tiff_bands']):
        rasterio.warp.reproject(
            source=original_tile[band_idx],
            destination=resampled_tile[band_idx],
            src_transform=window_transform,
            src_crs=src.crs,
            dst_transform=new_transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear  # You can change this to nearest, cubic, etc.
        )
    
    return resampled_tile, new_transform, new_height, new_width

def create_linestring_mask_for_resampled_tile(gdf, tile_bounds, new_height, new_width, new_transform, line_width_meters, target_resolution):
    """
    Create mask for resampled tile specifically for line strings with proper line width
    """
    tile_box = box(*tile_bounds)
    intersecting_geoms = gdf[gdf.geometry.intersects(tile_box)]
    
    if len(intersecting_geoms) == 0:
        return None
    
    # Convert line width from meters to pixels
    line_width_pixels = line_width_meters / target_resolution
    print(f"Line width: {line_width_meters}m = {line_width_pixels:.2f} pixels")
    
    # Buffer the line geometries to create width
    buffered_geoms = []
    for geom in intersecting_geoms.geometry:
        if geom.geom_type in ['LineString', 'MultiLineString']:
            # Buffer the line geometry by half the line width (radius)
            buffered_geom = geom.buffer(line_width_meters / 2)
            buffered_geoms.append(buffered_geom)
        else:
            # For other geometry types, use as is
            buffered_geoms.append(geom)
    
    if not buffered_geoms:
        return None
    
    # Create mask for the resampled dimensions
    shapes = [(geom, 1) for geom in buffered_geoms]
    mask_arr = rasterize(
        shapes,
        out_shape=(new_height, new_width),
        transform=new_transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    
    return mask_arr

def pad_or_crop_to_tile_size(data, tile_size):
    """
    Pad or crop data to exact tile size
    """
    if len(data.shape) == 3:  # Multi-band
        bands, height, width = data.shape
        result = np.zeros((bands, tile_size, tile_size), dtype=data.dtype)
        
        # Calculate how much to copy
        copy_height = min(height, tile_size)
        copy_width = min(width, tile_size)
        
        result[:, :copy_height, :copy_width] = data[:, :copy_height, :copy_width]
        
    else:  # Single band (mask)
        height, width = data.shape
        result = np.zeros((tile_size, tile_size), dtype=data.dtype)
        
        # Calculate how much to copy
        copy_height = min(height, tile_size)
        copy_width = min(width, tile_size)
        
        result[:copy_height, :copy_width] = data[:copy_height, :copy_width]
    
    return result

def calculate_tile_grid_with_overlap(width, height, tile_size, overlap_percentage):
    """
    Calculate tile positions with overlap
    Returns list of (row_start, col_start, actual_height, actual_width) for each tile
    """
    overlap_pixels = int(tile_size * overlap_percentage)
    step_size = tile_size - overlap_pixels
    
    print(f"Tile size: {tile_size}, Overlap: {overlap_pixels} pixels ({overlap_percentage*100}%)")
    print(f"Step size: {step_size}")
    
    tiles = []
    
    # Calculate number of tiles needed
    n_tiles_height = int(np.ceil((height - overlap_pixels) / step_size))
    n_tiles_width = int(np.ceil((width - overlap_pixels) / step_size))
    
    print(f"Grid size: {n_tiles_height} x {n_tiles_width} tiles")
    
    for i in range(n_tiles_height):
        for j in range(n_tiles_width):
            # Calculate tile position
            row_start = i * step_size
            col_start = j * step_size
            
            # Ensure we don't exceed image boundaries
            row_end = min(row_start + tile_size, height)
            col_end = min(col_start + tile_size, width)
            
            actual_height = row_end - row_start
            actual_width = col_end - col_start
            
            # Only add tiles that have reasonable size
            if actual_height >= tile_size // 2 and actual_width >= tile_size // 2:
                tiles.append((row_start, col_start, actual_height, actual_width, i, j))
    
    return tiles

def create_tiles_with_zoom(input_tif, input_shp, output_dir, config):
    """
    Create tiles with zoom level processing from TIF file and corresponding binary masks from SHP file
    Specifically designed for line string geometries with configurable line width
    """
    # Get configuration values
    tile_size = config['data']['input_size']
    target_resolution = config['data']['resolution']
    overlap_percentage = config['data']['overlap_percentage']
    line_width = config['data']['line_width']  # Line width in meters

    basename = os.path.splitext(os.path.basename(input_tif))[0]
    res_str = str(target_resolution).replace('.', 'p')
    overlap_str = str(int(overlap_percentage * 100))
    line_width_str = str(line_width)
    
    # Create output directories with line width info
    tiles_dir = os.path.join(output_dir, f"tiles_{tile_size}_{res_str}m_{overlap_str}_lw{line_width_str}_{basename}")
    masks_dir = os.path.join(output_dir, f"masks_{tile_size}_{res_str}m_{overlap_str}_lw{line_width_str}_{basename}")
    
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Read the TIF file with full metadata
    with rasterio.open(input_tif) as src:
        # Store the source metadata for later use
        profile = src.profile.copy()
        
        # Calculate zoom factor
        src_resolution = src.res  # (x_res, y_res)
        zoom_factor = calculate_zoom_factor(src_resolution, target_resolution)
        
        # Read the shapefile
        gdf = gpd.read_file(input_shp)
        print(f"Loaded {len(gdf)} geometries from shapefile")
        print(f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
        
        # CRS handling
        if gdf.crs is None:
            print(f"Setting shapefile CRS to match raster: {src.crs}")
            gdf.set_crs(src.crs, inplace=True)
        elif gdf.crs != src.crs:
            print(f"Reprojecting shapefile from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # Filter geometries
        raster_bounds = box(*src.bounds)
        gdf = gdf[gdf.geometry.intersects(raster_bounds)]
        
        if len(gdf) == 0:
            print(f"Warning: No geometries intersect with {basename}")
            return
        
        print(f"Found {len(gdf)} geometries intersecting with raster bounds")
        
        # Get dimensions
        height = src.height
        width = src.width
        
        # Calculate processing tile size (larger than output tile size to handle resampling)
        process_tile_size = int(tile_size / zoom_factor) + 100  # Add buffer for edge effects
        
        tile_positions = calculate_tile_grid_with_overlap(
            width, height, process_tile_size, overlap_percentage
        )

        print(f"Processing {len(tile_positions)} overlapping tiles")
        print(f"Process tile size: {process_tile_size}, Output tile size: {tile_size}")
        print(f"Overlap: {overlap_percentage*100}%")
        print(f"Line width: {line_width} meters")

        tile_count = 0
        tiles_with_lines = 0
        
        # Process each tile
        for row_start, col_start, actual_height, actual_width, i, j in tile_positions:
            # Define the processing window
            window = Window(col_start, row_start, actual_width, actual_height)
            
            try:
                # Process tile with zoom
                resampled_tile, new_transform, new_height, new_width = process_tile_with_zoom(
                    src, window, zoom_factor, target_resolution, tile_size, config
                )
                
                # Skip empty tiles (very dark images)
                if np.mean(resampled_tile) < 0.01:
                    continue
                
                # Calculate tile bounds for the resampled tile
                tile_bounds = rasterio.transform.array_bounds(new_height, new_width, new_transform)
                
                # Create mask for resampled tile with line width
                mask_arr = create_linestring_mask_for_resampled_tile(
                    gdf, tile_bounds, new_height, new_width, new_transform, line_width, target_resolution
                )
                
                # Skip tiles with no line intersections
                if mask_arr is None or np.sum(mask_arr) == 0:
                    continue
                
                tiles_with_lines += 1
                
                # Pad or crop to exact tile size
                final_tile = pad_or_crop_to_tile_size(resampled_tile, tile_size)
                final_mask = pad_or_crop_to_tile_size(mask_arr, tile_size)
                
                # Update transform for final tile size
                final_transform = rasterio.Affine(
                    target_resolution, 0.0, new_transform.c,
                    0.0, -target_resolution, new_transform.f
                )
                
                # Create metadata for this tile
                tile_profile = profile.copy()
                tile_profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': final_transform,
                    'dtype': src.dtypes[0],
                    'count': config['data']['tiff_bands']
                })
                
                # Save tile with overlap and line width info in filename
                tile_filename = f"{basename}_{i}_{j}_zoom{res_str}m_overlap{overlap_str}_lw{line_width_str}.tif"
                tile_path = os.path.join(tiles_dir, tile_filename)
                with rasterio.open(tile_path, 'w', **tile_profile) as dst:
                    dst.write(final_tile)
                
                # Create mask profile
                mask_profile = profile.copy()
                mask_profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'count': 1,
                    'dtype': np.uint8,
                    'transform': final_transform,
                })
                
                # Save mask
                mask_filename = f"{basename}_{i}_{j}_zoom{res_str}m_overlap{overlap_str}_lw{line_width_str}.tif"
                mask_path = os.path.join(masks_dir, mask_filename)
                with rasterio.open(mask_path, 'w', **mask_profile) as mask_dst:
                    mask_dst.write(final_mask[np.newaxis, :, :])
                
                tile_count += 1
                
                if tile_count % 50 == 0:
                    print(f"Processed {tile_count} tiles with lines...")
            
            except Exception as e:
                print(f"Error processing tile {i}_{j}: {str(e)}")
                continue
        
        print(f"Total tiles processed: {tile_count}")
        print(f"Tiles with line intersections: {tiles_with_lines}")

def create_full_linestring_mask(input_tif, input_shp, output_mask_path, line_width_meters):
    """
    Create a full-size binary mask from a shapefile that matches the input TIF.
    Specifically designed for line string geometries with configurable line width.
    """
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        target_resolution = min(src.res)  # Use finer resolution
 
        # Read shapefile
        gdf = gpd.read_file(input_shp)
        print(f"Loaded {len(gdf)} geometries")
        print(f"Geometry types: {gdf.geometry.geom_type.value_counts().to_dict()}")
 
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
        
        print(f"Found {len(gdf)} geometries intersecting with raster")
 
        # Buffer the line geometries to create width
        buffered_geoms = []
        for geom in gdf.geometry:
            if geom.geom_type in ['LineString', 'MultiLineString']:
                # Buffer the line geometry by half the line width (radius)
                buffered_geom = geom.buffer(line_width_meters / 2)
                buffered_geoms.append(buffered_geom)
            else:
                # For other geometry types, use as is
                buffered_geoms.append(geom)

        # Create binary mask from buffered geometries
        shapes = [(geom, 1) for geom in buffered_geoms]
        mask_arr = rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        
        print(f"Mask statistics: min={mask_arr.min()}, max={mask_arr.max()}, mean={mask_arr.mean():.4f}")
        print(f"Line pixels: {np.sum(mask_arr)} / {mask_arr.size} ({np.sum(mask_arr)/mask_arr.size*100:.2f}%)")
 
        # Write the full mask to a GeoTIFF
        mask_profile = profile.copy()
        mask_profile.update({
            'count': 1,
            'dtype': np.uint8
        })
 
        with rasterio.open(output_mask_path, 'w', **mask_profile) as dst:
            dst.write(mask_arr[np.newaxis, :, :])
 
        print(f"Full mask saved to: {output_mask_path}")

def load_config(config_path='config/config_v1.yaml'):
    """
    Load configuration from YAML file with fallback defaults
    """
    default_config = {
        'data': {
            'input_size': 1024,
            'channels': 3,
            'batch_size': 1,
            'train_split': 0.8,
            'validation_split': 0.2,
            'random_state': 42,
            'resolution': 0.3,
            'overlap_percentage': 0.25,
            'line_width': 5,
            'tiff_bands': 3
        }
    }
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using default configuration")
        return default_config
    except Exception as e:
        print(f"Error loading config file: {str(e)}, using default configuration")
        return default_config

def process_all_files_with_zoom(input_dir, output_dir, config_path='config/config_v1.yaml'):
    """
    Process all TIF and SHP files in input directory with zoom level and line string handling
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get all TIF files
    tif_files = list(Path(input_dir).glob("*.tif"))
    tif_files.extend(list(Path(input_dir).glob("*.TIF")))  # Include uppercase
    
    if not tif_files:
        print("No TIF files found in input directory")
        return
    
    print(f"Found {len(tif_files)} TIF files")
    print("Checking for corresponding SHP files...")
    
    processed_count = 0
    
    for tif_file in tif_files:
        # Find corresponding SHP file
        shp_file = tif_file.with_suffix(".shp")
        if not shp_file.exists():
            # Try uppercase
            shp_file = tif_file.with_suffix(".SHP")
        
        basename = tif_file.stem
        
        if shp_file.exists():
            print(f"\n{'='*60}")
            print(f"Processing {tif_file.name} and {shp_file.name}")
            print(f"Configuration: tile_size={config['data']['input_size']}, "
                  f"resolution={config['data']['resolution']}m, "
                  f"line_width={config['data']['line_width']}m")
            try:
                create_tiles_with_zoom(
                    str(tif_file), 
                    str(shp_file), 
                    output_dir, 
                    config=config
                )
                print(f"✓ Successfully created tiles and masks for {basename}")
                processed_count += 1
            except Exception as e:
                print(f"✗ Error processing {tif_file.name}: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ No corresponding SHP file found for {tif_file.name}")
    
    print(f"\n{'='*60}")
    print(f"Processing completed: {processed_count}/{len(tif_files)} file pairs processed successfully")

if __name__ == "__main__":
    print("Line String Tile and Mask Generator")
    print("="*50)
    
    # Example usage
    input_folder = input("Enter input folder path: ").strip()
    output_folder = input("Enter output folder path: ").strip()
    config_path = input("Enter config file path (or press Enter for default 'config/config_v1.yaml'): ").strip()
    
    if not config_path:
        config_path = 'config/config_v1.yaml'
    
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    
    # Validate paths
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist!")
        exit(1)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Confirm with user
    print(f"\nInput folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Config file:   {config_path}")
    
    proceed = input("\nProceed with processing? (y/N): ").strip().lower()
    if proceed in ['y', 'yes']:
        print("\nStarting processing...")
        process_all_files_with_zoom(input_folder, output_folder, config_path)
        print("\nProcessing completed!")
    else:
        print("Processing cancelled.")