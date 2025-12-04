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
from scipy import ndimage
from skimage.morphology import skeletonize

def create_tiles(input_tif, input_shp, output_dir, tile_size=256, line_width=3):
    """
    Create tiles from TIF file and corresponding binary line masks from SHP file
    """
    # Create output directories
    basename = os.path.splitext(os.path.basename(input_tif))[0]
    tiles_dir = os.path.join(output_dir, f"tiles_{tile_size}_{basename}")
    masks_dir = os.path.join(output_dir, f"masks_{tile_size}_{basename}")
    
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Read the TIF file with full metadata
    with rasterio.open(input_tif) as src:
        # Store the source metadata for later use
        profile = src.profile.copy()
        
        # Read the shapefile
        gdf = gpd.read_file(input_shp)
        
        # Filter for LineString and MultiLineString geometries only
        line_geoms = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        
        if len(line_geoms) == 0:
            print(f"Warning: No line geometries found in {input_shp}")
            return
        
        # CRS handling
        if line_geoms.crs is None:
            print(f"Setting shapefile CRS to match raster: {src.crs}")
            line_geoms.set_crs(src.crs, inplace=True)
        elif line_geoms.crs != src.crs:
            print(f"Reprojecting shapefile from {line_geoms.crs} to {src.crs}")
            line_geoms = line_geoms.to_crs(src.crs)
        
        # Filter geometries that intersect with raster bounds
        raster_bounds = box(*src.bounds)
        line_geoms = line_geoms[line_geoms.geometry.intersects(raster_bounds)]
        
        if len(line_geoms) == 0:
            print(f"Warning: No line geometries intersect with {basename}")
            return
        
        # Get dimensions
        height = src.height
        width = src.width
        num_bands = src.count
        
        # Calculate tiles
        n_tiles_height = int(np.ceil(height / tile_size))
        n_tiles_width = int(np.ceil(width / tile_size))

        selected_bands = [1, 2, 3] 
        
        # Read the entire raster into memory (if it's not too large)
        try:
            # This can be memory-intensive for large rasters
            full_img = src.read(selected_bands)
            use_full_img = True
            num_bands = len(selected_bands)
            print(f"Read full image into memory: {full_img.shape}")
        except MemoryError:
            use_full_img = False
            num_bands = len(selected_bands)
            print("Image too large, processing tile by tile")
        
        # Process each tile
        for i in range(n_tiles_height):
            for j in range(n_tiles_width):
                # Define the exact window dimensions
                col_off = j * tile_size
                row_off = i * tile_size
                width_window = min(tile_size, width - col_off)
                height_window = min(tile_size, height - row_off)
                
                window = Window(col_off, row_off, width_window, height_window)
                
                # Get tile data
                if use_full_img:
                    # Extract from the full image in memory
                    raw_tile = full_img[:, row_off:row_off+height_window, col_off:col_off+width_window]
                else:
                    # Read directly from disk
                    raw_tile = src.read(selected_bands, window=window)
                
                # Handle padding for edge tiles
                tile = np.zeros((num_bands, tile_size, tile_size), dtype=src.dtypes[0])
                tile[:, :height_window, :width_window] = raw_tile
                
                # Skip empty tiles (all zeros or very low values)
                if np.mean(tile) < 0.01:
                    continue
                
                # Get precise window transform
                window_transform = rasterio.windows.transform(window, src.transform)
                
                # Calculate tile bounds with buffer for line intersection
                tile_bounds = rasterio.windows.bounds(window, src.transform)
                # Add small buffer to catch lines that might cross tile boundaries
                buffer_size = abs(src.transform[0]) * line_width  # Buffer based on pixel size and line width
                tile_box = box(tile_bounds[0] - buffer_size, tile_bounds[1] - buffer_size, 
                              tile_bounds[2] + buffer_size, tile_bounds[3] + buffer_size)
                
                # Find intersecting line geometries
                intersecting_lines = line_geoms[line_geoms.geometry.intersects(tile_box)]
                if len(intersecting_lines) == 0:
                    continue
                
                # Create line mask using rasterize with line width
                shapes = [(geom, 1) for geom in intersecting_lines.geometry]
                
                # Create initial thin line mask
                thin_mask = rasterize(
                    shapes,
                    out_shape=(tile_size, tile_size),
                    transform=window_transform,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
                )
                
                # Apply line width using morphological dilation
                if line_width > 1:
                    # Create structuring element for line thickness
                    struct_elem = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
                    # Dilate to create line width
                    mask_arr = ndimage.binary_dilation(thin_mask, 
                                                     structure=struct_elem, 
                                                     iterations=line_width//2).astype(np.uint8)
                else:
                    mask_arr = thin_mask
                
                # If mask is empty, skip
                if np.sum(mask_arr) == 0:
                    continue
                
                try:
                    # Create metadata for this tile
                    tile_profile = profile.copy()
                    tile_profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'count': num_bands,
                        'transform': window_transform,
                        # Keep all other metadata the same (nodata, dtype, etc.)
                    })
                    
                    # Save tile
                    tile_path = os.path.join(tiles_dir, f"{basename}_{i*tile_size}_{j*tile_size}.tif")
                    with rasterio.open(tile_path, 'w', **tile_profile) as dst:
                        dst.write(tile)
                    
                    # Create mask profile
                    mask_profile = profile.copy()
                    mask_profile.update({
                        'height': tile_size,
                        'width': tile_size,
                        'count': 1,
                        'dtype': np.uint8,
                        'transform': window_transform,
                    })

                    mask_profile.pop('nodata', None)
                    
                    # Save mask
                    mask_path = os.path.join(masks_dir, f"{basename}_{i*tile_size}_{j*tile_size}.tif")
                    with rasterio.open(mask_path, 'w', **mask_profile) as mask_dst:
                        mask_dst.write(mask_arr[np.newaxis, :, :])
                
                except Exception as e:
                    print(f"Error processing tile {i}_{j}: {str(e)}")
                    continue

def create_full_mask(input_tif, input_shp, output_mask_path, line_width=3):
    """
    Create a full-size binary line mask from a shapefile that matches the input TIF.
    """
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
 
        # Read shapefile
        gdf = gpd.read_file(input_shp)
        
        # Filter for line geometries only
        line_geoms = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        
        if len(line_geoms) == 0:
            print("No line geometries found in shapefile.")
            return
 
        # Align CRS
        if line_geoms.crs is None:
            print("Setting CRS of shapefile to match raster.")
            line_geoms.set_crs(src.crs, inplace=True)
        elif line_geoms.crs != src.crs:
            print(f"Reprojecting shapefile from {line_geoms.crs} to {src.crs}")
            line_geoms = line_geoms.to_crs(src.crs)
 
        # Filter geometries within raster bounds
        raster_bounds = box(*src.bounds)
        line_geoms = line_geoms[line_geoms.geometry.intersects(raster_bounds)]
 
        if len(line_geoms) == 0:
            print("No line geometries intersect with raster.")
            return
 
        # Create binary line mask from geometries
        shapes = [(geom, 1) for geom in line_geoms.geometry]
        
        # Create thin line mask first
        thin_mask = rasterize(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=True,
            dtype=np.uint8
        )
        
        # Apply line width using morphological dilation
        if line_width > 1:
            # Create structuring element for line thickness
            struct_elem = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
            # Dilate to create line width
            mask_arr = ndimage.binary_dilation(thin_mask, 
                                             structure=struct_elem, 
                                             iterations=line_width//2).astype(np.uint8)
        else:
            mask_arr = thin_mask
 
        # Write the full mask to a GeoTIFF
        mask_profile = profile.copy()
        mask_profile.update({
            'count': 1,
            'dtype': np.uint8
        })
 
        with rasterio.open(output_mask_path, 'w', **mask_profile) as dst:
            dst.write(mask_arr[np.newaxis, :, :])
 
        print(f"Full line mask saved to: {output_mask_path}")

def process_all_files(tiff_file_dir, output_dir, line_width=3):
    """
    Process all TIF and SHP files in input directory for line segmentation
    """
    # Get all TIF files
    tif_files = list(Path(tiff_file_dir).glob("*.tif"))
    print("Checking pairs...")

    try:
        with open('config/config_v1.yaml', 'r') as f:
            config = yaml.safe_load(f)
            tile_size = config['data']['input_size']
    except (FileNotFoundError, KeyError):
        print("Config file not found or missing input_size, using default tile_size=256")
        tile_size = 256
    
    for tif_file in tif_files:
        # Find corresponding SHP file
        basename = tif_file.stem

        shp_file = tif_file.with_suffix(".shp")
        basename = tif_file.stem
        
        if shp_file.exists():
            print(f"Processing {tif_file.name} and {shp_file.name} for line segmentation")
            try:
                full_mask_path = os.path.join(output_dir, f"{basename}_full_line_mask.tif")
                create_full_mask(str(tif_file), str(shp_file), full_mask_path, line_width)

                create_tiles(str(tif_file), str(shp_file), output_dir, 
                           tile_size=tile_size, line_width=line_width)
            except Exception as e:
                print(f"Error processing {tif_file.name}: {str(e)}")
        else:
            print(f"No corresponding SHP file found for {tif_file.name}")

if __name__ == "__main__":

    # Example usage
    tiff_files_folder = input("Enter input tiff file folder path: ").strip()
    output_folder = input("Enter output folder path: ").strip()
    
    # Ask for line width
    line_width_input = input("Enter line width in pixels (default=3): ").strip()
    line_width = int(line_width_input) if line_width_input else 3

    tiff_files_folder = os.path.abspath(tiff_files_folder)
    output_folder = os.path.abspath(output_folder)
    
    # Confirm with user
    print(f"\Tiff File folder:  {tiff_files_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Line width:    {line_width} pixels")
    print("Processing for line segmentation...")
    process_all_files(tiff_files_folder, output_folder, line_width)