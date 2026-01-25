# OPTIMIZED TILE AND MASK CREATION FOR VEGETATION DETECTION
# Specifically designed for RGB data with vegetation digitization

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
from tqdm import tqdm


def adaptive_scale_tile(tile):
    """
    Adaptive scaling for vegetation detection with RGB data.
    
    Args:
        tile: (bands, H, W) numpy array
        
    Returns:
        scaled_tile: (bands, H, W) uint8 array
    """
    scaled_bands = []
    
    for b in range(tile.shape[0]):
        band = tile[b].copy().astype(np.float32)
        
        if np.any(band > 0):
            # Use aggressive contrast enhancement for vegetation
            # Vegetation typically has high green values
            p_low = np.percentile(band[band > 0], 1.0)    # 1% instead of 2.5%
            p_high = np.percentile(band[band > 0], 99.0)  # 99% instead of 99%
            
            # Clip
            band = np.clip(band, p_low, p_high)
            
            # Normalize
            if (p_high - p_low) > 0:
                band = (band - p_low) / (p_high - p_low)
            else:
                band = np.zeros_like(band)
            
            # Convert to uint8
            band = (band * 255).astype(np.uint8)
            
            # Avoid pure zeros
            band[band == 0] = 1
        else:
            band = np.zeros_like(band, dtype=np.uint8)
        
        scaled_bands.append(band)
    
    return np.stack(scaled_bands)


def extract_vegetation_patches(band):
    """
    Extract connected vegetation patches to focus sampling on actual vegetation.
    """
    from scipy import ndimage
    
    # Binary mask
    binary = (band > 0).astype(np.uint8)
    
    # Label connected components
    labeled, num_features = ndimage.label(binary)
    
    # Get patch info
    patches = []
    for i in range(1, num_features + 1):
        patch_coords = np.argwhere(labeled == i)
        if len(patch_coords) > 0:
            patches.append(patch_coords)
    
    return patches


def create_vegetation_tiles(input_tif, input_shp, output_dir, tile_size=512, overlap_percent=0.25):
    """
    Create tiles specifically optimized for vegetation detection.
    
    Args:
        input_tif: Path to input RGB TIFF
        input_shp: Path to vegetation mask shapefile
        output_dir: Output directory for tiles and masks
        tile_size: Size of tiles (default 512x512 for 0.1m = 51.2m ground size)
        overlap_percent: Overlap percentage (0.25 = 25%)
    """
    
    print("\n" + "="*70)
    print("🌳 VEGETATION TILE CREATION - OPTIMIZED")
    print("="*70)
    print(f"📊 Tile size: {tile_size}x{tile_size} pixels")
    print(f"🔄 Overlap: {overlap_percent*100:.0f}%")
    
    basename = os.path.splitext(os.path.basename(input_tif))[0]
    
    tiles_dir = os.path.join(output_dir, "tiles_veg")
    masks_dir = os.path.join(output_dir, "masks_veg")
    
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    print(f"\n📁 Output directories:")
    print(f"   • Tiles: {tiles_dir}")
    print(f"   • Masks: {masks_dir}\n")
    
    # Read raster
    with rasterio.open(input_tif) as src:
        overlap_pixels = int(tile_size * overlap_percent)
        stride = tile_size - overlap_pixels
        
        if stride <= 0:
            raise ValueError("Overlap percentage too high")
        
        profile = src.profile.copy()
        width = src.width
        height = src.height
        transform = src.transform
        
        # Verify RGB (3 bands)
        if src.count != 3:
            print(f"⚠️  Warning: Expected 3 bands, got {src.count}")
        
        selected_bands = list(range(1, min(4, src.count + 1)))
        num_bands = len(selected_bands)
        
        # Load and fix shapefile
        gdf = gpd.read_file(input_shp)
        
        if gdf.crs is None:
            print(f"🔧 Setting CRS to: {src.crs}")
            gdf.set_crs(src.crs, inplace=True)
        elif gdf.crs != src.crs:
            print(f"🔧 Reprojecting shapefile from {gdf.crs} → {src.crs}")
            gdf = gdf.to_crs(src.crs)
        
        # Filter geometries outside raster
        raster_bounds = box(*src.bounds)
        gdf_clipped = gdf[gdf.geometry.intersects(raster_bounds)].copy()
        
        print(f"📊 Shapefile contains {len(gdf)} features")
        print(f"✅ {len(gdf_clipped)} features within raster bounds\n")
        
        if len(gdf_clipped) == 0:
            print("❌ No features found within raster bounds!")
            return
        
        # Create tiles
        tile_count = 0
        pbar = tqdm(desc="🔨 Creating tiles", unit="tile")
        
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                # Calculate window
                window_height = min(tile_size, height - y)
                window_width = min(tile_size, width - x)
                
                if window_height < tile_size // 2 or window_width < tile_size // 2:
                    continue
                
                window = Window(x, y, window_width, window_height)
                
                # Read tile
                tile_data = src.read(selected_bands, window=window)
                
                # Skip empty tiles
                if np.all(tile_data == 0):
                    continue
                
                # Scale tile
                tile_scaled = adaptive_scale_tile(tile_data)
                
                # Pad if necessary
                if window_height < tile_size or window_width < tile_size:
                    tile_padded = np.zeros((num_bands, tile_size, tile_size), dtype=np.uint8)
                    tile_padded[:, :window_height, :window_width] = tile_scaled
                    tile_scaled = tile_padded
                
                # Create mask from shapefile
                tile_window_bounds = src.window_bounds(window)
                tile_geom_box = box(*tile_window_bounds)
                
                # Clip geometries to tile
                tile_geoms = gdf_clipped[gdf_clipped.geometry.intersects(tile_geom_box)].copy()
                
                if len(tile_geoms) == 0:
                    continue
                
                # Create binary mask
                mask_data = np.zeros((tile_size, tile_size), dtype=np.uint8)
                
                try:
                    # Rasterize vegetation geometries
                    from rasterio.features import rasterize
                    
                    # Transform geometries to tile coordinates
                    tile_transform = rasterio.transform.from_bounds(
                        *tile_window_bounds,
                        tile_size,
                        tile_size
                    )
                    
                    shapes = [(geom, 1) for geom in tile_geoms.geometry]
                    if shapes:
                        mask_rasterized = rasterize(
                            shapes,
                            out_shape=(tile_size, tile_size),
                            transform=tile_transform,
                            default_value=0
                        )
                        mask_data = (mask_rasterized > 0).astype(np.uint8) * 255
                
                except Exception as e:
                    print(f"⚠️  Error creating mask: {e}")
                    continue
                
                # Skip tiles with very little vegetation
                veg_pixels = np.sum(mask_data > 0)
                veg_percentage = (veg_pixels / (tile_size * tile_size)) * 100
                
                if veg_percentage < 1:  # At least 1% vegetation
                    continue
                
                # Save tile
                tile_filename = f"{basename}_tile_{tile_count:05d}.tif"
                tile_path = os.path.join(tiles_dir, tile_filename)
                
                profile_tile = profile.copy()
                profile_tile.update(
                    height=tile_size,
                    width=tile_size,
                    count=num_bands,
                    dtype='uint8'
                )
                
                with rasterio.open(tile_path, 'w', **profile_tile) as dst:
                    dst.write(tile_scaled)
                
                # Save mask
                mask_filename = f"{basename}_tile_{tile_count:05d}.tif"
                mask_path = os.path.join(masks_dir, mask_filename)
                
                profile_mask = profile.copy()
                profile_mask.update(
                    height=tile_size,
                    width=tile_size,
                    count=1,
                    dtype='uint8'
                )
                
                with rasterio.open(mask_path, 'w', **profile_mask) as dst:
                    dst.write(mask_data, 1)
                
                tile_count += 1
                pbar.update(1)
        
        pbar.close()
        
        print(f"\n✅ Created {tile_count} vegetation tiles")
        print(f"📊 Average tile size: {tile_size}x{tile_size} pixels")
        print(f"🔗 0.1m resolution = ~{tile_size * 0.1:.1f}m x {tile_size * 0.1:.1f}m ground area")
        
        return tiles_dir, masks_dir, tile_count


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create optimized vegetation tiles from RGB raster and shapefile'
    )
    parser.add_argument('--input_tif', required=True,
                       help='Path to input RGB TIFF file')
    parser.add_argument('--input_shp', required=True,
                       help='Path to vegetation mask shapefile')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for tiles and masks')
    parser.add_argument('--tile_size', type=int, default=512,
                       help='Tile size in pixels (default 512)')
    parser.add_argument('--overlap', type=float, default=0.25,
                       help='Tile overlap as fraction (default 0.25 = 25%%)')
    
    args = parser.parse_args()
    
    # Verify inputs
    if not os.path.exists(args.input_tif):
        print(f"❌ Input TIFF not found: {args.input_tif}")
        return
    
    if not os.path.exists(args.input_shp):
        print(f"❌ Input SHP not found: {args.input_shp}")
        return
    
    # Create tiles
    create_vegetation_tiles(
        args.input_tif,
        args.input_shp,
        args.output_dir,
        tile_size=args.tile_size,
        overlap_percent=args.overlap
    )
    
    print("\n" + "="*70)
    print("🎉 Tile creation completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
