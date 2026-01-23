import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy.ndimage import minimum_filter, percentile_filter
import geopandas as gpd
from shapely.geometry import box
import os
from tqdm import tqdm
import multiprocessing
import argparse
import gc

os.environ["GDAL_CACHEMAX"] = "512"
os.environ["GDAL_NUM_THREADS"] = "2"


class FastTreeHeightCalculator:
    """
    Ultra-fast tree height calculator optimized for LARGE files (100+ GB).
    """
    
    def __init__(self, dem_path, shapefile_path, output_path, 
                 buffer_distance=10, ground_percentile=5, chunk_size=1024, 
                 use_simple_ground=True, num_workers=None):
        """
        Initialize the calculator.
        
        Parameters:
        -----------
        dem_path : str
            Path to input DEM raster
        shapefile_path : str
            Path to tree polygons shapefile
        output_path : str
            Path to output height raster
        buffer_distance : float
            Buffer distance in meters (default: 10m)
        ground_percentile : int
            Percentile for ground estimation (default: 5)
        chunk_size : int
            Size of processing chunks (default: 1024 - SMALLER for large files)
        use_simple_ground : bool
            Use fast minimum filter (default: True)
        num_workers : int
            Number of parallel workers (default: 1 for large files)
        """
        self.dem_path = dem_path
        self.shapefile_path = shapefile_path
        self.output_path = output_path
        self.buffer_distance = buffer_distance
        self.ground_percentile = ground_percentile
        self.chunk_size = chunk_size
        self.use_simple_ground = use_simple_ground
        self.num_workers = num_workers or 1
        
    def load_trees(self):
        """Load tree polygons and reproject to DEM CRS if needed."""
        print("Loading tree polygons...")
        self.trees = gpd.read_file(self.shapefile_path)
        
        with rasterio.open(self.dem_path) as src:
            self.dem_crs = src.crs
            self.dem_transform = src.transform
            self.dem_bounds = src.bounds
            self.dem_shape = src.shape
            self.dem_res = src.res[0]
            self.dem_nodata = src.nodata
            self.dem_dtype = src.dtypes[0]
            
        # Reproject trees if needed
        if self.trees.crs != self.dem_crs:
            print(f"Reprojecting trees from {self.trees.crs} to {self.dem_crs}")
            self.trees = self.trees.to_crs(self.dem_crs)
            
        # Create spatial index for faster queries
        self.trees_sindex = self.trees.sindex
            
        print(f"Loaded {len(self.trees)} tree polygons")
        print(f"DEM size: {self.dem_shape[0]} x {self.dem_shape[1]} pixels")
        print(f"DEM resolution: {self.dem_res:.2f}m")
        
        # Check file size and warn user
        file_size_gb = os.path.getsize(self.dem_path) / (1024**3)
        print(f"DEM file size: {file_size_gb:.2f} GB")
        if file_size_gb > 50:
            print("⚠️  LARGE FILE DETECTED - Using optimized settings")
            self.chunk_size = min(512, self.chunk_size)
            print(f"   Adjusted chunk size to: {self.chunk_size} pixels")
        
    def calculate_ground_simple(self):
        """
        Fast ground estimation with PROPER memory management for large files.
        """
        print(f"\n🚀 Fast ground estimation...")
        
        # Calculate filter size in pixels
        window_size = max(3, int(self.buffer_distance * 2 / self.dem_res))
        if window_size % 2 == 0:
            window_size += 1
            
        print(f"Filter window: {window_size} pixels ({window_size * self.dem_res:.1f}m)")
        
        ground_path = self.output_path.replace('.tif', '_ground.tif')
        
        # Open source once to get profile
        with rasterio.open(self.dem_path) as src:
            profile = src.profile.copy()
            profile.update(
                compress='lzw',
                tiled=True,
                blockxsize=256,
                blockysize=256,
                BIGTIFF='YES'
            )
        
        # Process in smaller chunks with proper memory management
        with rasterio.Env(GDAL_CACHEMAX=256, GDAL_SWATH_SIZE=200000000):
            with rasterio.open(self.dem_path) as src:
                with rasterio.open(ground_path, 'w', **profile) as dst:
                    
                    # Calculate chunks
                    num_chunks_y = int(np.ceil(self.dem_shape[0] / self.chunk_size))
                    num_chunks_x = int(np.ceil(self.dem_shape[1] / self.chunk_size))
                    total_chunks = num_chunks_y * num_chunks_x
                    
                    print(f"Processing {total_chunks} chunks of {self.chunk_size}x{self.chunk_size} pixels")
                    
                    with tqdm(total=total_chunks, desc="Ground surface") as pbar:
                        for i in range(num_chunks_y):
                            for j in range(num_chunks_x):
                                try:
                                    # Calculate window with overlap
                                    overlap = window_size // 2
                                    row_start = max(0, i * self.chunk_size - overlap)
                                    col_start = max(0, j * self.chunk_size - overlap)
                                    row_end = min(self.dem_shape[0], (i + 1) * self.chunk_size + overlap)
                                    col_end = min(self.dem_shape[1], (j + 1) * self.chunk_size + overlap)
                                    
                                    window = Window(col_start, row_start, 
                                                  col_end - col_start, row_end - row_start)
                                    
                                    # Read chunk
                                    chunk = src.read(1, window=window)
                                    
                                    # Handle nodata
                                    if self.dem_nodata is not None:
                                        mask = chunk == self.dem_nodata
                                    else:
                                        mask = np.isnan(chunk)
                                    
                                    # Fast ground estimation
                                    if not mask.all():
                                        chunk_filled = chunk.copy()
                                        if mask.any():
                                            chunk_filled[mask] = np.nanpercentile(chunk[~mask], 50)
                                        
                                        # Use minimum filter (MUCH faster than percentile)
                                        if self.use_simple_ground:
                                            ground_chunk = minimum_filter(chunk_filled, size=window_size)
                                        else:
                                            ground_chunk = percentile_filter(chunk_filled, 
                                                                            self.ground_percentile, 
                                                                            size=window_size)
                                        
                                        ground_chunk[mask] = self.dem_nodata if self.dem_nodata is not None else np.nan
                                    else:
                                        ground_chunk = chunk
                                    
                                    # Trim overlap
                                    actual_row_start = i * self.chunk_size
                                    actual_col_start = j * self.chunk_size
                                    actual_row_end = min(self.dem_shape[0], (i + 1) * self.chunk_size)
                                    actual_col_end = min(self.dem_shape[1], (j + 1) * self.chunk_size)
                                    
                                    trim_top = actual_row_start - row_start
                                    trim_left = actual_col_start - col_start
                                    trim_bottom = trim_top + (actual_row_end - actual_row_start)
                                    trim_right = trim_left + (actual_col_end - actual_col_start)
                                    
                                    ground_trimmed = ground_chunk[trim_top:trim_bottom, trim_left:trim_right]
                                    
                                    write_window = Window(actual_col_start, actual_row_start,
                                                        actual_col_end - actual_col_start,
                                                        actual_row_end - actual_row_start)
                                    
                                    # Write chunk
                                    dst.write(ground_trimmed, 1, window=write_window)
                                    
                                    # Clean up memory aggressively every 50 chunks
                                    if (i * num_chunks_x + j) % 50 == 0 and (i * num_chunks_x + j) > 0:
                                        gc.collect()
                                    
                                    # Clean up large arrays
                                    del chunk, ground_chunk, ground_trimmed
                                    if 'chunk_filled' in locals():
                                        del chunk_filled
                                    
                                except Exception as e:
                                    print(f"\n❌ Error at chunk ({i}, {j}): {str(e)}")
                                    raise
                                    
                                pbar.update(1)
        
        # Force garbage collection after processing
        gc.collect()
        
        print(f"✅ Ground surface saved")
        return ground_path
    
    def calculate_tree_heights_fast(self, ground_path):
        """
        Ultra-fast height calculation with proper memory management.
        """
        print(f"\n🌲 Calculating tree heights...")
        
        with rasterio.Env(GDAL_CACHEMAX=256):
            with rasterio.open(self.dem_path) as dem_src:
                with rasterio.open(ground_path) as ground_src:
                    
                    profile = dem_src.profile.copy()
                    profile.update(
                        dtype=rasterio.float32, 
                        nodata=-9999, 
                        compress='lzw',
                        tiled=True,
                        blockxsize=256,
                        blockysize=256,
                        BIGTIFF='YES'
                    )
                    
                    with rasterio.open(self.output_path, 'w', **profile) as dst:
                        num_chunks_y = int(np.ceil(self.dem_shape[0] / self.chunk_size))
                        num_chunks_x = int(np.ceil(self.dem_shape[1] / self.chunk_size))
                        total_chunks = num_chunks_y * num_chunks_x
                        
                        with tqdm(total=total_chunks, desc="Tree heights") as pbar:
                            for i in range(num_chunks_y):
                                for j in range(num_chunks_x):
                                    try:
                                        row_start = i * self.chunk_size
                                        col_start = j * self.chunk_size
                                        row_end = min(self.dem_shape[0], (i + 1) * self.chunk_size)
                                        col_end = min(self.dem_shape[1], (j + 1) * self.chunk_size)
                                        
                                        window = Window(col_start, row_start,
                                                      col_end - col_start, row_end - row_start)
                                        
                                        # Read chunks
                                        dem_chunk = dem_src.read(1, window=window)
                                        ground_chunk = ground_src.read(1, window=window)
                                        
                                        # Get chunk bounds and find intersecting trees
                                        chunk_bounds = rasterio.windows.bounds(window, dem_src.transform)
                                        chunk_box = box(*chunk_bounds)
                                        
                                        # Use spatial index for fast intersection
                                        possible_matches_idx = list(self.trees_sindex.intersection(chunk_bounds))
                                        possible_matches = self.trees.iloc[possible_matches_idx]
                                        trees_in_chunk = possible_matches[possible_matches.intersects(chunk_box)]
                                        
                                        # Initialize with nodata
                                        height_chunk = np.full_like(dem_chunk, -9999, dtype=np.float32)
                                        
                                        if len(trees_in_chunk) > 0:
                                            # Rasterize trees
                                            chunk_transform = dem_src.window_transform(window)
                                            tree_mask = rasterize(
                                                [(geom, 1) for geom in trees_in_chunk.geometry],
                                                out_shape=(row_end - row_start, col_end - col_start),
                                                transform=chunk_transform,
                                                fill=0,
                                                dtype=np.uint8
                                            )
                                            
                                            # Calculate heights
                                            valid_dem = dem_chunk != (self.dem_nodata if self.dem_nodata is not None else np.nan)
                                            valid_ground = ground_chunk != (self.dem_nodata if self.dem_nodata is not None else np.nan)
                                            valid_mask = valid_dem & valid_ground & (tree_mask == 1)
                                            
                                            if valid_mask.any():
                                                heights = dem_chunk - ground_chunk
                                                height_chunk[valid_mask] = heights[valid_mask]
                                                
                                                # Clean up negative heights
                                                height_chunk[(height_chunk >= 0) & (height_chunk < 0.3)] = 0.3
                                                height_chunk[height_chunk < 0] = 0
                                            
                                            del tree_mask
                                        
                                        dst.write(height_chunk, 1, window=window)
                                        
                                        # Garbage collection every 50 chunks
                                        if (i * num_chunks_x + j) % 50 == 0 and (i * num_chunks_x + j) > 0:
                                            gc.collect()
                                        
                                        # Clean up
                                        del dem_chunk, ground_chunk, height_chunk
                                        if 'heights' in locals():
                                            del heights
                                        
                                    except Exception as e:
                                        print(f"\n❌ Error at chunk ({i}, {j}): {str(e)}")
                                        raise
                                        
                                    pbar.update(1)
        
        # Force garbage collection
        gc.collect()
        
        print(f"✅ Tree heights saved to: {self.output_path}")
    
    def process(self):
        """Run the complete processing pipeline."""
        print("="*70)
        print("⚡ FAST TREE HEIGHT CALCULATOR (LARGE FILE OPTIMIZED)")
        print("="*70)
        
        import time
        start_time = time.time()
        
        # Load data
        self.load_trees()
        
        # Calculate ground surface
        ground_path = self.calculate_ground_simple()
        
        # Calculate tree heights
        self.calculate_tree_heights_fast(ground_path)
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f"✅ COMPLETE! Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print("="*70)


# =============================================================================
# MAIN EXECUTION - OPTIMIZED FOR LARGE FILES (100+ GB)
# =============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process DEM and calculate tree heights (optimized for large files)")
    parser.add_argument("--dem_path", required=True, help="Path to DEM file")
    parser.add_argument("--output_path", required=True, help="Path to output height file")
    parser.add_argument("--shapefile_path", required=True, help="Path to tree polygons shapefile")
    parser.add_argument("--buffer_distance", type=float, default=8, help="Buffer distance in meters (default: 8)")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size in pixels (default: 512, use 256-512 for 100+GB files)")
    
    args = parser.parse_args()

    # Configuration
    DEM_PATH = args.dem_path
    SHAPEFILE_PATH = args.shapefile_path
    OUTPUT_PATH = args.output_path
    BUFFER_DISTANCE = args.buffer_distance
    CHUNK_SIZE = args.chunk_size
    USE_SIMPLE_GROUND = True
    
    print("\n⚙️  Configuration:")
    print(f"   Buffer distance: {BUFFER_DISTANCE}m")
    print(f"   Chunk size: {CHUNK_SIZE} pixels")
    print(f"   Ground method: Minimum filter (FAST)")
    print(f"   Memory: Optimized for large files (100+ GB)")
    
    # Create calculator
    calculator = FastTreeHeightCalculator(
        dem_path=DEM_PATH,
        shapefile_path=SHAPEFILE_PATH,
        output_path=OUTPUT_PATH,
        buffer_distance=BUFFER_DISTANCE,
        chunk_size=CHUNK_SIZE,
        use_simple_ground=USE_SIMPLE_GROUND
    )
    
    # Run processing
    try:
        calculator.process()
        
        print(f"\n📂 Output files:")
        print(f"   🌲 Tree heights: {OUTPUT_PATH}")
        print(f"   🏔️  Ground surface: {OUTPUT_PATH.replace('.tif', '_ground.tif')}")
        print(f"\n✨ Ready for classification!")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)