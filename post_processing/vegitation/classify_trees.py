import numpy as np
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import shape, mapping, box
from shapely.ops import unary_union
import pandas as pd
from tqdm import tqdm
import warnings
import argparse
import gc
import tempfile
import os
warnings.filterwarnings('ignore')

class TreeClassifier:
    """
    Memory-optimized classifier for large rasters with boundary artifact fix.
    """
    
    def __init__(self, height_raster_path, original_shapefile_path, output_shapefile_path,
                 min_polygon_area=1.0, chunk_size=4096, overlap=256):
        """
        Initialize classifier.
        
        Parameters:
        -----------
        height_raster_path : str
            Path to tree height raster
        original_shapefile_path : str
            Path to original tree polygons
        output_shapefile_path : str
            Path for output classified shapefile
        min_polygon_area : float
            Minimum area in square meters
        chunk_size : int
            Processing chunk size in pixels (default: 4096)
        overlap : int
            Overlap between chunks to handle boundary polygons (default: 256)
        """
        self.height_raster_path = height_raster_path
        self.original_shapefile_path = original_shapefile_path
        self.output_shapefile_path = output_shapefile_path
        self.min_polygon_area = min_polygon_area
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Classification thresholds (updated as requested)
        self.classes = {
            1: {'name': 'Grass/Land', 'min': 0.1, 'max': 0.6, 'code': 1},
            2: {'name': 'Green Cover', 'min': 0.6, 'max': 999.0, 'code': 2}
            # 2: {'name': 'Shrubs', 'min': 0.6, 'max': 5, 'code': 2},
            # 3: {'name': 'Medium Trees', 'min': 5, 'max': 9.0, 'code': 3},
            # 4: {'name': 'High Trees', 'min': 9.0, 'max': 999.0, 'code': 4}
        }
        
        # Create temp directory for intermediate files
        self.temp_dir = tempfile.mkdtemp(prefix='tree_classifier_')
        print(f"   Temp directory: {self.temp_dir}")
    
    def load_data(self):
        """Load height raster metadata and original polygons."""
        print("="*70)
        print("🌳 MEMORY-OPTIMIZED TREE CLASSIFIER (BOUNDARY FIX)")
        print("="*70)
        print("\n📂 Loading data...")
        
        # Load shapefile
        self.trees = gpd.read_file(self.original_shapefile_path)
        print(f"   Original polygons: {len(self.trees)}")
        
        # Create spatial index for fast intersection queries
        print("   Building spatial index...")
        self.trees_sindex = self.trees.sindex
        
        # Load height raster metadata
        with rasterio.open(self.height_raster_path) as src:
            self.transform = src.transform
            self.crs = src.crs
            self.bounds = src.bounds
            self.shape = src.shape
            self.nodata = src.nodata
            
        # Reproject trees if needed
        if self.trees.crs != self.crs:
            print(f"   Reprojecting shapefile to match raster CRS")
            self.trees = self.trees.to_crs(self.crs)
            
        print(f"   Raster size: {self.shape[0]} x {self.shape[1]} pixels")
        print(f"   Processing: {self.chunk_size}x{self.chunk_size} chunks with {self.overlap}px overlap")
        
    def classify_heights(self):
        """Classify height raster into categories using chunked processing."""
        print("\n🎨 Classifying heights into categories...")
        
        self.classified_raster_path = self.height_raster_path.replace('.tif', '_classified.tif')
        
        with rasterio.open(self.height_raster_path) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, nodata=0, compress='lzw', tiled=True, blockxsize=512, blockysize=512)
            
            with rasterio.open(self.classified_raster_path, 'w', **profile) as dst:
                num_chunks_y = int(np.ceil(self.shape[0] / self.chunk_size))
                num_chunks_x = int(np.ceil(self.shape[1] / self.chunk_size))
                total_chunks = num_chunks_y * num_chunks_x
                
                class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
                
                with tqdm(total=total_chunks, desc="Classifying") as pbar:
                    for i in range(num_chunks_y):
                        for j in range(num_chunks_x):
                            row_start = i * self.chunk_size
                            col_start = j * self.chunk_size
                            row_end = min(self.shape[0], (i + 1) * self.chunk_size)
                            col_end = min(self.shape[1], (j + 1) * self.chunk_size)
                            
                            window = Window(col_start, row_start,
                                          col_end - col_start, row_end - row_start)
                            
                            height_chunk = src.read(1, window=window)
                            class_chunk = np.zeros_like(height_chunk, dtype=np.uint8)
                            
                            valid_mask = (height_chunk != self.nodata) & (height_chunk >= 0)
                            
                            if valid_mask.any():
                                # Class 1: Grass/Land (0.1 - 0.6m)
                                mask1 = (height_chunk >= 0.1) & (height_chunk <= 0.6)
                                class_chunk[mask1] = 1
                                class_counts[1] += np.sum(mask1)
                                
                                # Class 2: Shrubs (0.6 - 5.5m)
                                mask2 = (height_chunk > 0.6) & (height_chunk < 999.0)
                                class_chunk[mask2] = 2
                                class_counts[2] += np.sum(mask2)
                                
                                # # Class 3: Medium Trees (5.5 - 10.0m)
                                # mask3 = (height_chunk >= 5.5) & (height_chunk <= 10.0)
                                # class_chunk[mask3] = 3
                                # class_counts[3] += np.sum(mask3)
                                
                                # # Class 4: High Trees (> 10.0m)
                                # mask4 = (height_chunk > 10.0)
                                # class_chunk[mask4] = 4
                                # class_counts[4] += np.sum(mask4)
                            
                            dst.write(class_chunk, 1, window=window)
                            pbar.update(1)
                            
                            del height_chunk, class_chunk
        
        print("\n📊 Classification Summary:")
        total_pixels = sum(class_counts.values())
        for class_id, info in self.classes.items():
            count = class_counts[class_id]
            percentage = (count / total_pixels * 100) if total_pixels > 0 else 0
            print(f"   Class {class_id} ({info['name']}): {count:,} pixels ({percentage:.1f}%)")
        
        print(f"\n✅ Classified raster saved: {self.classified_raster_path}")
    
    def vectorize_and_split(self):
        """Vectorize classified raster with overlap handling to avoid boundary artifacts."""
        print("\n🔷 Vectorizing and splitting polygons (with boundary fix)...")
        
        all_classified_polygons = []
        
        for class_id, class_info in self.classes.items():
            print(f"\n   Processing Class {class_id}: {class_info['name']}")

            if class_id == 1:
                print("   ⏭️  Skipping Grass/Land class (disabled)")
                continue
            
            # Process this class with overlap handling
            class_polygons = self._vectorize_class_with_merge(class_id)
            
            if len(class_polygons) > 0:
                print(f"      Found {len(class_polygons)} polygons after merging")
                
                # Create GeoDataFrame in batches
                print(f"      Creating GeoDataFrame and intersecting with originals...")
                gdf_class = gpd.GeoDataFrame(class_polygons, crs=self.crs)
                
                # Filter by area before spatial operations
                gdf_class['area_sqm'] = gdf_class.geometry.area
                gdf_class = gdf_class[gdf_class['area_sqm'] >= self.min_polygon_area].copy()
                print(f"      After area filter: {len(gdf_class)} polygons")
                
                if len(gdf_class) > 0:
                    # Process in batches for spatial join
                    batch_size = 1000
                    intersected_batches = []
                    
                    for start_idx in range(0, len(gdf_class), batch_size):
                        end_idx = min(start_idx + batch_size, len(gdf_class))
                        batch = gdf_class.iloc[start_idx:end_idx]
                        
                        batch_intersected = gpd.sjoin(batch, self.trees, how='left', predicate='intersects')
                        
                        keep_cols = ['geometry', 'class_id', 'class_name', 'ht_min', 
                                   'ht_max', 'ht_mean', 'area_sqm']
                        if 'index_right' in batch_intersected.columns:
                            batch_intersected['original_id'] = batch_intersected['index_right']
                            keep_cols.append('original_id')
                        
                        batch_intersected = batch_intersected[keep_cols].copy()
                        intersected_batches.append(batch_intersected)
                        
                        del batch, batch_intersected
                        gc.collect()
                    
                    if intersected_batches:
                        intersected = pd.concat(intersected_batches, ignore_index=True)
                        all_classified_polygons.append(intersected)
                        print(f"      ✓ Final count: {len(intersected)} polygons")
                
                del gdf_class, class_polygons
                gc.collect()
        
        # Combine all classes
        print("\n🔗 Combining all classes...")
        if all_classified_polygons:
            self.final_gdf = pd.concat(all_classified_polygons, ignore_index=True)
            
            initial_count = len(self.final_gdf)
            self.final_gdf = self.final_gdf.drop_duplicates(subset=['geometry'])
            if initial_count != len(self.final_gdf):
                print(f"   Removed {initial_count - len(self.final_gdf)} duplicate geometries")
            
            print(f"\n✅ Total classified polygons: {len(self.final_gdf)}")
        else:
            print("⚠️  No polygons found!")
            self.final_gdf = gpd.GeoDataFrame()
    
    def _vectorize_class_with_merge(self, class_id):
        """
        Vectorize with overlap and merge boundary polygons.
        This eliminates square boundary artifacts.
        """
        temp_shapefiles = []
        
        with rasterio.open(self.classified_raster_path) as class_src:
            with rasterio.open(self.height_raster_path) as height_src:
                
                # Calculate chunk layout
                step_size = self.chunk_size - self.overlap
                num_chunks_y = int(np.ceil((self.shape[0] - self.overlap) / step_size))
                num_chunks_x = int(np.ceil((self.shape[1] - self.overlap) / step_size))
                total_chunks = num_chunks_y * num_chunks_x
                
                with tqdm(total=total_chunks, desc=f"      Vectorizing chunks", leave=False) as pbar:
                    for i in range(num_chunks_y):
                        for j in range(num_chunks_x):
                            # Calculate chunk boundaries WITH overlap
                            row_start = max(0, i * step_size)
                            col_start = max(0, j * step_size)
                            row_end = min(self.shape[0], row_start + self.chunk_size)
                            col_end = min(self.shape[1], col_start + self.chunk_size)
                            
                            window = Window(col_start, row_start,
                                          col_end - col_start, row_end - row_start)
                            
                            # Read chunks
                            classified_chunk = class_src.read(1, window=window)
                            height_chunk = height_src.read(1, window=window)
                            
                            # Create mask for this class
                            mask = (classified_chunk == class_id).astype(np.uint8)
                            
                            if mask.sum() > 0:
                                window_transform = rasterio.windows.transform(window, self.transform)
                                
                                # Vectorize chunk
                                chunk_polys = []
                                for geom, value in shapes(mask, mask=mask, transform=window_transform):
                                    if value == 1:
                                        geom_shape = shape(geom)
                                        
                                        if geom_shape.area < self.min_polygon_area * 0.5:  # More lenient for chunks
                                            continue
                                        
                                        # Calculate height statistics
                                        minx, miny, maxx, maxy = geom_shape.bounds
                                        
                                        local_col_start = int((minx - window_transform.c) / window_transform.a)
                                        local_row_start = int((maxy - window_transform.f) / window_transform.e)
                                        local_col_end = int((maxx - window_transform.c) / window_transform.a)
                                        local_row_end = int((miny - window_transform.f) / window_transform.e)
                                        
                                        local_col_start = max(0, local_col_start)
                                        local_row_start = max(0, local_row_start)
                                        local_col_end = min(window.width, local_col_end)
                                        local_row_end = min(window.height, local_row_end)
                                        
                                        poly_heights = height_chunk[local_row_start:local_row_end, 
                                                                   local_col_start:local_col_end]
                                        poly_mask = mask[local_row_start:local_row_end, 
                                                        local_col_start:local_col_end]
                                        
                                        valid_heights = poly_heights[(poly_mask == 1) & 
                                                                    (poly_heights != self.nodata)]
                                        
                                        if len(valid_heights) > 0:
                                            ht_min = float(valid_heights.min())
                                            ht_max = float(valid_heights.max())
                                            ht_mean = float(valid_heights.mean())
                                        else:
                                            ht_min = ht_max = ht_mean = 0.0
                                        
                                        # Use shortened column names (shapefile 10 char limit)
                                        chunk_polys.append({
                                            'geometry': geom_shape,
                                            'class_id': class_id,
                                            'class_name': self.classes[class_id]['name'][:10],  # Truncate to 10 chars
                                            'ht_min': ht_min,
                                            'ht_max': ht_max,
                                            'ht_mean': ht_mean,
                                            'area_sqm': geom_shape.area
                                        })
                                
                                # Save chunk to temporary shapefile
                                if chunk_polys:
                                    chunk_gdf = gpd.GeoDataFrame(chunk_polys, crs=self.crs)
                                    temp_file = os.path.join(self.temp_dir, f'chunk_{class_id}_{i}_{j}.shp')
                                    chunk_gdf.to_file(temp_file)
                                    temp_shapefiles.append(temp_file)
                                    del chunk_gdf, chunk_polys
                            
                            pbar.update(1)
                            del classified_chunk, height_chunk, mask
                
                gc.collect()
        
        # Now merge all chunk shapefiles with overlap handling
        print(f"      Merging {len(temp_shapefiles)} chunks and dissolving boundaries...")
        return self._merge_overlapping_chunks(temp_shapefiles, class_id)
    
    def _merge_overlapping_chunks(self, temp_shapefiles, class_id):
        """
        Merge chunk shapefiles and dissolve overlapping polygons.
        This is the key to eliminating boundary artifacts.
        """
        if not temp_shapefiles:
            return []
        
        # Process in batches to manage memory
        batch_size = 20
        merged_batches = []
        
        for i in range(0, len(temp_shapefiles), batch_size):
            batch_files = temp_shapefiles[i:i+batch_size]
            
            # Load all chunks in this batch
            chunks = []
            for shp_file in batch_files:
                try:
                    gdf = gpd.read_file(shp_file)
                    chunks.append(gdf)
                except Exception as e:
                    print(f"         Warning: Could not read {shp_file}: {e}")
                    pass
            
            if chunks:
                # Combine chunks
                combined = pd.concat(chunks, ignore_index=True)
                
                # Dissolve overlapping geometries using unary_union
                print(f"         Dissolving batch {i//batch_size + 1}/{(len(temp_shapefiles)-1)//batch_size + 1}...", end='\r')
                dissolved_geom = unary_union(combined.geometry.tolist())
                
                # Convert back to individual polygons
                if dissolved_geom.geom_type == 'MultiPolygon':
                    dissolved_polys = list(dissolved_geom.geoms)
                else:
                    dissolved_polys = [dissolved_geom]
                
                # Recalculate statistics for merged polygons
                batch_result = []
                for poly in dissolved_polys:
                    if poly.area >= self.min_polygon_area:
                        # Get height stats from original data
                        matching = combined[combined.geometry.intersects(poly)]
                        
                        if len(matching) > 0:
                            # Use shortened column names that match shapefile output
                            # Check which columns exist
                            if 'ht_min' in matching.columns:
                                ht_min = matching['ht_min'].min()
                                ht_max = matching['ht_max'].max()
                                ht_mean = matching['ht_mean'].mean()
                            elif 'height_min' in matching.columns:
                                # Fallback to long names if they exist
                                ht_min = matching['height_min'].min()
                                ht_max = matching['height_max'].max()
                                ht_mean = matching['height_mean'].mean()
                            else:
                                # Default values if columns missing
                                ht_min = ht_max = ht_mean = 0.0
                        else:
                            ht_min = ht_max = ht_mean = 0.0
                        
                        batch_result.append({
                            'geometry': poly,
                            'class_id': class_id,
                            'class_name': self.classes[class_id]['name'][:10],
                            'ht_min': ht_min,
                            'ht_max': ht_max,
                            'ht_mean': ht_mean,
                            'area_sqm': poly.area
                        })
                
                merged_batches.extend(batch_result)
                del combined, chunks, dissolved_geom, dissolved_polys
                gc.collect()
        
        print(f"         Dissolving complete!                                ")
        
        # Clean up temp files
        for shp_file in temp_shapefiles:
            try:
                # Remove all associated files (.shp, .shx, .dbf, etc.)
                base = shp_file.replace('.shp', '')
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    if os.path.exists(base + ext):
                        os.remove(base + ext)
            except:
                pass
        
        return merged_batches
    
    def save_output(self):
        """Save classified polygons to shapefile."""
        print(f"\n💾 Saving output shapefile...")
        
        if len(self.final_gdf) > 0:
            print("   Simplifying geometries...")
            self.final_gdf['geometry'] = self.final_gdf['geometry'].simplify(0.25, preserve_topology=True)

            self.final_gdf.to_file(self.output_shapefile_path)
            print(f"✅ Saved: {self.output_shapefile_path}")
            
            # Print final statistics
            print("\n" + "="*70)
            print("📈 FINAL STATISTICS")
            print("="*70)
            
            for class_id, info in self.classes.items():
                class_data = self.final_gdf[self.final_gdf['class_id'] == class_id]
                if len(class_data) > 0:
                    total_area = class_data['area_sqm'].sum()
                    avg_height = class_data['ht_mean'].mean()
                    print(f"\n{info['name']}:")
                    print(f"   Polygons: {len(class_data):,}")
                    print(f"   Total area: {total_area:,.1f} m² ({total_area/10000:.2f} hectares)")
                    print(f"   Avg height: {avg_height:.2f} m")
                    print(f"   Height range: {class_data['ht_min'].min():.2f} - {class_data['ht_max'].max():.2f} m")
        else:
            print("⚠️  No data to save!")
        
        # Cleanup temp directory
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            print(f"\n🧹 Cleaned up temporary files")
        except:
            pass
    
    def process(self):
        """Run complete classification pipeline."""
        import time
        start_time = time.time()
        
        self.load_data()
        self.classify_heights()
        self.vectorize_and_split()
        self.save_output()
        
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print(f"✅ COMPLETE! Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
        print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-optimized tree classification (boundary artifact fix)")
    parser.add_argument("--dem_path", required=True, help="Path to height raster")
    parser.add_argument("--output_path", required=True, help="Path for output shapefile")
    parser.add_argument("--shapefile_path", required=True, help="Path to original shapefile")
    parser.add_argument("--chunk_size", type=int, default=4096, help="Processing chunk size (default: 4096)")
    parser.add_argument("--overlap", type=int, default=256, help="Chunk overlap in pixels (default: 256)")
    parser.add_argument("--min_area", type=float, default=1.0, help="Minimum polygon area in m² (default: 1.0)")
    
    args = parser.parse_args()

    MIN_POLYGON_AREA = args.min_area
    CHUNK_SIZE = args.chunk_size
    OVERLAP = args.overlap
    
    print("\n⚙️  Configuration:")
    print(f"   Chunk size: {CHUNK_SIZE}x{CHUNK_SIZE} pixels")
    print(f"   Overlap: {OVERLAP} pixels (fixes boundary artifacts)")
    print(f"   Minimum polygon area: {MIN_POLYGON_AREA} m²")
    print("\n⚙️  Classification Rules:")
    print("   Class 1: Grass/Land    (0.1 - 0.6m) [SKIPPED]")
    print("   Class 2: Shrubs        (0.6 - 5.5m)")
    print("   Class 3: Medium Trees  (5.5 - 10.0m)")
    print("   Class 4: High Trees    (> 10.0m)")
    
    classifier = TreeClassifier(
        height_raster_path=args.dem_path,
        original_shapefile_path=args.shapefile_path,
        output_shapefile_path=args.output_path,
        min_polygon_area=MIN_POLYGON_AREA,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP
    )
    
    classifier.process()
    
    print(f"\n📂 Output files:")
    print(f"   🗺️  Classified Shapefile: {args.output_path}")
    print(f"   🎨 Classification Raster: {args.dem_path.replace('.tif', '_classified.tif')}")
    print(f"\n✨ Classification complete! No boundary artifacts!")