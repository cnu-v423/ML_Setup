import numpy as np
import rasterio
import rasterio.features
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
import cv2
from skimage import morphology, measure
from skimage.filters import gaussian
from scipy import ndimage
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

class CultivationPolygonExtractor:
    """
    Complete pipeline for converting cultivation prediction raster to clean polygon shapefiles
    """
    
    def __init__(self, min_area_m2=100, simplify_tolerance=2.0, smooth_buffer_size=1.0):
        """
        Initialize the pipeline with processing parameters
        
        Args:
            min_area_m2 (float): Minimum polygon area in square meters
            simplify_tolerance (float): Douglas-Peucker simplification tolerance
            smooth_buffer_size (float): Buffer size for polygon smoothing
        """
        self.min_area_m2 = min_area_m2
        self.simplify_tolerance = simplify_tolerance
        self.smooth_buffer_size = smooth_buffer_size
        
    def load_raster(self, raster_path):
        """Load and validate the input binary raster (1=cultivation, 0=non-cultivation)"""
        print(f"Loading binary raster: {raster_path}")
        
        with rasterio.open(raster_path) as src:
            # Read the raster data
            data = src.read(1)  # Assuming single band
            transform = src.transform
            crs = src.crs
            profile = src.profile
            
            # Handle binary mask - ensure values are exactly 0 and 1
            unique_values = np.unique(data)
            print(f"Unique values in raster: {unique_values}")
            
            # Convert to proper binary mask
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                # Perfect binary mask
                binary_mask = data.astype(np.uint8)
                print("✅ Perfect binary mask detected (0s and 1s)")
            elif len(unique_values) == 2:
                # Binary but different values - convert to 0,1
                min_val, max_val = unique_values
                binary_mask = ((data == max_val) * 1).astype(np.uint8)
                print(f"✅ Binary mask converted: {min_val}→0, {max_val}→1")
            elif data.max() <= 1.0 and data.min() >= 0.0:
                # Float binary mask (0.0, 1.0)
                binary_mask = (data > 0.5).astype(np.uint8)
                print("✅ Float binary mask converted to integer")
            else:
                # Multi-value raster - assume cultivation is the maximum value
                max_val = data.max()
                binary_mask = (data == max_val).astype(np.uint8)
                print(f"⚠️  Multi-value raster: treating {max_val} as cultivation")
            
        print(f"Raster loaded successfully. Shape: {data.shape}, CRS: {crs}")
        print(f"Cultivation pixels: {np.sum(binary_mask):,} / {binary_mask.size:,} ({np.sum(binary_mask)/binary_mask.size*100:.1f}%)")
        
        return binary_mask, transform, crs, profile
    
    def clean_binary_mask(self, binary_mask, pixel_size_m=10):
        """
        Clean the binary mask using morphological operations
        
        Args:
            binary_mask: Binary numpy array
            pixel_size_m: Pixel size in meters for area calculations
        """
        print("Cleaning binary mask...")
        
        # Step 1: Remove salt and pepper noise
        kernel_small = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Step 2: Fill small holes
        kernel_medium = np.ones((5,5), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # Step 3: Remove small objects using connected components
        min_pixels = self.min_area_m2 / (pixel_size_m ** 2)  # Convert area to pixels
        cleaned = morphology.remove_small_objects(
            cleaned.astype(bool), 
            min_size=int(min_pixels)
        ).astype(np.uint8)
        
        # Step 4: Fill holes in remaining objects
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        
        # Step 5: Light smoothing with Gaussian filter
        cleaned_smooth = gaussian(cleaned.astype(float), sigma=0.8)
        cleaned = (cleaned_smooth > 0.5).astype(np.uint8)
        
        print(f"Mask cleaned. Cultivation pixels: {np.sum(cleaned)} (reduction: {np.sum(binary_mask) - np.sum(cleaned)})")
        
        return cleaned
    
    def raster_to_polygons(self, cleaned_mask, transform, crs):
        """
        Convert binary raster to vector polygons using rasterio
        """
        print("Converting raster to polygons...")
        
        # Generate polygon shapes from raster (only for cultivation areas = 1)
        shapes = rasterio.features.shapes(
            cleaned_mask.astype(np.uint8),
            mask=cleaned_mask == 1,  # Only process pixels with value 1
            transform=transform,
            connectivity=8  # 8-connected pixels
        )
        
        # Convert to shapely geometries
        polygons = []
        for geom, value in shapes:
            if value == 1:  # Only cultivation areas
                poly = shape(geom)
                if poly.is_valid and poly.area > 0:
                    polygons.append(poly)
        
        print(f"Generated {len(polygons)} raw polygons")
        return polygons
    
    def smooth_and_simplify_polygons(self, polygons, crs):
        """
        Smooth and simplify polygons for cleaner boundaries
        """
        print("Smoothing and simplifying polygons...")
        
        processed_polygons = []
        
        for poly in polygons:
            try:
                # Skip invalid or tiny polygons
                if not poly.is_valid or poly.area == 0:
                    continue
                
                # Convert area threshold to map units
                if crs and crs.to_epsg() == 4326:  # Geographic coordinates
                    # Rough conversion: 1 degree ≈ 111000 meters at equator
                    min_area_deg = self.min_area_m2 / (111000 ** 2)
                    if poly.area < min_area_deg:
                        continue
                else:  # Projected coordinates
                    if poly.area < self.min_area_m2:
                        continue
                
                # Smooth polygon using buffer trick
                if self.smooth_buffer_size > 0:
                    # Positive buffer then negative buffer for smoothing
                    smoothed = poly.buffer(self.smooth_buffer_size).buffer(-self.smooth_buffer_size)
                    
                    # Handle MultiPolygon results
                    if smoothed.geom_type == 'MultiPolygon':
                        for sub_poly in smoothed.geoms:
                            if sub_poly.area > self.min_area_m2:
                                simplified = sub_poly.simplify(self.simplify_tolerance, preserve_topology=True)
                                if simplified.is_valid:
                                    processed_polygons.append(simplified)
                    else:
                        if smoothed.is_valid:
                            simplified = smoothed.simplify(self.simplify_tolerance, preserve_topology=True)
                            if simplified.is_valid:
                                processed_polygons.append(simplified)
                else:
                    # Just simplify without smoothing
                    simplified = poly.simplify(self.simplify_tolerance, preserve_topology=True)
                    if simplified.is_valid:
                        processed_polygons.append(simplified)
                        
            except Exception as e:
                print(f"Error processing polygon: {e}")
                continue
        
        print(f"Processed {len(processed_polygons)} polygons")
        return processed_polygons
    
    def merge_adjacent_polygons(self, polygons, merge_distance=5.0):
        """
        Merge polygons that are very close to each other
        """
        print("Merging adjacent polygons...")
        
        if not polygons:
            return polygons
        
        # Buffer polygons slightly to connect nearby ones
        buffered = [poly.buffer(merge_distance) for poly in polygons]
        
        # Find overlapping groups and merge them
        merged_groups = []
        processed = set()
        
        for i, poly in enumerate(buffered):
            if i in processed:
                continue
                
            # Find all polygons that intersect with this one
            group = [polygons[i]]  # Original unbuffered polygon
            processed.add(i)
            
            for j, other_poly in enumerate(buffered):
                if j != i and j not in processed and poly.intersects(other_poly):
                    group.append(polygons[j])  # Original unbuffered polygon
                    processed.add(j)
            
            # Merge the group
            if len(group) > 1:
                try:
                    merged = unary_union(group)
                    if merged.geom_type == 'MultiPolygon':
                        merged_groups.extend(list(merged.geoms))
                    else:
                        merged_groups.append(merged)
                except:
                    merged_groups.extend(group)  # Keep original if merge fails
            else:
                merged_groups.extend(group)
        
        print(f"Merged to {len(merged_groups)} polygons")
        return merged_groups
    
    def create_geodataframe(self, polygons, crs):
        """
        Create GeoDataFrame with polygon attributes
        """
        print("Creating GeoDataFrame...")
        
        # Calculate polygon attributes
        data = []
        for i, poly in enumerate(polygons):
            area = poly.area
            perimeter = poly.length
            
            # Convert area to hectares if in geographic coordinates
            if crs and crs.to_epsg() == 4326:
                area_ha = area * 111000 * 111000 / 10000  # Rough conversion
            else:
                area_ha = area / 10000  # Square meters to hectares
            
            data.append({
                'field_id': i + 1,
                'area_m2': area,
                'area_ha': round(area_ha, 4),
                'perimeter_m': perimeter,
                'geometry': poly
            })
        
        gdf = gpd.GeoDataFrame(data, crs=crs)
        print(f"Created GeoDataFrame with {len(gdf)} polygons")
        
        return gdf
    
    def process_raster_to_polygons(self, input_raster_path, output_shapefile_path, 
                                 merge_nearby=True, merge_distance=5.0):
        """
        Complete pipeline: raster → clean polygons → shapefile
        
        Args:
            input_raster_path: Path to input raster (TIFF)
            output_shapefile_path: Path for output shapefile
            merge_nearby: Whether to merge nearby polygons
            merge_distance: Distance threshold for merging (map units)
        """
        print(f"\n{'='*60}")
        print("CULTIVATION POLYGON EXTRACTION PIPELINE")
        print(f"{'='*60}")
        
        try:
            # Step 1: Load raster
            binary_mask, transform, crs, profile = self.load_raster(input_raster_path)
            
            # Step 2: Clean binary mask
            cleaned_mask = self.clean_binary_mask(binary_mask)
            
            # Step 3: Convert to polygons
            raw_polygons = self.raster_to_polygons(cleaned_mask, transform, crs)
            
            if not raw_polygons:
                print("No polygons generated! Check your raster data and threshold.")
                return None
            
            # Step 4: Smooth and simplify
            processed_polygons = self.smooth_and_simplify_polygons(raw_polygons, crs)
            
            if not processed_polygons:
                print("No polygons after processing! Try adjusting parameters.")
                return None
            
            # Step 5: Merge nearby polygons (optional)
            if merge_nearby:
                final_polygons = self.merge_adjacent_polygons(processed_polygons, merge_distance)
            else:
                final_polygons = processed_polygons
            
            # Step 6: Create GeoDataFrame
            gdf = self.create_geodataframe(final_polygons, crs)
            
            # Step 7: Save to shapefile
            print(f"Saving to shapefile: {output_shapefile_path}")
            os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)
            gdf.to_file(output_shapefile_path)
            
            # Print summary statistics
            print(f"\n{'='*60}")
            print("PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Total cultivation fields: {len(gdf)}")
            print(f"Total area: {gdf['area_ha'].sum():.2f} hectares")
            print(f"Average field size: {gdf['area_ha'].mean():.2f} hectares")
            print(f"Largest field: {gdf['area_ha'].max():.2f} hectares")
            print(f"Smallest field: {gdf['area_ha'].min():.2f} hectares")
            print(f"Output saved to: {output_shapefile_path}")
            
            return gdf
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize the extractor with parameters
    extractor = CultivationPolygonExtractor(
        min_area_m2=100,      # Minimum field size: 500 m²
        simplify_tolerance=2.0,  # Simplification level
        smooth_buffer_size=1.5   # Smoothing strength
    )

    parser = argparse.ArgumentParser(description='Extract building polygons with straight lines')
    parser.add_argument('--predicted_tiff', required=True, help='Input binary TIFF file')
    parser.add_argument('--output_path', required=True, help='Output directory')
    
    args = parser.parse_args()

    # input_raster = input("Enter predicted tiff file path: ").strip()
    # output_folder = input("Enter output shapefile folder path: ").strip()


    # Convert to absolute paths
    input_raster = os.path.abspath(args.predicted_tiff)
    output_folder = os.path.abspath(args.output_path)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Derive shapefile name from the TIFF name
    raster_name = os.path.splitext(os.path.basename(input_raster))[0]  # e.g. predictions_building_res0p3m
    output_shapefile = os.path.join(output_folder, f"{raster_name}.shp")

    result_gdf = extractor.process_raster_to_polygons(
        input_raster_path=input_raster,
        output_shapefile_path=output_shapefile,
        merge_nearby=True,
        merge_distance=10.0
    )
    
    if result_gdf is not None:
        print("\nProcessing completed successfully!")
        print(f"Fields extracted: {len(result_gdf)}")
        
        # Optional: Preview the results
        print("\nFirst 5 fields:")
        print(result_gdf.head()[['field_id', 'area_ha', 'perimeter_m']])
    else:
        print("Processing failed. Check your input data and parameters.")