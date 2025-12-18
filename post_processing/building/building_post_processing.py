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
from rasterio.mask import mask
from scipy import ndimage
from shapely import affinity
import os
import warnings
import argparse
from erosion_pixels import BuildingSeparator
warnings.filterwarnings('ignore')

class BuildingPolygonExtractor:
    """
    Complete pipeline for converting building prediction raster to clean polygon shapefiles
    """
    
    def __init__(self, min_area_m2=50, simplify_tolerance=1.0, smooth_buffer_size=0.5):
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
        """Load and validate the input binary raster (1=building, 0=non-building)"""
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
                # Multi-value raster - assume building is the maximum value
                max_val = data.max()
                binary_mask = (data == max_val).astype(np.uint8)
                print(f"⚠️  Multi-value raster: treating {max_val} as building")
            
        print(f"Raster loaded successfully. Shape: {data.shape}, CRS: {crs}")
        print(f"Building pixels: {np.sum(binary_mask):,} / {binary_mask.size:,} ({np.sum(binary_mask)/binary_mask.size*100:.1f}%)")
        
        return binary_mask, transform, crs, profile
    
    def clean_building_mask(self, binary_mask, pixel_size_m=0.3):
        """
        Clean the binary mask using morphological operations optimized for buildings
        
        Args:
            binary_mask: Binary numpy array
            pixel_size_m: Pixel size in meters for area calculations
        """
        print("Cleaning building mask...")
        
        # Step 1: Close small gaps between building parts (fill 1-2 pixel gaps)
        # This helps connect building parts separated by thin black lines
        kernel_close = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Step 2: Remove very small noise pixels
        kernel_open = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
        # Step 3: Fill holes inside buildings
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        
        # Step 4: Remove small objects based on minimum area
        min_pixels = self.min_area_m2 / (pixel_size_m ** 2)  # Convert area to pixels
        print(f"Minimum pixels for building: {min_pixels:.1f}")
        
        # Use connected components to remove small objects
        cleaned = morphology.remove_small_objects(
            cleaned.astype(bool), 
            min_size=int(min_pixels)
        ).astype(np.uint8)
        
        # Step 5: Slight erosion followed by dilation to regularize building shapes
        kernel_reg = np.ones((2,2), np.uint8)
        cleaned = cv2.erode(cleaned, kernel_reg, iterations=1)
        cleaned = cv2.dilate(cleaned, kernel_reg, iterations=1)
        
        print(f"Mask cleaned. Building pixels: {np.sum(cleaned)} (reduction: {np.sum(binary_mask) - np.sum(cleaned)})")
        
        return cleaned
    
    def raster_to_polygons(self, cleaned_mask, transform, crs):
        """
        Convert binary raster to vector polygons using rasterio
        """
        print("Converting raster to polygons...")

        # Ensure binary mask
        cleaned_mask = (cleaned_mask > 0).astype(np.uint8)

        # Shift polygons half a pixel up-left
        corrected_transform = transform * rasterio.transform.Affine.translation(-1.8, -1.8)

        print(f"Original transform: {transform}")
        print(f"Corrected transform: {corrected_transform}")
        
        # Generate polygon shapes from raster (only for building areas = 1)
        shapes = rasterio.features.shapes(
            cleaned_mask,
            mask=cleaned_mask == 1,
            transform=corrected_transform,
            connectivity=4
        )
        
        polygons = []
        for geom, value in shapes:
            if value == 1:
                poly = shape(geom)
                if poly.is_valid and poly.area > 0:
                    poly_rotated = affinity.rotate(poly, -10, origin="centroid", use_radians=False)
                    polygons.append(poly)
        
        print(f"Generated {len(polygons)} raw polygons")
        return polygons

    
    def smooth_and_simplify_buildings(self, polygons, crs):
        """
        Smooth and simplify building polygons to create clean rectangular shapes
        """
        print("Smoothing and simplifying building polygons...")
        
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
                
                # For buildings, we want to maintain rectangular shapes
                # Use a minimal smoothing approach
                if self.smooth_buffer_size > 0:
                    # Very light smoothing to maintain building edges
                    smoothed = poly.buffer(self.smooth_buffer_size, join_style=2)  # Mitered joins
                    smoothed = smoothed.buffer(-self.smooth_buffer_size, join_style=2)
                    
                    # Handle MultiPolygon results
                    if smoothed.geom_type == 'MultiPolygon':
                        for sub_poly in smoothed.geoms:
                            if sub_poly.area > self.min_area_m2:
                                # Light simplification to maintain straight edges
                                simplified = sub_poly.simplify(self.simplify_tolerance, preserve_topology=True)
                                if simplified.is_valid and simplified.area > self.min_area_m2:
                                    processed_polygons.append(simplified)
                    else:
                        if smoothed.is_valid and smoothed.area > self.min_area_m2:
                            simplified = smoothed.simplify(self.simplify_tolerance, preserve_topology=True)
                            if simplified.is_valid and simplified.area > self.min_area_m2:
                                processed_polygons.append(simplified)
                else:
                    # Just simplify without smoothing
                    simplified = poly.simplify(self.simplify_tolerance, preserve_topology=True)
                    if simplified.is_valid and simplified.area > self.min_area_m2:
                        processed_polygons.append(simplified)
                        
            except Exception as e:
                print(f"Error processing polygon: {e}")
                continue
        
        print(f"Processed {len(processed_polygons)} building polygons")
        return processed_polygons
    
    def regularize_building_shapes(self, polygons):
        """
        Additional step to make building shapes more rectangular
        """
        print("Regularizing building shapes...")
        
        regularized_polygons = []
        
        for poly in polygons:
            try:
                # Get the minimum rotated rectangle (oriented bounding box)
                min_rect = poly.minimum_rotated_rectangle
                
                # Calculate overlap ratio
                overlap_ratio = poly.intersection(min_rect).area / poly.area
                
                # If the building is already quite rectangular (>80% overlap), use simplified version
                if overlap_ratio > 0.8:
                    # Use a light simplification to maintain the natural shape
                    regularized = poly.simplify(self.simplify_tolerance * 0.5, preserve_topology=True)
                    if regularized.is_valid and regularized.area > self.min_area_m2:
                        regularized_polygons.append(regularized)
                else:
                    # For irregular shapes, still use the original but simplified
                    simplified = poly.simplify(self.simplify_tolerance, preserve_topology=True)
                    if simplified.is_valid and simplified.area > self.min_area_m2:
                        regularized_polygons.append(simplified)
                        
            except Exception as e:
                print(f"Error regularizing polygon: {e}")
                # Keep original if regularization fails
                if poly.is_valid and poly.area > self.min_area_m2:
                    regularized_polygons.append(poly)
                continue
        
        print(f"Regularized {len(regularized_polygons)} building polygons")
        return regularized_polygons
    
    def create_geodataframe(self, polygons, crs):
        """
        Create GeoDataFrame with building polygon attributes
        """
        print("Creating GeoDataFrame...")
        
        # Calculate polygon attributes
        data = []
        for i, poly in enumerate(polygons):
            area = poly.area
            perimeter = poly.length
            
            # Convert area to square meters if in geographic coordinates
            if crs and crs.to_epsg() == 4326:
                area_m2 = area * 111000 * 111000  # Rough conversion
            else:
                area_m2 = area  # Already in square meters for projected coordinates
            
            # Calculate building metrics
            try:
                min_rect = poly.minimum_rotated_rectangle
                rectangularity = poly.area / min_rect.area if min_rect.area > 0 else 0
            except:
                rectangularity = 0
            
            data.append({
                'building_id': i + 1,
                'area_m2': round(area_m2, 2),
                'perimeter_m': round(perimeter, 2),
                'rectangularity': round(rectangularity, 3),
                'geometry': poly
            })
        
        gdf = gpd.GeoDataFrame(data, crs=crs)
        print(f"Created GeoDataFrame with {len(gdf)} building polygons")
        
        return gdf
    
    def process_raster_to_polygons(self, input_raster_path, output_shapefile_path, 
                                 regularize_shapes=True, eroded_shape = None):
        """
        Complete pipeline: raster → clean building polygons → shapefile
        
        Args:
            input_raster_path: Path to input raster (TIFF)
            output_shapefile_path: Path for output shapefile
            regularize_shapes: Whether to regularize building shapes
        """
        print(f"\n{'='*60}")
        print("BUILDING POLYGON EXTRACTION PIPELINE")
        print(f"{'='*60}")
        
        try:
            # Step 1: Load raster
            binary_mask, transform, crs, profile = self.load_raster(input_raster_path)
            
            # Step 2: Clean binary mask for buildings

            if eroded_shape is not None:
                cleaned_mask = self.clean_building_mask(eroded_shape)

            else :
                cleaned_mask = self.clean_building_mask(binary_mask)
            
            # Step 3: Convert to polygons
            raw_polygons = self.raster_to_polygons(cleaned_mask, transform, crs)
            
            if not raw_polygons:
                print("No polygons generated! Check your raster data and threshold.")
                return None
            
            # Step 4: Smooth and simplify for buildings
            processed_polygons = self.smooth_and_simplify_buildings(raw_polygons, crs)
            
            if not processed_polygons:
                print("No polygons after processing! Try adjusting parameters.")
                return None
            
            # Step 5: Regularize building shapes (optional)
            if regularize_shapes:
                final_polygons = self.regularize_building_shapes(processed_polygons)
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
            print(f"Total building polygons: {len(gdf)}")
            print(f"Total building area: {gdf['area_m2'].sum():.2f} square meters")
            print(f"Average building size: {gdf['area_m2'].mean():.2f} square meters")
            print(f"Largest building: {gdf['area_m2'].max():.2f} square meters")
            print(f"Smallest building: {gdf['area_m2'].min():.2f} square meters")
            print(f"Average rectangularity: {gdf['rectangularity'].mean():.3f}")
            print(f"Output saved to: {output_shapefile_path}")
            
            return gdf
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            return None
        
    def filter_polygons_by_height(self, shapefile_path_gdf, dsm_path, output_path, height_threshold=22, percentage_threshold=0.60):

        # Load DSM raster
        raster = rasterio.open(dsm_path)

        # Load polygons + convert to raster CRS
        gdf = shapefile_path_gdf
        gdf = gdf.to_crs(raster.crs)

        valid_rows = []
        removed_polygons = []

        for idx, row in gdf.iterrows():
            geom = [row.geometry]

            try:
                # Mask DSM with polygon
                out_image, out_transform = mask(raster, geom, crop=True)

                # Flatten DSM values
                data = out_image[0].flatten()

                # Remove nodata
                data = data[data != raster.nodata]

                if len(data) == 0:
                    removed_polygons.append(idx)
                    continue

                # Count pixels lower than threshold
                low_pixels = np.sum(data < height_threshold)
                total_pixels = len(data)

                low_percentage = low_pixels / total_pixels

                # Decide whether to keep polygon
                if low_percentage >= percentage_threshold:
                    removed_polygons.append(idx)
                else:
                    valid_rows.append(row)

            except Exception as e:
                removed_polygons.append(idx)

        # Convert list of rows → GeoDataFrame (CRITICALLY FIXED)
        if len(valid_rows) == 0:
            print("No polygons kept!")
            return None

        cleaned_gdf = gpd.GeoDataFrame(valid_rows, geometry="geometry", crs=gdf.crs)

        # Save output shapefile
        cleaned_gdf.to_file(output_path)

        print("Done!")
        print(f"Total polygons: {len(gdf)}")
        print(f"Removed polygons: {len(removed_polygons)}")
        print(f"Remaining polygons: {len(cleaned_gdf)}")

        return cleaned_gdf


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract building polygons with straight lines')
    parser.add_argument('--predicted_tiff', required=True, help='Input binary TIFF file')
    parser.add_argument('--output_path', required=True, help='Output directory')
    # parser.add_argument('--dsm_path', required=True, help='DSM FILE PATH')

    parser.add_argument('--apply_erosion',default=True, help='Output directory')
    parser.add_argument('--erosion_pixels', type=int, default=10, 
                       help='Erosion amount in pixels (recommended: 5-15, default: 10)')
    parser.add_argument('--kernel_shape', choices=['square', 'circle', 'cross', 'diamond'],
                       default='square', help='Erosion kernel shape (default: square)')
    parser.add_argument('--method', choices=['smart', 'gentle', 'opening'],
                       default='smart', help='Erosion method (default: smart)')
    
    args = parser.parse_args()

    # Convert to absolute paths
    input_raster = os.path.abspath(args.predicted_tiff)
    output_base_path = os.path.abspath(args.output_path)


    file_name = os.path.basename(input_raster)   # Mandadam.tif
    village_name = os.path.splitext(file_name)[0]         # Mandadam

    # Create output folder: /workspace/output/building_predictions/Mandadam
    output_folder = os.path.join(output_base_path, village_name)
    os.makedirs(output_folder, exist_ok=True)

    # Output shapefile path
    output_shapefile_path = os.path.join(output_folder, f"{village_name}.shp")


    extractor = BuildingPolygonExtractor(
        min_area_m2=2,      # Minimum building size: 50 m²
        simplify_tolerance=1.0,  # Light simplification to maintain building edges
        smooth_buffer_size=0.5   # Minimal smoothing for buildings
    )

    result_gdf = extractor.process_raster_to_polygons(
        input_raster_path=input_raster,
        eroded_shape = None,
        output_shapefile_path=output_shapefile_path,
        regularize_shapes=True
    )
    
    # -----------------------------
    # RUN
    # -----------------------------
    # cleaned = extractor.filter_polygons_by_height(
    #     shapefile_path_gdf=result_gdf,
    #     dsm_path=args.dsm_path,
    #     output_path=output_shapefile_path,
    #     height_threshold=22,
    #     percentage_threshold=0.60
    # )

    if cleaned is not None:
        print("\nProcessing completed successfully!")
        print(f"Buildings extracted: {len(result_gdf)}")
        
        # Optional: Preview the results
        print("\nFirst 5 buildings:")
        print(result_gdf.head()[['building_id', 'area_m2', 'rectangularity']])
    else:
        print("Processing failed. Check your input data and parameters.")






######################################## Old Code ########################################










# import numpy as np
# import rasterio
# import rasterio.features
# from rasterio.crs import CRS
# import geopandas as gpd
# from shapely.geometry import shape, Polygon
# from shapely.ops import unary_union
# import cv2
# from skimage import morphology, measure
# from skimage.filters import gaussian
# from scipy import ndimage
# from shapely import affinity
# import os
# import warnings
# import argparse
# from erosion_pixels import BuildingSeparator
# warnings.filterwarnings('ignore')

# class BuildingPolygonExtractor:
#     """
#     Complete pipeline for converting building prediction raster to clean polygon shapefiles
#     """
    
#     def __init__(self, min_area_m2=50, simplify_tolerance=1.0, smooth_buffer_size=0.5):
#         """
#         Initialize the pipeline with processing parameters
        
#         Args:
#             min_area_m2 (float): Minimum polygon area in square meters
#             simplify_tolerance (float): Douglas-Peucker simplification tolerance
#             smooth_buffer_size (float): Buffer size for polygon smoothing
#         """
#         self.min_area_m2 = min_area_m2
#         self.simplify_tolerance = simplify_tolerance
#         self.smooth_buffer_size = smooth_buffer_size
        
#     def load_raster(self, raster_path):
#         """Load and validate the input binary raster (1=building, 0=non-building)"""
#         print(f"Loading binary raster: {raster_path}")
        
#         with rasterio.open(raster_path) as src:
#             # Read the raster data
#             data = src.read(1)  # Assuming single band
#             transform = src.transform
#             crs = src.crs
#             profile = src.profile
            
#             # Handle binary mask - ensure values are exactly 0 and 1
#             unique_values = np.unique(data)
#             print(f"Unique values in raster: {unique_values}")
            
#             # Convert to proper binary mask
#             if len(unique_values) == 2 and set(unique_values) == {0, 1}:
#                 # Perfect binary mask
#                 binary_mask = data.astype(np.uint8)
#                 print("✅ Perfect binary mask detected (0s and 1s)")
#             elif len(unique_values) == 2:
#                 # Binary but different values - convert to 0,1
#                 min_val, max_val = unique_values
#                 binary_mask = ((data == max_val) * 1).astype(np.uint8)
#                 print(f"✅ Binary mask converted: {min_val}→0, {max_val}→1")
#             elif data.max() <= 1.0 and data.min() >= 0.0:
#                 # Float binary mask (0.0, 1.0)
#                 binary_mask = (data > 0.5).astype(np.uint8)
#                 print("✅ Float binary mask converted to integer")
#             else:
#                 # Multi-value raster - assume building is the maximum value
#                 max_val = data.max()
#                 binary_mask = (data == max_val).astype(np.uint8)
#                 print(f"⚠️  Multi-value raster: treating {max_val} as building")
            
#         print(f"Raster loaded successfully. Shape: {data.shape}, CRS: {crs}")
#         print(f"Building pixels: {np.sum(binary_mask):,} / {binary_mask.size:,} ({np.sum(binary_mask)/binary_mask.size*100:.1f}%)")
        
#         return binary_mask, transform, crs, profile
    
#     def clean_building_mask(self, binary_mask, pixel_size_m=0.3):
#         """
#         Clean the binary mask using morphological operations optimized for buildings
        
#         Args:
#             binary_mask: Binary numpy array
#             pixel_size_m: Pixel size in meters for area calculations
#         """
#         print("Cleaning building mask...")
        
#         # Step 1: Close small gaps between building parts (fill 1-2 pixel gaps)
#         # This helps connect building parts separated by thin black lines
#         kernel_close = np.ones((3,3), np.uint8)
#         cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
        
#         # Step 2: Remove very small noise pixels
#         kernel_open = np.ones((2,2), np.uint8)
#         cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        
#         # Step 3: Fill holes inside buildings
#         cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        
#         # Step 4: Remove small objects based on minimum area
#         min_pixels = self.min_area_m2 / (pixel_size_m ** 2)  # Convert area to pixels
#         print(f"Minimum pixels for building: {min_pixels:.1f}")
        
#         # Use connected components to remove small objects
#         cleaned = morphology.remove_small_objects(
#             cleaned.astype(bool), 
#             min_size=int(min_pixels)
#         ).astype(np.uint8)
        
#         # Step 5: Slight erosion followed by dilation to regularize building shapes
#         kernel_reg = np.ones((2,2), np.uint8)
#         cleaned = cv2.erode(cleaned, kernel_reg, iterations=1)
#         cleaned = cv2.dilate(cleaned, kernel_reg, iterations=1)
        
#         print(f"Mask cleaned. Building pixels: {np.sum(cleaned)} (reduction: {np.sum(binary_mask) - np.sum(cleaned)})")
        
#         return cleaned
    
#     def raster_to_polygons(self, cleaned_mask, transform, crs):
#         """
#         Convert binary raster to vector polygons using rasterio
#         """
#         print("Converting raster to polygons...")

#         # Ensure binary mask
#         cleaned_mask = (cleaned_mask > 0).astype(np.uint8)

#         # Shift polygons half a pixel up-left
#         corrected_transform = transform * rasterio.transform.Affine.translation(-1.8, -1.8)

#         print(f"Original transform: {transform}")
#         print(f"Corrected transform: {corrected_transform}")
        
#         # Generate polygon shapes from raster (only for building areas = 1)
#         shapes = rasterio.features.shapes(
#             cleaned_mask,
#             mask=cleaned_mask == 1,
#             transform=corrected_transform,
#             connectivity=4
#         )
        
#         polygons = []
#         for geom, value in shapes:
#             if value == 1:
#                 poly = shape(geom)
#                 if poly.is_valid and poly.area > 0:
#                     poly_rotated = affinity.rotate(poly, -10, origin="centroid", use_radians=False)
#                     polygons.append(poly)
        
#         print(f"Generated {len(polygons)} raw polygons")
#         return polygons

    
#     def smooth_and_simplify_buildings(self, polygons, crs):
#         """
#         Smooth and simplify building polygons to create clean rectangular shapes
#         """
#         print("Smoothing and simplifying building polygons...")
        
#         processed_polygons = []
        
#         for poly in polygons:
#             try:
#                 # Skip invalid or tiny polygons
#                 if not poly.is_valid or poly.area == 0:
#                     continue
                
#                 # Convert area threshold to map units
#                 if crs and crs.to_epsg() == 4326:  # Geographic coordinates
#                     # Rough conversion: 1 degree ≈ 111000 meters at equator
#                     min_area_deg = self.min_area_m2 / (111000 ** 2)
#                     if poly.area < min_area_deg:
#                         continue
#                 else:  # Projected coordinates
#                     if poly.area < self.min_area_m2:
#                         continue
                
#                 # For buildings, we want to maintain rectangular shapes
#                 # Use a minimal smoothing approach
#                 if self.smooth_buffer_size > 0:
#                     # Very light smoothing to maintain building edges
#                     smoothed = poly.buffer(self.smooth_buffer_size, join_style=2)  # Mitered joins
#                     smoothed = smoothed.buffer(-self.smooth_buffer_size, join_style=2)
                    
#                     # Handle MultiPolygon results
#                     if smoothed.geom_type == 'MultiPolygon':
#                         for sub_poly in smoothed.geoms:
#                             if sub_poly.area > self.min_area_m2:
#                                 # Light simplification to maintain straight edges
#                                 simplified = sub_poly.simplify(self.simplify_tolerance, preserve_topology=True)
#                                 if simplified.is_valid and simplified.area > self.min_area_m2:
#                                     processed_polygons.append(simplified)
#                     else:
#                         if smoothed.is_valid and smoothed.area > self.min_area_m2:
#                             simplified = smoothed.simplify(self.simplify_tolerance, preserve_topology=True)
#                             if simplified.is_valid and simplified.area > self.min_area_m2:
#                                 processed_polygons.append(simplified)
#                 else:
#                     # Just simplify without smoothing
#                     simplified = poly.simplify(self.simplify_tolerance, preserve_topology=True)
#                     if simplified.is_valid and simplified.area > self.min_area_m2:
#                         processed_polygons.append(simplified)
                        
#             except Exception as e:
#                 print(f"Error processing polygon: {e}")
#                 continue
        
#         print(f"Processed {len(processed_polygons)} building polygons")
#         return processed_polygons
    
#     def regularize_building_shapes(self, polygons):
#         """
#         Additional step to make building shapes more rectangular
#         """
#         print("Regularizing building shapes...")
        
#         regularized_polygons = []
        
#         for poly in polygons:
#             try:
#                 # Get the minimum rotated rectangle (oriented bounding box)
#                 min_rect = poly.minimum_rotated_rectangle
                
#                 # Calculate overlap ratio
#                 overlap_ratio = poly.intersection(min_rect).area / poly.area
                
#                 # If the building is already quite rectangular (>80% overlap), use simplified version
#                 if overlap_ratio > 0.8:
#                     # Use a light simplification to maintain the natural shape
#                     regularized = poly.simplify(self.simplify_tolerance * 0.5, preserve_topology=True)
#                     if regularized.is_valid and regularized.area > self.min_area_m2:
#                         regularized_polygons.append(regularized)
#                 else:
#                     # For irregular shapes, still use the original but simplified
#                     simplified = poly.simplify(self.simplify_tolerance, preserve_topology=True)
#                     if simplified.is_valid and simplified.area > self.min_area_m2:
#                         regularized_polygons.append(simplified)
                        
#             except Exception as e:
#                 print(f"Error regularizing polygon: {e}")
#                 # Keep original if regularization fails
#                 if poly.is_valid and poly.area > self.min_area_m2:
#                     regularized_polygons.append(poly)
#                 continue
        
#         print(f"Regularized {len(regularized_polygons)} building polygons")
#         return regularized_polygons
    
#     def create_geodataframe(self, polygons, crs):
#         """
#         Create GeoDataFrame with building polygon attributes
#         """
#         print("Creating GeoDataFrame...")
        
#         # Calculate polygon attributes
#         data = []
#         for i, poly in enumerate(polygons):
#             area = poly.area
#             perimeter = poly.length
            
#             # Convert area to square meters if in geographic coordinates
#             if crs and crs.to_epsg() == 4326:
#                 area_m2 = area * 111000 * 111000  # Rough conversion
#             else:
#                 area_m2 = area  # Already in square meters for projected coordinates
            
#             # Calculate building metrics
#             try:
#                 min_rect = poly.minimum_rotated_rectangle
#                 rectangularity = poly.area / min_rect.area if min_rect.area > 0 else 0
#             except:
#                 rectangularity = 0
            
#             data.append({
#                 'building_id': i + 1,
#                 'area_m2': round(area_m2, 2),
#                 'perimeter_m': round(perimeter, 2),
#                 'rectangularity': round(rectangularity, 3),
#                 'geometry': poly
#             })
        
#         gdf = gpd.GeoDataFrame(data, crs=crs)
#         print(f"Created GeoDataFrame with {len(gdf)} building polygons")
        
#         return gdf
    
#     def process_raster_to_polygons(self, input_raster_path, output_shapefile_path, 
#                                  regularize_shapes=True, eroded_shape = None):
#         """
#         Complete pipeline: raster → clean building polygons → shapefile
        
#         Args:
#             input_raster_path: Path to input raster (TIFF)
#             output_shapefile_path: Path for output shapefile
#             regularize_shapes: Whether to regularize building shapes
#         """
#         print(f"\n{'='*60}")
#         print("BUILDING POLYGON EXTRACTION PIPELINE")
#         print(f"{'='*60}")
        
#         try:
#             # Step 1: Load raster
#             binary_mask, transform, crs, profile = self.load_raster(input_raster_path)
            
#             # Step 2: Clean binary mask for buildings

#             if eroded_shape is not None:
#                 cleaned_mask = self.clean_building_mask(eroded_shape)

#             else :
#                 cleaned_mask = self.clean_building_mask(binary_mask)
            
#             # Step 3: Convert to polygons
#             raw_polygons = self.raster_to_polygons(cleaned_mask, transform, crs)
            
#             if not raw_polygons:
#                 print("No polygons generated! Check your raster data and threshold.")
#                 return None
            
#             # Step 4: Smooth and simplify for buildings
#             processed_polygons = self.smooth_and_simplify_buildings(raw_polygons, crs)
            
#             if not processed_polygons:
#                 print("No polygons after processing! Try adjusting parameters.")
#                 return None
            
#             # Step 5: Regularize building shapes (optional)
#             if regularize_shapes:
#                 final_polygons = self.regularize_building_shapes(processed_polygons)
#             else:
#                 final_polygons = processed_polygons
            
#             # Step 6: Create GeoDataFrame
#             gdf = self.create_geodataframe(final_polygons, crs)
            
#             # Step 7: Save to shapefile
#             print(f"Saving to shapefile: {output_shapefile_path}")
#             os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)
#             gdf.to_file(output_shapefile_path)
            
#             # Print summary statistics
#             print(f"\n{'='*60}")
#             print("PROCESSING SUMMARY")
#             print(f"{'='*60}")
#             print(f"Total building polygons: {len(gdf)}")
#             print(f"Total building area: {gdf['area_m2'].sum():.2f} square meters")
#             print(f"Average building size: {gdf['area_m2'].mean():.2f} square meters")
#             print(f"Largest building: {gdf['area_m2'].max():.2f} square meters")
#             print(f"Smallest building: {gdf['area_m2'].min():.2f} square meters")
#             print(f"Average rectangularity: {gdf['rectangularity'].mean():.3f}")
#             print(f"Output saved to: {output_shapefile_path}")
            
#             return gdf
            
#         except Exception as e:
#             print(f"Error in processing pipeline: {e}")
#             return None


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description='Extract building polygons with straight lines')
#     parser.add_argument('--predicted_tiff', required=True, help='Input binary TIFF file')
#     parser.add_argument('--output_path', required=True, help='Output directory')
#     parser.add_argument('--apply_erosion',default=True, help='Output directory')
#     parser.add_argument('--erosion_pixels', type=int, default=10, 
#                        help='Erosion amount in pixels (recommended: 5-15, default: 10)')
#     parser.add_argument('--kernel_shape', choices=['square', 'circle', 'cross', 'diamond'],
#                        default='square', help='Erosion kernel shape (default: square)')
#     parser.add_argument('--method', choices=['smart', 'gentle', 'opening'],
#                        default='smart', help='Erosion method (default: smart)')
    
#     args = parser.parse_args()

#     # Convert to absolute paths
#     input_raster = os.path.abspath(args.predicted_tiff)
#     output_base_path = os.path.abspath(args.output_path)


#     file_name = os.path.basename(input_raster)   # Mandadam.tif
#     village_name = os.path.splitext(file_name)[0]         # Mandadam

#     # Create output folder: /workspace/output/building_predictions/Mandadam
#     output_folder = os.path.join(output_base_path, village_name)
#     os.makedirs(output_folder, exist_ok=True)

#     # Output shapefile path
#     output_shapefile_path = os.path.join(output_folder, f"{village_name}.shp")


#     # separator = BuildingSeparator()
    
#     # result = separator.process_building_separation(
#     #     input_raster_path=args.predicted_tiff,
#     #     output_raster_path=args.output_path,
#     #     erosion_pixels=args.erosion_pixels,
#     #     kernel_shape=args.kernel_shape,
#     #     method=args.method,
#     #     save=False
#     # )

#     extractor = BuildingPolygonExtractor(
#         min_area_m2=2,      # Minimum building size: 50 m²
#         simplify_tolerance=1.0,  # Light simplification to maintain building edges
#         smooth_buffer_size=0.5   # Minimal smoothing for buildings
#     )

#     result_gdf = extractor.process_raster_to_polygons(
#         input_raster_path=input_raster,
#         eroded_shape = None,
#         output_shapefile_path=output_shapefile_path,
#         regularize_shapes=True
#     )
    
#     if result_gdf is not None:
#         print("\nProcessing completed successfully!")
#         print(f"Buildings extracted: {len(result_gdf)}")
        
#         # Optional: Preview the results
#         print("\nFirst 5 buildings:")
#         print(result_gdf.head()[['building_id', 'area_m2', 'rectangularity']])
#     else:
#         print("Processing failed. Check your input data and parameters.")
