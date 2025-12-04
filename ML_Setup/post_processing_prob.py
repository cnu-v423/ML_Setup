import numpy as np
import rasterio
import rasterio.features
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
import cv2
from skimage import morphology, measure, segmentation, filters
from skimage.filters import gaussian, median, rank
from skimage.morphology import disk, square, closing, opening, erosion, dilation
from scipy import ndimage
from scipy.spatial.distance import cdist
import os
import warnings
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
import skimage
warnings.filterwarnings('ignore')

class BinaryMaskPolygonExtractor:
    """
    Enhanced pipeline for converting binary cultivation masks to clean polygon shapefiles
    with advanced post-processing for accurate boundary detection
    """
    
    def __init__(self, min_area_m2=100, simplify_tolerance=2.0, 
                 smooth_buffer_size=1.0, edge_enhancement=True, boundary_refinement=True):
        """
        Initialize the pipeline with processing parameters
        
        Args:
            min_area_m2 (float): Minimum polygon area in square meters
            simplify_tolerance (float): Douglas-Peucker simplification tolerance
            smooth_buffer_size (float): Buffer size for polygon smoothing
            edge_enhancement (bool): Apply edge enhancement preprocessing
            boundary_refinement (bool): Apply advanced boundary refinement
        """
        self.min_area_m2 = min_area_m2
        self.simplify_tolerance = simplify_tolerance
        self.smooth_buffer_size = smooth_buffer_size
        self.edge_enhancement = edge_enhancement
        self.boundary_refinement = boundary_refinement
        
    def load_raster(self, raster_path):
        """Load and validate the input binary mask raster"""
        print(f"Loading binary mask raster: {raster_path}")
        
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs
            profile = src.profile
            
            # Handle NoData values
            if src.nodata is not None:
                data = np.where(data == src.nodata, 0, data)
            
            # Convert to binary (0 or 1)
            data = (data > 0).astype(np.uint8)
            
        print(f"Binary mask loaded. Shape: {data.shape}, Cultivation pixels: {np.sum(data):,}")
        return data, transform, crs, profile
    
    def enhance_mask_edges(self, binary_data):
        """
        Enhance edges in binary mask for better boundary detection
        """
        if not self.edge_enhancement:
            return binary_data
            
        print("Enhancing mask edges for better boundary detection...")
        
        # Apply median filter to reduce noise while preserving edges
        enhanced = median(binary_data.astype(np.float32), disk(1))
        
        # Convert back to binary
        enhanced = (enhanced > 0.5).astype(np.uint8)
        
        print("Edge enhancement complete")
        return enhanced
    
    def morphological_refinement(self, binary_mask, pixel_size_m=0.3):
        """
        Advanced morphological operations for cleaner shapes - MUCH more conservative
        """
        print("Applying conservative morphological refinement...")
        
        # Convert pixel area threshold - be very conservative
        min_pixels = max(4, int(self.min_area_m2 / (pixel_size_m ** 2)))
        print(f"Minimum pixels for {self.min_area_m2}m² at {pixel_size_m}m resolution: {min_pixels}")
        
        cleaned = binary_mask.copy()
        
        # Step 1: VERY light noise removal - only 1 pixel kernel
        kernel_tiny = disk(1)
        cleaned = opening(cleaned, kernel_tiny)
        print(f"After opening: {np.sum(cleaned):,} pixels")
        
        # Step 2: Fill very small gaps only
        kernel_small = disk(2)
        cleaned = closing(cleaned, kernel_small)
        print(f"After closing: {np.sum(cleaned):,} pixels")
        
        # Step 3: Remove only very small objects in first pass
        initial_objects = np.sum(cleaned > 0)
        cleaned_bool = morphology.remove_small_objects(
            cleaned.astype(bool), 
            min_size=max(1, min_pixels//8)  # Very lenient first pass
        )
        cleaned = cleaned_bool.astype(np.uint8)
        print(f"After small object removal (pass 1): {np.sum(cleaned):,} pixels")
        
        # Step 4: Very light boundary smoothing - minimal erosion/dilation
        if self.boundary_refinement:
            kernel_smooth = disk(1)  # Very small kernel
            temp = erosion(cleaned, kernel_smooth)
            temp = dilation(temp, kernel_smooth)
            # Only apply if we don't lose too many pixels
            if np.sum(temp) > np.sum(cleaned) * 0.8:  # Keep at least 80%
                cleaned = temp
            print(f"After boundary smoothing: {np.sum(cleaned):,} pixels")
        
        # Step 5: Fill holes
        cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)
        print(f"After hole filling: {np.sum(cleaned):,} pixels")
        
        # Step 6: Final object removal - still conservative
        cleaned_bool = morphology.remove_small_objects(
            cleaned.astype(bool), 
            min_size=max(1, min_pixels//4)  # Less strict than original
        )
        cleaned = cleaned_bool.astype(np.uint8)
        print(f"After final object removal: {np.sum(cleaned):,} pixels")
        
        print(f"Morphological refinement complete")
        return cleaned
    
    def watershed_segmentation_refinement(self, binary_mask):
        """
        Use watershed segmentation to separate merged fields - more conservative
        """
        print("Applying watershed segmentation for field separation...")
        
        # Distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima as seeds - more permissive parameters
        coordinates = skimage.feature.peak_local_max(
            distance, 
            min_distance=10,  # Reduced from 20
            threshold_abs=3   # Reduced from 5
        )
        
        if len(coordinates) == 0:
            print("No watershed seeds found, skipping watershed segmentation")
            return None
            
        local_maxima = np.zeros_like(distance, dtype=bool)
        local_maxima[tuple(coordinates.T)] = True
        markers = measure.label(local_maxima)
        
        # Use inverted distance as elevation
        elevation = -distance
        elevation = np.where(binary_mask == 0, 0, elevation)  # Set background to 0
        
        # Apply watershed
        watershed_result = segmentation.watershed(elevation, markers, mask=binary_mask)
        
        print(f"Watershed segmentation identified {len(np.unique(watershed_result))-1} regions")
        return watershed_result
    
    def raster_to_polygons_advanced(self, cleaned_mask, transform, crs, watershed_result=None):
        """
        Enhanced polygon generation with multiple strategies
        """
        print("Converting raster to polygons with advanced processing...")
        
        polygons = []
        
        if watershed_result is not None:
            # Process each watershed region separately
            unique_regions = np.unique(watershed_result)
            unique_regions = unique_regions[unique_regions > 0]  # Skip background (0)
            
            print(f"Processing {len(unique_regions)} watershed regions...")
            
            for region_id in unique_regions:
                region_mask = (watershed_result == region_id).astype(np.uint8)
                
                # Generate shapes for this region
                try:
                    shapes = rasterio.features.shapes(
                        region_mask,
                        mask=region_mask == 1,
                        transform=transform,
                        connectivity=8
                    )
                    
                    for geom, value in shapes:
                        if value == 1:
                            poly = shape(geom)
                            if poly.is_valid and poly.area > 0:
                                polygons.append(poly)
                except Exception as e:
                    print(f"Error processing watershed region {region_id}: {e}")
                    continue
        else:
            # Standard approach
            print("Using standard polygon extraction...")
            try:
                shapes = rasterio.features.shapes(
                    cleaned_mask,
                    mask=cleaned_mask == 1,
                    transform=transform,
                    connectivity=8
                )
                
                for geom, value in shapes:
                    if value == 1:
                        poly = shape(geom)
                        if poly.is_valid and poly.area > 0:
                            polygons.append(poly)
            except Exception as e:
                print(f"Error in standard polygon extraction: {e}")
                return []
        
        print(f"Generated {len(polygons)} raw polygons")
        return polygons
    
    def conservative_polygon_smoothing(self, polygons, crs):
        """
        Much more conservative polygon smoothing to prevent polygon loss
        """
        print("Applying conservative polygon smoothing...")
        
        processed_polygons = []
        lost_polygons = 0
        
        for i, poly in enumerate(polygons):
            try:
                if not poly.is_valid:
                    # Try to fix invalid polygons
                    poly = poly.buffer(0)
                    if not poly.is_valid:
                        continue
                
                if poly.area == 0:
                    continue
                
                # Calculate area threshold based on CRS
                if crs and hasattr(crs, 'to_epsg') and crs.to_epsg() == 4326:
                    # For geographic coordinates (degrees)
                    min_area_deg = self.min_area_m2 / (111000 ** 2)  # Rough conversion
                    area_threshold = min_area_deg
                else:
                    # For projected coordinates (meters)
                    area_threshold = self.min_area_m2
                
                # Area filtering - be more lenient
                if poly.area < area_threshold * 0.5:  # Use half the threshold
                    lost_polygons += 1
                    continue
                
                # Start with original polygon
                smoothed = poly
                
                if self.boundary_refinement and poly.area > area_threshold * 2:  # Only smooth larger polygons
                    original_area = poly.area
                    
                    try:
                        # Step 1: Very light buffer smoothing
                        buffer_size = min(self.smooth_buffer_size * 0.3, 1.0)  # Smaller buffer
                        temp = smoothed.buffer(buffer_size).buffer(-buffer_size)
                        
                        # Only keep if we don't lose too much area
                        if temp.is_valid and temp.area > original_area * 0.7:
                            smoothed = temp
                        
                        # Step 2: Light simplification
                        if smoothed.is_valid:
                            tolerance = min(self.simplify_tolerance * 0.5, 1.0)  # Smaller tolerance
                            temp = smoothed.simplify(tolerance, preserve_topology=True)
                            
                            if temp.is_valid and temp.area > original_area * 0.7:
                                smoothed = temp
                        
                    except Exception as e:
                        print(f"Smoothing failed for polygon {i}: {e}, keeping original")
                        smoothed = poly
                
                # Handle MultiPolygon results
                if smoothed.geom_type == 'MultiPolygon':
                    for sub_poly in smoothed.geoms:
                        if sub_poly.is_valid and sub_poly.area >= area_threshold * 0.5:
                            processed_polygons.append(sub_poly)
                        else:
                            lost_polygons += 1
                else:
                    if smoothed.is_valid and smoothed.area > 0:
                        processed_polygons.append(smoothed)
                    else:
                        lost_polygons += 1
                        
            except Exception as e:
                print(f"Error processing polygon {i}: {e}")
                # Keep original if processing fails
                if poly.is_valid and poly.area >= area_threshold * 0.5:
                    processed_polygons.append(poly)
                else:
                    lost_polygons += 1
                continue
        
        print(f"Conservative smoothing complete: {len(processed_polygons)} polygons retained, {lost_polygons} lost")
        return processed_polygons
    
    def intelligent_polygon_merging(self, polygons, merge_distance=5.0):
        """
        Intelligent merging based on proximity - more conservative
        """
        if not polygons or len(polygons) <= 1:
            return polygons
            
        print(f"Applying intelligent polygon merging (distance: {merge_distance}m)...")
        
        # Skip merging if too many polygons (performance)
        if len(polygons) > 1000:
            print("Too many polygons for merging, skipping...")
            return polygons
        
        try:
            merged_polygons = []
            processed = set()
            
            for i, poly1 in enumerate(polygons):
                if i in processed:
                    continue
                    
                merge_group = [poly1]
                processed.add(i)
                
                # Find nearby polygons to merge
                for j, poly2 in enumerate(polygons):
                    if j <= i or j in processed:
                        continue
                        
                    try:
                        if poly1.distance(poly2) < merge_distance:
                            merge_group.append(poly2)
                            processed.add(j)
                    except:
                        continue
                
                # Merge the group
                if len(merge_group) > 1:
                    try:
                        merged = unary_union(merge_group)
                        if merged.geom_type == 'MultiPolygon':
                            merged_polygons.extend(list(merged.geoms))
                        else:
                            merged_polygons.append(merged)
                    except:
                        # If merging fails, keep originals
                        merged_polygons.extend(merge_group)
                else:
                    merged_polygons.extend(merge_group)
            
            print(f"Intelligent merging complete: {len(merged_polygons)} polygons")
            return merged_polygons
            
        except Exception as e:
            print(f"Error in polygon merging: {e}, returning original polygons")
            return polygons
    
    def create_geodataframe(self, polygons, crs):
        """Create GeoDataFrame with enhanced attributes"""
        print("Creating enhanced GeoDataFrame...")
        
        data = []
        for i, poly in enumerate(polygons):
            try:
                area = poly.area
                perimeter = poly.length
                
                # Convert area based on CRS
                if crs and hasattr(crs, 'to_epsg') and crs.to_epsg() == 4326:
                    # Rough conversion for geographic coordinates
                    area_ha = area * 111000 * 111000 / 10000
                else:
                    area_ha = area / 10000
                
                # Calculate shape metrics
                compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                elongation = perimeter / (2 * np.sqrt(np.pi * area)) if area > 0 else 0
                
                data.append({
                    'field_id': i + 1,
                    'area_m2': round(area, 2),
                    'area_ha': round(area_ha, 4),
                    'perimeter_m': round(perimeter, 2),
                    'compactness': round(compactness, 4),
                    'elongation': round(elongation, 4),
                    'geometry': poly
                })
            except Exception as e:
                print(f"Error creating record for polygon {i}: {e}")
                continue
        
        if not data:
            print("No valid polygon data to create GeoDataFrame!")
            return None
            
        gdf = gpd.GeoDataFrame(data, crs=crs)
        print(f"Enhanced GeoDataFrame created with {len(gdf)} polygons")
        return gdf
    
    def process_binary_mask_to_polygons(self, input_raster_path, output_shapefile_path, 
                                      merge_nearby=True, merge_distance=5.0, 
                                      use_watershed=True, pixel_size_m=0.3):
        """
        Complete enhanced pipeline for binary masks with conservative processing
        """
        print(f"\n{'='*70}")
        print("BINARY MASK TO POLYGON EXTRACTION PIPELINE")
        print(f"{'='*70}")
        
        try:
            # Step 1: Load binary mask
            binary_data, transform, crs, profile = self.load_raster(input_raster_path)
            
            if np.sum(binary_data) == 0:
                print("No cultivation pixels found in mask!")
                return None
            
            # Step 2: Edge enhancement (very light)
            binary_data = self.enhance_mask_edges(binary_data)
            
            # Step 3: Conservative morphological refinement
            cleaned_mask = self.morphological_refinement(binary_data, pixel_size_m)
            
            if np.sum(cleaned_mask) == 0:
                print("No pixels remaining after morphological processing! Try reducing min_area_m2.")
                return None
            
            # Step 4: Watershed segmentation (optional)
            watershed_result = None
            if use_watershed and np.sum(cleaned_mask) > 1000:  # Only for larger areas
                watershed_result = self.watershed_segmentation_refinement(cleaned_mask)
            
            # Step 5: Convert to polygons
            raw_polygons = self.raster_to_polygons_advanced(cleaned_mask, transform, crs, watershed_result)
            
            if not raw_polygons:
                print("No polygons generated! Try reducing min_area_m2 or disabling morphological processing.")
                return None
            
            # Step 6: Conservative polygon smoothing
            processed_polygons = self.conservative_polygon_smoothing(raw_polygons, crs)
            
            if not processed_polygons:
                print("No polygons after smoothing! Try increasing area tolerance or disabling smoothing.")
                return None
            
            # Step 7: Intelligent merging (optional)
            if merge_nearby and len(processed_polygons) < 500:  # Only merge if reasonable number
                final_polygons = self.intelligent_polygon_merging(processed_polygons, merge_distance)
            else:
                final_polygons = processed_polygons
            
            # Step 8: Create enhanced GeoDataFrame
            gdf = self.create_geodataframe(final_polygons, crs)
            
            if gdf is None or len(gdf) == 0:
                print("Failed to create GeoDataFrame!")
                return None
            
            # Step 9: Save results
            print(f"Saving results to: {output_shapefile_path}")
            os.makedirs(os.path.dirname(output_shapefile_path), exist_ok=True)
            gdf.to_file(output_shapefile_path)
            
            # Summary statistics
            print(f"\n{'='*70}")
            print("PROCESSING SUMMARY")
            print(f"{'='*70}")
            print(f"Total cultivation fields: {len(gdf)}")
            print(f"Total area: {gdf['area_ha'].sum():.2f} hectares")
            print(f"Average field size: {gdf['area_ha'].mean():.2f} hectares")
            print(f"Median field size: {gdf['area_ha'].median():.2f} hectares")
            print(f"Largest field: {gdf['area_ha'].max():.2f} hectares")
            print(f"Smallest field: {gdf['area_ha'].min():.2f} hectares")
            
            if len(gdf) > 0:
                print(f"Average compactness: {gdf['compactness'].mean():.3f}")
                print(f"Average elongation: {gdf['elongation'].mean():.3f}")
                
                # Field size distribution
                small_fields = len(gdf[gdf['area_ha'] < 0.5])
                medium_fields = len(gdf[(gdf['area_ha'] >= 0.5) & (gdf['area_ha'] < 2.0)])
                large_fields = len(gdf[gdf['area_ha'] >= 2.0])
                
                print(f"\nField size distribution:")
                print(f"  Small (< 0.5 ha): {small_fields} fields")
                print(f"  Medium (0.5-2 ha): {medium_fields} fields") 
                print(f"  Large (≥ 2 ha): {large_fields} fields")
            
            print(f"\nOutput saved to: {output_shapefile_path}")
            
            return gdf
            
        except Exception as e:
            print(f"Error in processing pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None


# Usage example for binary masks
if __name__ == "__main__":
    # Create extractor with conservative parameters for binary masks
    extractor = BinaryMaskPolygonExtractor(
        min_area_m2=50,                # Reduced minimum area
        simplify_tolerance=1.0,        # Reduced simplification
        smooth_buffer_size=1.0,        # Conservative smoothing
        edge_enhancement=False,        # Disable for binary masks
        boundary_refinement=True       # Light refinement
    )
    
    # Process binary mask
    input_raster = "/workspace/output/outputs/predictions_building_res0p3m.tif"
    output_shapefile = "/workspace/output/outputs/binary_cultivation_polygons.shp"
    
    result_gdf = extractor.process_binary_mask_to_polygons(
        input_raster_path=input_raster,
        output_shapefile_path=output_shapefile,
        merge_nearby=False,            # Disable merging initially
        merge_distance=3.0,            # Smaller merge distance
        use_watershed=False,           # Disable watershed initially  
        pixel_size_m=0.3              # Match your raster resolution
    )
    
    if result_gdf is not None:
        print("\n✅ Binary mask processing completed successfully!")
        print(f"📊 Fields extracted: {len(result_gdf)}")
        
        # Display quality metrics
        if len(result_gdf) > 0:
            print(f"\n📈 Quality Metrics:")
            print(f"   Average compactness: {result_gdf['compactness'].mean():.3f} (higher = more circular)")
            print(f"   Average elongation: {result_gdf['elongation'].mean():.3f} (lower = more compact)")
            
            # Preview results
            print("\n🔍 Sample results:")
            display_cols = ['field_id', 'area_ha', 'compactness', 'elongation']
            print(result_gdf[display_cols].head())
        
    else:
        print("❌ Processing failed. Try these adjustments:")
        print("   - Reduce min_area_m2 (e.g., 25 or 10)")
        print("   - Set edge_enhancement=False")
        print("   - Set boundary_refinement=False") 
        print("   - Set use_watershed=False")
        print("   - Check if your raster actually contains binary values (0, 1)")