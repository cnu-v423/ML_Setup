import rasterio
import numpy as np
import cv2
from scipy import ndimage
from skimage.measure import label, regionprops
import argparse
import os
import json
from shapely.geometry import Polygon
from rasterio.features import rasterize
import geopandas as gpd
import warnings


def process_building_mask_to_rotated_rectangles(input_raster_path, output_raster_path, building_value=1, min_area_pixels=10):
    """
    Applies post-processing, fills interior holes, extracts connected components,
    and then converts each component into its MINIMUM ROTATED BOUNDING BOX rectangle.
    Saves these rotated rectangles as both raster and shapefile.

    Args:
        input_raster_path (str): Path to the input raster file.
        output_raster_path (str): Path to save the processed rotated rectangular raster file.
        building_value (int): The pixel value representing buildings.
        min_area_pixels (int): Minimum area (in pixels) for a detected building component
                               to be considered valid and converted to a bounding box.
    """
    if not os.path.exists(input_raster_path):
        print(f"❌ Error: Input file not found at {input_raster_path}")
        return

    print(f"--- Starting Building Mask to Rotated Rectangles Processing ---")
    print(f"Reading raster: {input_raster_path}")

    try:
        with rasterio.open(input_raster_path) as src:
            original_mask = src.read(1)
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs

            # --- Step 1: Create a binary mask for the specified building value ---
            print(f"Isolating pixels with value '{building_value}' for processing...")
            binary_buildings = (original_mask == building_value).astype(np.uint8)
            
            initial_pixel_count = np.sum(binary_buildings)
            if initial_pixel_count == 0:
                print(f"⚠️ Warning: No pixels with value '{building_value}' found. Output will be an empty mask.")
                output_mask = np.zeros_like(original_mask, dtype=np.uint8)
                
                # Create empty shapefile
                empty_gdf = gpd.GeoDataFrame(columns=['bldg_id', 'area_px', 'width_px', 'height_px', 'angle_deg'], 
                                           geometry=[], crs=crs)
                    
            else:
                print(f"Found {initial_pixel_count:,} building pixels to process.")

                # --- Step 2: Fill Holes Inside Buildings ---
                print("Filling interior holes in building shapes...")
                filled_mask = ndimage.binary_fill_holes(binary_buildings).astype(np.uint8)
                
                # --- Step 3: Find contours for each connected component ---
                print("Finding contours for connected components...")
                contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # --- Step 4: Extract minimum rotated bounding box for each contour ---
                print("Extracting minimum rotated bounding boxes (MRBBs)...")
                
                # Lists to hold data for shapefile creation
                polygons = []
                properties = []
                shapes_to_rasterize = []
                
                building_id_counter = 0

                for contour in contours:
                    contour_area = cv2.contourArea(contour)
                    if contour_area >= min_area_pixels:
                        building_id_counter += 1

                        # Get the minimum rotated bounding rectangle
                        rect = cv2.minAreaRect(contour)
                        center, (width, height), angle = rect
                        
                        # Get the 4 corner points of the rectangle
                        box_points = cv2.boxPoints(rect)
                        
                        # Convert pixel coordinates to geographic coordinates
                        geo_corners = []
                        for px_x, px_y in box_points:
                            geo_x, geo_y = transform * (px_x, px_y)
                            geo_corners.append((geo_x, geo_y))
                        
                        # Close the polygon by adding the first point at the end
                        geo_corners.append(geo_corners[0])
                        
                        # Create Shapely Polygon
                        polygon_geom = Polygon(geo_corners[:-1])  # Polygon constructor automatically closes
                        
                        # Store data
                        polygons.append(polygon_geom)
                        properties.append({
                            'bldg_id': building_id_counter,
                            'area_px': int(contour_area),
                            'width_px': round(width, 2),
                            'height_px': round(height, 2),
                            'angle_deg': round(angle, 2)
                        })
                        
                        # For rasterization
                        shapes_to_rasterize.append((polygon_geom, building_value))
                
                # --- Step 5: Create GeoDataFrame and save shapefile ---
                if polygons:
                    print(f"Creating GeoDataFrame with {len(polygons)} building polygons...")
                    
                    gdf = gpd.GeoDataFrame(properties, geometry=polygons, crs=crs)
                    
                    # Generate shapefile path
                    input_dir = os.path.dirname(input_raster_path)
                    input_filename = os.path.splitext(os.path.basename(input_raster_path))[0]
                    shapefile_output_path = os.path.join(input_dir, f"{input_filename}_buildings.shp")
                    
                    # Save shapefile with suppressed warnings about column name truncation
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gdf.to_file(shapefile_output_path)
                    
                    print(f"✅ Shapefile saved to: {shapefile_output_path}")
                    
                    # Print summary statistics
                    print(f"\n--- Summary Statistics ---")
                    print(f"Total buildings processed: {len(polygons)}")
                    print(f"Average building area: {gdf['area_px'].mean():.1f} pixels")
                    print(f"Largest building: {gdf['area_px'].max()} pixels")
                    print(f"Smallest building: {gdf['area_px'].min()} pixels")
                
            #     # --- Step 6: Rasterize the rotated bounding box polygons ---
            #     print(f"Rasterizing {len(shapes_to_rasterize)} rotated building rectangles...")
                
            #     if shapes_to_rasterize:
            #         output_mask = rasterize(
            #             shapes_to_rasterize,
            #             out_shape=original_mask.shape,
            #             transform=transform,
            #             fill=0,
            #             all_touched=False,
            #             dtype=np.uint8
            #         )
                    
            #         final_pixel_count = np.sum(output_mask == building_value)
            #         print(f"Total building pixels in output rectangles: {final_pixel_count:,}")
            #     else:
            #         output_mask = np.zeros_like(original_mask, dtype=np.uint8)

            # # --- Step 7: Save the final rotated rectangular raster ---
            # profile.update(dtype=rasterio.uint8, compress='lzw', nodata=0)

            # print(f"Saving processed rotated rectangular mask to: {output_raster_path}")
            # with rasterio.open(output_raster_path, 'w', **profile) as dst:
            #     dst.write(output_mask, 1)

            print(f"✅ Processing complete!")

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()


def validate_geometries(gdf):
    """
    Validate and fix any invalid geometries in the GeoDataFrame
    """
    print("Validating geometries...")
    invalid_count = 0
    
    for idx, geom in enumerate(gdf.geometry):
        if not geom.is_valid:
            invalid_count += 1
            # Try to fix invalid geometry
            fixed_geom = geom.buffer(0)
            if fixed_geom.is_valid:
                gdf.loc[idx, 'geometry'] = fixed_geom
                print(f"Fixed invalid geometry at index {idx}")
            else:
                print(f"Could not fix geometry at index {idx}")
    
    if invalid_count == 0:
        print("✅ All geometries are valid")
    else:
        print(f"⚠️ Found and attempted to fix {invalid_count} invalid geometries")
    
    return gdf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a building raster mask to minimum rotated rectangular shapes and export to Shapefile.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_raster", required=True, 
                       help="Path to the input multi-value TIFF mask.")
    parser.add_argument("--output_raster", required=False, 
                       help="Path to save the processed rotated rectangular TIFF mask.")
    parser.add_argument("--building_value", type=int, default=1, 
                       help="The integer pixel value that represents buildings.")
    parser.add_argument("--min_area_pixels", type=int, default=10, 
                       help="Minimum area (in pixels) for a building component to be considered valid.")
    
    args = parser.parse_args()
    
    process_building_mask_to_rotated_rectangles(
        input_raster_path=args.input_raster,
        output_raster_path=None,
        building_value=args.building_value,
        min_area_pixels=args.min_area_pixels
    )
