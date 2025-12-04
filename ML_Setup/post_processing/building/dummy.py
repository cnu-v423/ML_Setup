
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


def process_building_mask_to_rotated_rectangles(input_raster_path, output_raster_path, output_geojson_path=None, building_value=1, min_area_pixels=10):
    """
    Applies post-processing, fills interior holes, extracts connected components,
    and then converts each component into its MINIMUM ROTATED BOUNDING BOX rectangle.
    Optionally saves these rotated rectangles as a GeoJSON file.

    Args:
        input_raster_path (str): Path to the input raster file.
        output_raster_path (str): Path to save the processed rotated rectangular raster file.
        output_geojson_path (str, optional): Path to save the GeoJSON file of rotated bounding boxes.
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
            
            # Ensure the transform is compatible for minAreaRect (non-rotated images might simplify things,
            # but we handle the general case of the image transform here).
            # For minAreaRect, we work in pixel coordinates directly.

            # --- Step 1: Create a binary mask for the specified building value ---
            print(f"Isolating pixels with value '{building_value}' for processing...")
            binary_buildings = (original_mask == building_value).astype(np.uint8)
            
            initial_pixel_count = np.sum(binary_buildings)
            if initial_pixel_count == 0:
                print(f"⚠️ Warning: No pixels with value '{building_value}' found. Output will be an empty mask.")
                output_mask = np.zeros_like(original_mask, dtype=np.uint8)
                if output_geojson_path:
                    with open(output_geojson_path, 'w') as f:
                        json.dump({"type": "FeatureCollection", "features": []}, f)
            else:
                print(f"Found {initial_pixel_count:,} building pixels to process.")

                # --- Step 2: Fill Holes Inside Buildings ---
                # We do this before finding contours to ensure holes don't break a single building into multiple components.
                print("Filling interior holes in building shapes...")
                filled_mask = ndimage.binary_fill_holes(binary_buildings).astype(np.uint8)
                
                # --- Step 3: Find contours for each connected component ---
                # cv2.findContours expects a single-channel binary image.
                print("Finding contours for connected components...")
                # Use a copy to avoid modifying filled_mask
                contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # --- Step 4: Extract minimum rotated bounding box for each contour ---
                print("Extracting minimum rotated bounding boxes (MRBBs)...")
                
                # List to hold Shapely Polygons for rasterization
                shapes_to_rasterize = []
                features = [] # For GeoJSON output

                building_id_counter = 0

                for contour in contours:
                    # Filter out small contours that might be noise
                    # Calculate area from contour directly, or use filled_mask and regionprops if preferred.
                    # Here we use contourArea for a direct measure of the contour's size.
                    contour_area = cv2.contourArea(contour)
                    if contour_area >= min_area_pixels:
                        building_id_counter += 1

                        # Get the minimum rotated bounding rectangle
                        # rect: (center (x,y), size (width, height), angle of rotation)
                        rect = cv2.minAreaRect(contour)
                        
                        # Get the 4 corner points of the rectangle
                        box_points = cv2.boxPoints(rect) # Returns points as (x,y)
                        
                        # Convert the pixel coordinates of the box corners to geographic coordinates
                        # We need to apply the raster's affine transform to each corner.
                        geo_corners = []
                        for px_x, px_y in box_points:
                            # rasterio's transform expects (col, row), returns (x, y)
                            # OpenCV uses (x,y) where x is column, y is row.
                            geo_x, geo_y = transform * (px_x, px_y)
                            geo_corners.append((geo_x, geo_y))
                        
                        # Create a Shapely Polygon from the geographic corners
                        # Make sure to close the polygon (first point == last point)
                        polygon_geom = Polygon(geo_corners)
                        
                        # Store for rasterization (tuple of (geometry, value))
                        shapes_to_rasterize.append((polygon_geom, building_value))
                        
                        # Store for GeoJSON output
                        features.append({
                            "type": "Feature",
                            "geometry": json.loads(json.dumps(polygon_geom.__geo_interface__)), # Ensure JSON serializable
                            "properties": {
                                "building_id": building_id_counter,
                                "original_contour_area_pixels": contour_area,
                                "bbox_width_pixels": rect[1][0],
                                "bbox_height_pixels": rect[1][1],
                                "bbox_angle_degrees": rect[2]
                            }
                        })
                
                # --- Step 5: Rasterize the rotated bounding box polygons ---
                print(f"Rasterizing {len(shapes_to_rasterize)} rotated building rectangles...")
                # rasterize expects a list of (geometry, value) tuples
                # `fill=0` means areas not covered by polygons will be 0 (background)
                # `all_touched=True` means pixels that are partially covered are included
                output_mask = rasterize(
                    shapes_to_rasterize,
                    out_shape=original_mask.shape,
                    transform=transform,
                    fill=0,
                    all_touched=False, # Use False for crisper edges, True if you want to capture all partial overlaps
                    dtype=np.uint8
                )
                
                final_pixel_count = np.sum(output_mask == building_value)
                print(f"Processed into {len(features)} rectangular building features.")
                print(f"Total building pixels in output rectangles: {final_pixel_count:,}.")

                # --- Step 6: Save GeoJSON if requested ---
                if output_geojson_path:
                    input_dir = os.path.dirname(input_raster_path)
                    input_filename = os.path.splitext(os.path.basename(input_raster_path))[0]
                    shapefile_output_path = os.path.join(input_dir, f"{input_filename}_post_v3.shp")

                    gdf = gpd.GeoDataFrame(
                        [f["properties"] for f in features],
                        geometry=[Polygon(f["geometry"]["coordinates"][0]) for f in features],
                        crs=src.crs  # Preserve spatial reference
                    )
                    gdf.to_file(shapefile_output_path)

            # --- Step 7: Save the final rotated rectangular raster ---
            profile.update(dtype=rasterio.uint8, compress='lzw', nodata=0) # Assume 0 is nodata/background

            print(f"Saving processed rotated rectangular mask to: {output_raster_path}")
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                dst.write(output_mask, 1)

            print(f"✅ Processing complete!")

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a building raster mask to minimum rotated rectangular shapes and optionally export to GeoJSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_raster", required=True, help="Path to the input multi-value TIFF mask.")
    parser.add_argument("--output_raster", required=True, help="Path to save the processed rotated rectangular TIFF mask.")
    parser.add_argument("--output_geojson", help="Optional: Path to save the GeoJSON file of rotated rectangular building footprints.")
    parser.add_argument("--building_value", type=int, default=1, help="The integer pixel value that represents buildings.")
    parser.add_argument("--min_area_pixels", type=int, default=10, help="Minimum area (in pixels) for a building component to be considered valid.")
    
    args = parser.parse_args()
    
    process_building_mask_to_rotated_rectangles(
        input_raster_path=args.input_raster,
        output_raster_path=args.output_raster,
        output_geojson_path=args.output_geojson,
        building_value=args.building_value,
        min_area_pixels=args.min_area_pixels
    )