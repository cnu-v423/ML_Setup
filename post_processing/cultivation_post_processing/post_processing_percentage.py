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
import numpy as np
from rasterio.mask import mask
import argparse
import warnings
warnings.filterwarnings('ignore')

def process_prediction_with_polygons(prediction_tiff_path, shapefile_path, output_base_path, threshold_min=30, threshold_max=40):
    """
    Process binary prediction TIFF with shapefile polygons and filter based on cultivation area percentage.
    
    Parameters:
    -----------
    prediction_tiff_path : str
        Path to the binary prediction TIFF file (0=non-cultivation, 1=cultivation)
    shapefile_path : str
        Path to the input shapefile containing polygons
    output_base_path : str
        Path for the output filtered folder base path
    threshold_min : float
        Minimum percentage threshold for white pixels (default: 30%)
    threshold_max : float
        Maximum percentage threshold for white pixels (default: 40%)
    """
    
    print("Loading shapefile...")
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    file_name = os.path.basename(prediction_tiff_path)   # Mandadam.tif
    village_name = os.path.splitext(file_name)[0]         # Mandadam

    # Create output folder: /workspace/output/cultivation_predictions/Mandadam
    output_folder = os.path.join(output_base_path, village_name)
    os.makedirs(output_folder, exist_ok=True)

    # Output shapefile path
    output_shapefile_path = os.path.join(output_folder, f"{village_name}.shp")

    print(f"Output folder: {output_folder}")
    print(f"Shapefile will be saved as: {output_shapefile_path}")
    
    print("Loading prediction TIFF...")
    # Load the prediction TIFF
    with rasterio.open(prediction_tiff_path) as src:
        prediction_data = src.read(1)  # Read first band
        prediction_transform = src.transform
        prediction_crs = src.crs
        prediction_bounds = src.bounds
    
    print(f"Prediction TIFF shape: {prediction_data.shape}")
    print(f"Prediction TIFF CRS: {prediction_crs}")
    print(f"Shapefile CRS: {gdf.crs}")
    
    # Ensure CRS compatibility
    if gdf.crs != prediction_crs:
        print("Reprojecting shapefile to match TIFF CRS...")
        gdf = gdf.to_crs(prediction_crs)
    
    # Initialize list to store valid polygons
    valid_polygons = []
    
    print(f"Processing {len(gdf)} polygons...")
    
    for idx, row in gdf.iterrows():
        try:
            polygon = row.geometry
            
            # Check if polygon intersects with raster bounds
            polygon_bounds = polygon.bounds
            if (polygon_bounds[0] > prediction_bounds[2] or 
                polygon_bounds[2] < prediction_bounds[0] or
                polygon_bounds[1] > prediction_bounds[3] or 
                polygon_bounds[3] < prediction_bounds[1]):
                print(f"Polygon {idx} is outside raster bounds, skipping...")
                continue
            
            # Create a temporary raster dataset for masking
            with rasterio.open(prediction_tiff_path) as src:
                # Mask the raster with the polygon
                try:
                    masked_data, masked_transform = mask(src, [polygon], crop=True, filled=True)
                    masked_data = masked_data[0]  # Get first band
                    
                    # Calculate total pixels in the masked area (non-zero pixels after masking)
                    # valid_mask = masked_data != 0  # Pixels that are actually within the polygon
                    # valid_mask = masked_data != src.nodata if src.nodata is not None else np.ones_like(masked_data, dtype=bool)
                    inside_mask = masked_data >= 0  # everything inside polygon

                    total_pixels = np.sum(inside_mask)
                    
                    if total_pixels == 0:
                        print(f"Polygon {idx}: No valid pixels found, skipping...")
                        continue
                    
                    # Calculate white pixels (cultivation land) within the valid area
                    white_pixels = np.sum(masked_data == 1)
                    
                    # Calculate percentage
                    white_percentage = (white_pixels / total_pixels) * 100
                    
                    print(f"Polygon {idx}: {white_percentage:.2f}% cultivation area ({white_pixels}/{total_pixels} pixels)")
                    
                    # Check if percentage meets threshold criteria
                    if white_percentage >= threshold_min:
                        # print(f"Polygon {idx}: ACCEPTED (within {threshold_min}-{threshold_max}% threshold)")
                        valid_polygons.append(row)
                    else:
                        print(f"Polygon {idx}: REJECTED (outside {threshold_min}-{threshold_max}% threshold)")
                        
                except Exception as e:
                    print(f"Error processing polygon {idx}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error with polygon {idx}: {str(e)}")
            continue
    
    # Create output GeoDataFrame with valid polygons
    if valid_polygons:
        print(f"\nCreating output shapefile with {len(valid_polygons)} valid polygons...")
        output_gdf = gpd.GeoDataFrame(valid_polygons, crs=gdf.crs)
        
        # Reset index
        output_gdf = output_gdf.reset_index(drop=True)
        
        # Save the filtered shapefile
        output_gdf.to_file(output_shapefile_path)
        print(f"Output shapefile saved to: {output_shapefile_path}")
        
        # Print summary statistics
        print(f"\nSummary:")
        print(f"Total input polygons: {len(gdf)}")
        print(f"Valid polygons (within {threshold_min}-{threshold_max}% threshold): {len(valid_polygons)}")
        print(f"Acceptance rate: {(len(valid_polygons)/len(gdf)*100):.2f}%")
        
    else:
        print("No valid polygons found that meet the threshold criteria!")
        
    return len(valid_polygons) if valid_polygons else 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter polygons based on prediction TIFF overlap thresholds")

    parser.add_argument('--prediction_tiff', required=True,
                        help='Path to input prediction TIFF file')
    parser.add_argument('--input_shapefile', required=True,
                        help='Path to input shapefile with polygons')
    parser.add_argument('--output_base_path', required=True,
                        help='Path for output filtered shapefile')
    parser.add_argument('--threshold_min', type=int, default=20,
                        help='Minimum overlap threshold (default: 30)')
    parser.add_argument('--threshold_max', type=int, default=40,
                        help='Maximum overlap threshold (default: 40)')

    args = parser.parse_args()

    print(f"=== Processing with {args.threshold_min}-{args.threshold_max}% threshold range ===")
    valid_count = process_prediction_with_polygons(
        args.prediction_tiff,
        args.input_shapefile,
        args.output_base_path,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max
    )
    
   