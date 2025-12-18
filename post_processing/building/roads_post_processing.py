import rasterio
import numpy as np
import cv2
from scipy import ndimage
import argparse
import os
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape, Polygon

def chaikin_smooth(polygon, iterations=3):
    """
    Smooth polygon using Chaikin's corner cutting algorithm.
    """
    coords = list(polygon.exterior.coords)
    for _ in range(iterations):
        new_coords = []
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]

            Q = (0.75 * p1[0] + 0.25 * p2[0],
                 0.75 * p1[1] + 0.25 * p2[1])

            R = (0.25 * p1[0] + 0.75 * p2[0],
                 0.25 * p1[1] + 0.75 * p2[1])

            new_coords.extend([Q, R])

        new_coords.append(new_coords[0])  # close ring
        coords = new_coords

    return Polygon(coords)


def process_road_mask_to_shapefile(
    input_raster_path,
    output_shapefile_path,
    road_value=1,
    kernel_size=2,
    min_area=0.5
):
    print(f"--- Raster → Road Post-process → Shapefile ---")

    with rasterio.open(input_raster_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    # ------------------------------------------------
    # STEP 1: Binary mask
    # ------------------------------------------------
    binary = (mask == road_value).astype(np.uint8)

    if np.sum(binary) == 0:
        print("⚠️ No road pixels found")
        return

    # ------------------------------------------------
    # STEP 2: VERY LIGHT dilation (NOT closing)
    # ------------------------------------------------
    print("Applying controlled dilation...")

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS,  # 🔑 prevents side merging
        (kernel_size, kernel_size)
    )

    processed = cv2.dilate(binary, kernel, iterations=1)

    # ------------------------------------------------
    # STEP 3: Polygonize
    # ------------------------------------------------
    polygons = []
    values = []

    for geom, val in shapes(processed, transform=transform):
        if val != 1:
            continue

        poly = shape(geom)

        # Light smoothing
        poly = chaikin_smooth(poly, iterations=1)

        # Remove noise
        if poly.area < min_area:
            continue

        polygons.append(poly)
        values.append(road_value)

    if not polygons:
        print("⚠️ No road polygons generated")
        return

    print(f"Generated {len(polygons)} road polygons")

    gdf = gpd.GeoDataFrame(
        {"class": values},
        geometry=polygons,
        crs=crs
    )

    gdf.to_file(output_shapefile_path)
    print("✅ Road shapefile saved")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mask and save polygons as shapefile")
    parser.add_argument("--predicted_tiff", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--building_value", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3)
    
    args = parser.parse_args()

    # -------------------------------
    # 1. Get TIFF filename (without extension)
    # -------------------------------
    base_name = os.path.basename(args.predicted_tiff)            # example: Mandadam_12.tif
    name_no_ext = os.path.splitext(base_name)[0]               # example: Mandadam_12

    # -------------------------------
    # 2. Create folder inside output_shapefile
    # -------------------------------
    output_folder = os.path.join(args.output_path, name_no_ext)
    os.makedirs(output_folder, exist_ok=True)

    # -------------------------------
    # 3. Create shapefile path inside folder
    # -------------------------------
    output_shp_path = os.path.join(output_folder, f"{name_no_ext}.shp")

    # -------------------------------
    # 4. Run processing
    # -------------------------------
    process_road_mask_to_shapefile(
        input_raster_path=args.predicted_tiff,
        output_shapefile_path=output_shp_path,
        road_value=255,
        kernel_size=args.kernel_size,
        min_area=0.1
    )
