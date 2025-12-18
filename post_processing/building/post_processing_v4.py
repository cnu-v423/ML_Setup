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


def process_building_mask_to_shapefile(input_raster_path, output_shapefile_path,
                                      building_value=1, kernel_size=3):
    """
    Performs the same processing as your raster code but exports the result
    as a shapefile with polygons representing buildings.
    """

    if not os.path.exists(input_raster_path):
        print(f"❌ Error: Input file not found: {input_raster_path}")
        return

    print(f"--- Raster → Post-process → Shapefile ---")
    print(f"Reading raster: {input_raster_path}")

    try:
        with rasterio.open(input_raster_path) as src:
            original_mask = src.read(1)
            transform = src.transform
            crs = src.crs

            # --- STEP 1: Extract building pixels ---
            print(f"Creating building binary mask (value={building_value})...")
            binary_buildings = (original_mask == building_value).astype(np.uint8)

            # --- STEP 2: Morphological opening ---
            print(f"Applying opening (kernel={kernel_size})...")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            opened_mask = cv2.morphologyEx(binary_buildings, cv2.MORPH_OPEN, kernel)

            # --- STEP 3: Fill holes ---
            print(f"Filling internal holes...")
            final_mask = ndimage.binary_fill_holes(opened_mask).astype(np.uint8)

        # --- STEP 4: Raster → Polygons ---
        print(f"Converting final mask to polygons...")

        polygons = []
        values = []

        for geom, val in shapes(final_mask, transform=transform):
            if val == 1:   # Only building pixels
                # polygons.append(shape(geom))
                # values.append(building_value)
                poly = shape(geom)

                # Smooth polygon
                poly_smooth = chaikin_smooth(poly, iterations=2)

                # poly_smooth = poly_smooth.simplify(
                #     tolerance=0.5,
                #     preserve_topology=True
                # )

                 # --- 🔥 REMOVE SMALL POLYGONS HERE ---
                area = poly_smooth.area 
                if area < 2:              # REMOVE tiny polygons (threshold = 1)
                    continue

                polygons.append(poly_smooth)
                values.append(building_value)

        if len(polygons) == 0:
            print("⚠️ No polygons generated. No buildings after processing.")
            return

        print(f"Generated {len(polygons):,} building polygons.")

        # --- STEP 5: Save Shapefile ---
        gdf = gpd.GeoDataFrame({"value": values}, geometry=polygons, crs=crs)

        print(f"Saving shapefile: {output_shapefile_path}")
        gdf.to_file(output_shapefile_path)

        print("✅ Shapefile saved successfully!")

    except Exception as e:
        print(f"❌ Error occurred: {e}")


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
    process_building_mask_to_shapefile(
        input_raster_path=args.predicted_tiff,
        output_shapefile_path=output_shp_path,
        building_value=args.building_value,
        kernel_size=args.kernel_size
    )
