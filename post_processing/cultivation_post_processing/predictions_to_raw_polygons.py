import os
import argparse
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_polygons_in_memory(input_tif, output_path, target_value=1):
    """Convert raster to polygons and return as GeoDataFrame."""
    with rasterio.open(input_tif) as src:
        image = src.read(1)
        mask = image == target_value

        polygons = []
        for geom, val in shapes(image, mask=mask, transform=src.transform):
            polygons.append(shape(geom))

        if not polygons:
            return gpd.GeoDataFrame(columns=["geometry"], crs=src.crs)

        dataframe = gpd.GeoDataFrame(geometry=polygons, crs=src.crs)

        # Create output folder for this TIFF
        individual_output_folder = os.path.join(
            output_path,
            os.path.splitext(os.path.basename(input_tif))[0]
        )
        os.makedirs(individual_output_folder, exist_ok=True)

        # FIX: shapefile must end with .shp
        individual_output = os.path.join(
            individual_output_folder,
            os.path.splitext(os.path.basename(input_tif))[0] + ".shp"
        )

        # Write shapefile
        dataframe.to_file(individual_output, driver="ESRI Shapefile")

        return dataframe

        



def process_folder(base_folder, output_shapefile):
    all_gdfs = []

    for file in os.listdir(base_folder):
        if not file.endswith(".tif"):
            continue
        if file.endswith("_prob.tif"):
            continue

        tif_path = os.path.join(base_folder, file)
        print(f"Processing: {tif_path}")

        gdf = raster_to_polygons_in_memory(tif_path, output_shapefile)

        if len(gdf) > 0:
            all_gdfs.append(gdf)

    if not all_gdfs:
        print("No TIFF files processed.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TIFFs to polygons without merging")
    parser.add_argument("--input_folder", required=True, help="Folder containing TIFF files")
    parser.add_argument("--output_shape_file_path", required=True, help="Output shapefile path")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_shape_file_path)
