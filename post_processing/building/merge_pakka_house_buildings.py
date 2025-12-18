import geopandas as gpd
import argparse
from shapely.ops import unary_union
import pandas as pd

def load_and_fix(path):
    gdf = gpd.read_file(path)
    gdf["geometry"] = gdf["geometry"].buffer(0)  # Fix geometry
    return gdf

def merge_shapefiles(building_shp, pakka_shp, output_path):
    # Load shapefiles
    gdf1 = load_and_fix(building_shp)
    gdf2 = load_and_fix(pakka_shp)

    # Merge them
    merged = gpd.GeoDataFrame(
        pd.concat([gdf1, gdf2], ignore_index=True),
        crs=gdf1.crs
    )

    # Dissolve ALL overlaps → becomes MultiPolygon
    dissolved = unary_union(merged.geometry)

    # Explode multipolygons into separate rows (single polygons)
    out_gdf = gpd.GeoDataFrame(geometry=[dissolved], crs=merged.crs)
    out_gdf = out_gdf.explode(index_parts=False, ignore_index=True)

    # Save output
    out_gdf.to_file(output_path)
    print("✔ Done! Overlaps dissolved but polygons remain separate.")
    print("✔ You can now select each polygon individually in QGIS.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dissolve overlapping polygons but keep them selectable as separate polygons"
    )
    parser.add_argument("--building_shapefile", required=True)
    parser.add_argument("--pakka_house_shapefile", required=True)
    parser.add_argument("--output_shapefile", required=True)

    args = parser.parse_args()

    merge_shapefiles(
        args.building_shapefile,
        args.pakka_house_shapefile,
        args.output_shapefile
    )
