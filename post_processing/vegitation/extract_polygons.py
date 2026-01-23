import rasterio
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
import argparse
import os


def extract_polygons_for_value(
    input_raster_path,
    output_vector_path,
    target_value=2,
    simplify_tolerance=0.5,   # meters (adjust!)
    dissolve=True
):
    if not os.path.exists(input_raster_path):
        raise FileNotFoundError(f"Input raster not found: {input_raster_path}")

    print("📖 Reading raster:", input_raster_path)

    with rasterio.open(input_raster_path) as src:
        band = src.read(1)
        transform = src.transform
        crs = src.crs

    mask = band == target_value

    polygons = []

    print("🧩 Raster → Polygon")

    for geom, val in shapes(band, mask=mask, transform=transform):
        if val == target_value:
            polygons.append(shape(geom))

    if not polygons:
        print("⚠️ No polygons found")
        return

    print(f"✅ Initial polygons: {len(polygons):,}")

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)

    # ----------------------------------
    # FIX GEOMETRY (MANDATORY)
    # ----------------------------------
    print("🛠 Fixing invalid geometries")
    gdf["geometry"] = gdf.geometry.buffer(0)

    # ----------------------------------
    # DISSOLVE (MERGE touching polygons)
    # # ----------------------------------
    # if dissolve:
    #     print("🧬 Dissolving polygons")
    #     merged = unary_union(gdf.geometry)
    #     gdf = gpd.GeoDataFrame(geometry=[merged], crs=crs)

    # ----------------------------------
    # SIMPLIFY (HUGE size reduction)
    # ----------------------------------
    print(f"📉 Simplifying geometry (tolerance={simplify_tolerance})")
    gdf["geometry"] = gdf.geometry.simplify(
        tolerance=simplify_tolerance,
        preserve_topology=True
    )

    print("💾 Saving vector:", output_vector_path)

    # ----------------------------------
    # SAVE AS GEOPACKAGE (VERY IMPORTANT)
    # ----------------------------------
    gdf.to_file(output_vector_path)

    print("🎉 Done! File size drastically reduced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_tiff", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--band_value", type=int, default=2)
    parser.add_argument("--simplify", type=float, default=0.1)

    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.predicted_tiff))[0]
    os.makedirs(args.output_path, exist_ok=True)

    output_gpkg = os.path.join(
        args.output_path,
        f"{base}_value_{args.band_value}.shp"
    )

    extract_polygons_for_value(
        input_raster_path=args.predicted_tiff,
        output_vector_path=output_gpkg,
        target_value=args.band_value,
        simplify_tolerance=args.simplify
    )
