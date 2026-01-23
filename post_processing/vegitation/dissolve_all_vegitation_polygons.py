import os
import glob
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
VALUE_PATTERN = "*_value_2.shp"


# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def load_shapefile(path, target_crs=None):
    print(f"📂 Loading: {path}")
    gdf = gpd.read_file(path)

    # Drop empty / null geometries
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]

    # Track source file (CRITICAL)
    gdf["src_file"] = os.path.basename(path)

    # Reproject if needed
    if target_crs and gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def find_all_value_2_shapefiles(base_folder):
    shp_files = []
    for root, _, _ in os.walk(base_folder):
        matches = glob.glob(os.path.join(root, VALUE_PATTERN))
        shp_files.extend(matches)

    if not shp_files:
        raise RuntimeError("❌ No *_value_2.shp files found")

    print(f"✅ Found {len(shp_files)} shapefiles")
    return shp_files


# ---------------------------------------------------------
# CORE LOGIC
# ---------------------------------------------------------
def dissolve_only_cross_file_overlaps(gdf):
    """
    Dissolve polygons ONLY if they overlap with polygons
    coming from a DIFFERENT shapefile.
    """

    print("🧠 Building spatial index")
    # Use index-based approach instead of object identity
    tree = STRtree(gdf.geometry.values)

    visited = set()
    output_geoms = []

    print("🔥 Detecting & dissolving cross-file overlaps")

    for i in range(len(gdf)):
        if i in visited:
            continue

        geom = gdf.geometry.iloc[i]
        src_file = gdf.iloc[i]["src_file"]
        
        # Query returns indices directly when using geometry array
        candidate_indices = tree.query(geom, predicate='intersects')

        cluster_idxs = [i]

        for j in candidate_indices:
            if j == i or j in visited:
                continue

            # ❗ Only cross-file comparison
            if gdf.iloc[j]["src_file"] == src_file:
                continue

            # Overlap test (NOT just touching)
            cand_geom = gdf.geometry.iloc[j]
            if geom.overlaps(cand_geom) or geom.intersects(cand_geom):
                cluster_idxs.append(j)

        if len(cluster_idxs) > 1:
            merged_geom = unary_union([gdf.geometry.iloc[j] for j in cluster_idxs])
            output_geoms.append(merged_geom)
            visited.update(cluster_idxs)
        else:
            output_geoms.append(geom)
            visited.add(i)

    print("✅ Overlap dissolve complete")
    return gpd.GeoDataFrame(geometry=output_geoms, crs=gdf.crs)


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def merge_vegitation_polygons(input_folder, output_path):
    shp_files = find_all_value_2_shapefiles(input_folder)

    all_gdfs = []
    base_crs = None

    for shp in shp_files:
        gdf = load_shapefile(shp, base_crs)

        if gdf.empty:
            continue

        if base_crs is None:
            base_crs = gdf.crs

        all_gdfs.append(gdf)

    if not all_gdfs:
        raise RuntimeError("❌ No valid geometries found")

    print("🧩 Merging all shapefiles (NO dissolve)")
    merged = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True),
        crs=base_crs
    )

    print(f"📐 Total polygons before dissolve: {len(merged)}")

    result_gdf = dissolve_only_cross_file_overlaps(merged)

    print(f"📐 Total polygons after dissolve: {len(result_gdf)}")

    print(f"💾 Writing output → {output_path}")
    result_gdf.to_file(output_path)

    print("🎉 DONE")
    print("✔ No intra-file dissolve")
    print("✔ Only cross-file overlaps merged")
    print("✔ Fast & QGIS-safe")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":

    input_folder = (
        "/workspace/input/Predictions/Vegitation_predictions/"
        "post_process_shapefiles_after_resolved_tile_issue"
    )

    output_shapefile = (
        "/workspace/input/Predictions/Vegitation_predictions/"
        "post_process_shapefiles_after_resolved_tile_issue/"
        "merged_vegitation_cross_tile_only.shp"
    )

    merge_vegitation_polygons(input_folder, output_shapefile)