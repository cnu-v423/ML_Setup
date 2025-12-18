import geopandas as gpd
import os

def keep_b_polygons_intersecting_a(
    shapefile_a,
    shapefile_b,
    output_shapefile
):
    """
    Keeps only polygons from B that intersect with polygons from A
    """

    # -----------------------------
    # Read shapefiles
    # -----------------------------
    gdf_a = gpd.read_file(shapefile_a)
    gdf_b = gpd.read_file(shapefile_b)

    # -----------------------------
    # Ensure same CRS
    # -----------------------------
    if gdf_a.crs != gdf_b.crs:
        gdf_b = gdf_b.to_crs(gdf_a.crs)

    # -----------------------------
    # Spatial filter: B ∩ A
    # -----------------------------
    # This keeps only B polygons that intersect any A polygon
    gdf_b_filtered = gpd.sjoin(
        gdf_b,
        gdf_a,
        how="inner",
        predicate="intersects"
    )

    # -----------------------------
    # Clean up join columns
    # -----------------------------
    gdf_b_filtered = gdf_b_filtered.drop(
        columns=[col for col in gdf_b_filtered.columns if col.startswith("index_")],
        errors="ignore"
    )

    # -----------------------------
    # Save output
    # -----------------------------
    os.makedirs(os.path.dirname(output_shapefile), exist_ok=True)
    gdf_b_filtered.to_file(output_shapefile)

    print(f"✅ Saved filtered shapefile: {output_shapefile}")
    print(f"Original B polygons: {len(gdf_b)}")
    print(f"Kept polygons: {len(gdf_b_filtered)}")


# -----------------------------
# USAGE
# -----------------------------
shapefile_a = "/workspace/input/Predictions/Building_predictions/building_predictions_shapefile_new/building_predictions_merged_shape_file.shp"
shapefile_b = "/workspace/input/Predictions/Building_predictions/New_building_post_processing_results/building_predictions_merged_shape_file.shp"
output_shapefile = "/workspace/input/Predictions/Building_predictions/building_predictions_shapefile_new/building_predictions_merged_shape_file_new.shp"

keep_b_polygons_intersecting_a(
    shapefile_a,
    shapefile_b,
    output_shapefile
)
