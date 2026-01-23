import geopandas as gpd
import os
import glob

# --------------------------------------------------
# BASE INPUT FOLDER
# --------------------------------------------------
BASE_FOLDER = "/workspace/input/Predictions/Vegitation_predictions/post_process_shapefiles_after_resolved_tile_issue"

# --------------------------------------------------
# PROCESS EACH SUBFOLDER
# --------------------------------------------------
for folder_name in os.listdir(BASE_FOLDER):
    folder_path = os.path.join(BASE_FOLDER, folder_name)

    if not os.path.isdir(folder_path):
        continue

    # Find *_classified_trees.shp in this folder
    shp_files = glob.glob(os.path.join(folder_path, "*_classified_trees.shp"))

    if not shp_files:
        print(f"⚠️ No classified trees shapefile found in {folder_name}")
        continue

    input_shp = shp_files[0]

    output_shp = input_shp.replace(
        "_classified_trees.shp",
        "_classified_trees_dissolved.shp"
    )

    print(f"\n🚀 Processing: {input_shp}")

    # -----------------------------------------
    # READ
    # -----------------------------------------
    gdf = gpd.read_file(input_shp)

    # -----------------------------------------
    # FIX GEOMETRY
    # -----------------------------------------
    gdf["geometry"] = gdf.geometry.buffer(0)

    # -----------------------------------------
    # DISSOLVE ALL
    # -----------------------------------------
    dissolved = gdf.dissolve()

    # -----------------------------------------
    # EXPLODE → SINGLE POLYGONS
    # -----------------------------------------
    dissolved = dissolved.explode(index_parts=False)

    # -----------------------------------------
    # SAVE
    # -----------------------------------------
    dissolved.to_file(
        output_shp,
        driver="ESRI Shapefile"
    )

    print(f"✅ Saved: {output_shp}")

print("\n🎯 All folders processed successfully")
