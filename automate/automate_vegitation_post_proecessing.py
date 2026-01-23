import os
import glob
import subprocess
import sys

# ---------------------------------------------------------
# CONFIG (SINGLE OUTPUT BASE PATH)
# ---------------------------------------------------------
PREDICTIONS_DIR = "/workspace/input/Predictions/Vegitation_predictions"
DEM_DIR = "/workspace/input/DEM"

OUTPUT_BASE_PATH = "/workspace/input/Predictions/Vegitation_predictions/post_process_shapefiles_after_resolved_tile_issue"

POST_PROCESS_SCRIPT = "/workspace/ML_Setup/post_processing/building/post_processing_v4.py"
NORMALIZE_SCRIPT = "/workspace/ML_Setup/post_processing/vegitation/normalize_dem_heights.py"
CLASSIFY_SCRIPT = "/workspace/ML_Setup/post_processing/vegitation/classify_trees.py"


# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------
def run_command(cmd, title):
    print(f"\n🚀 {title}")
    print("-" * 80)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )

    for line in process.stdout:
        print(line, end="")

    process.wait()

    if process.returncode != 0:
        print(f"\n❌ FAILED: {title}")
        sys.exit(process.returncode)

    print(f"\n✅ SUCCESS: {title}")
    print("-" * 80)


def find_matching_dem(predicted_tif_name):
    """
    Example:
        Penumaka_10.tif → matches → *penumaka*.tif
    """
    village_key = predicted_tif_name.split("_")[0].lower()

    for dem in glob.glob(os.path.join(DEM_DIR, "*.tif")):
        if village_key in os.path.basename(dem).lower():
            return dem

    return None


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def automate_post_processing():

    tif_files = sorted(
        f for f in glob.glob(os.path.join(PREDICTIONS_DIR, "*.tif"))
        if not f.endswith("_prob.tif")
    )

    if not tif_files:
        print("❌ No predicted TIFF files found.")
        return

    print(f"🔍 Found {len(tif_files)} TIFF files")

    for tif_path in tif_files:

        file_name = os.path.basename(tif_path)
        base_name = os.path.splitext(file_name)[0]

        ALLOWED_FILES = {
            # "Kuragallu_10",
            # "Lingayapalem_10",
            # "Malkapuram_10",
            # "Mandadam_10",
            # "Mangalagiri_10",
            # "Nekkallu_10",
            # "Nelapadu_10",
            # "Nowluru_10",
            # "Penumaka_10",
            # "Rayapudi_10",
            # "Sakhamuru_10",
            # "Tadepalli_10",
            # "Tulluru_10",
            # "Uddandarayanipalem_10",
            # "Undavalli_10",
            # "Velagapudi_10",
            "Venkatapalem_10"
        }

        if base_name not in ALLOWED_FILES:
            print(f"Skipping post-processing for {file_name} as this is already Done.")
            continue

        print(f"\n\n================ PROCESSING: {file_name} ================\n")

        # -----------------------------------------------------
        # STEP 1: POST PROCESSING
        # -----------------------------------------------------
        run_command(
            [
                "python",
                POST_PROCESS_SCRIPT,
                "--predicted_tiff", tif_path,
                "--output_path", OUTPUT_BASE_PATH
            ],
            title="Post Processing (post_processing_v4.py)"
        )

        # Expected outputs
        shapefile_dir = os.path.join(OUTPUT_BASE_PATH, base_name)
        shapefile_path = os.path.join(shapefile_dir, f"{base_name}.shp")

        if not os.path.exists(shapefile_path):
            print(f"❌ Shapefile not found: {shapefile_path}")
            sys.exit(1)

        # -----------------------------------------------------
        # STEP 2: FIND MATCHING DEM
        # -----------------------------------------------------
        dem_tif = find_matching_dem(file_name)

        if dem_tif is None:
            print(f"❌ No matching DEM found for {file_name}")
            sys.exit(1)

        print(f"🗺️  Matched DEM: {os.path.basename(dem_tif)}")

        heights_tif = os.path.join(
            shapefile_dir, f"{base_name}_heights.tif"
        )

        # -----------------------------------------------------
        # STEP 3: NORMALIZE DEM HEIGHTS
        # -----------------------------------------------------
        run_command(
            [
                "python",
                NORMALIZE_SCRIPT,
                "--dem_path", dem_tif,
                "--shapefile_path", shapefile_path,
                "--output_path", heights_tif
            ],
            title="Normalize DEM Heights (normalize_dem_heights.py)"
        )

        # -----------------------------------------------------
        # STEP 4: CLASSIFY TREES
        # -----------------------------------------------------
        classified_tif = os.path.join(
            shapefile_dir, f"{base_name}_classified_trees.shp"
        )

        run_command(
            [
                "python",
                CLASSIFY_SCRIPT,
                "--dem_path", heights_tif,
                "--shapefile_path", shapefile_path,
                "--output_path", classified_tif
            ],
            title="Tree Classification (classify_trees.py)"
        )

        print(f"\n🎯 COMPLETED PIPELINE FOR: {file_name}")

        # -----------------------------------------------------
        # CLEANUP: REMOVE INTERMEDIATE TIFF FILES
        # -----------------------------------------------------
        cleanup_files = [
            f"{base_name}_heights.tif",
            # f"{base_name}_heights_classified.tif",
            f"{base_name}_heights_ground.tif",
        ]

        for fname in cleanup_files:
            fpath = os.path.join(shapefile_dir, fname)
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    print(f"🧹 Removed intermediate file: {fname}")
                except Exception as e:
                    print(f"⚠️ Failed to remove {fname}: {e}")
            else:
                print(f"ℹ️ File not found (skip): {fname}")


    print("\n🎉 ALL FILES PROCESSED SUCCESSFULLY 🎉")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    automate_post_processing()
