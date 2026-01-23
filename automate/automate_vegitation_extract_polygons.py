import os
import glob
import subprocess
import sys

# ---------------------------------------------------------
# CONFIG (SINGLE OUTPUT BASE PATH)
# ---------------------------------------------------------
SHAPE_FILES = "/workspace/input/Predictions/Vegitation_predictions/post_process_shapefiles_after_resolved_tile_issue"
EXTRACT_POLYGONS_SCRIPT = "/workspace/ML_Setup/post_processing/vegitation/extract_polygons.py"

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
        f for f in glob.glob(
            os.path.join(SHAPE_FILES, "**", "*.tif"),
            recursive=True
        )
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
            "Abbarajupalem_10_heights_classified",
            "Ainavolu_10_heights_classified",
            "Anantavaram_10_heights_classified",
            "Borupalem_10_heights_classified",
            "Dondapadu_10_heights_classified",
            "Kondamarajupalem_10_heights_classified",
            "Krishnayyapalem_10_heights_classified",
            "Kuragallu_10_heights_classified",
            "Lingayapalem_10_heights_classified",
            "Malkapuram_10_heights_classified",
            # "Mandadam_10_heights_classified", ## It is failed
            'Mangalagiri_10_heights_classified',
            'Nekkallu_10_heights_classified',
            'Nelapadu_10_heights_classified',
            'Nidamarru_10_heights_classified',
            'Nowluru_10_heights_classified',
            'Penumaka_10_heights_classified',
            'Pichikalapalem_10_heights_classified',
            # 'Rayapudi_10_heights_classified', ## It is failed,
            'Sakhamuru_10_heights_classified',
            'Tadepalli_10_heights_classified',
            # 'Tulluru_10_heights_classified', ## It is failed
            'Uddandarayanipalem_10_heights_classified'

        }

        if base_name in ALLOWED_FILES:
            print(f"Skipping post-processing for {file_name} as this is already Done.")
            continue

        print(f"\n\n================ PROCESSING: {file_name} ================\n")

        # -----------------------------------------------------
        # STEP 1: POST PROCESSING
        # -----------------------------------------------------
        output_path = os.path.dirname(tif_path)
        run_command(
            [
                "python",
                EXTRACT_POLYGONS_SCRIPT,
                "--predicted_tiff", tif_path,
                "--output_path", output_path
            ],
            title="Post Processing (extract_polygons.py)"
        )


    print("\n🎉 ALL FILES PROCESSED SUCCESSFULLY 🎉")


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    automate_post_processing()
