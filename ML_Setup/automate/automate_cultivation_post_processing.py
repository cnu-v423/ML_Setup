import os
import glob
import subprocess

def automate_post_processing():
    # Base directory where TIFF files are stored
    predictions_dir = "/workspace/output/cultivation_predictions"

    # Input shapefile (constant as per your requirement)
    input_shapefile = "/workspace/output/Cultivation_predictions/cultivation_prediction_shapefiles/field_wise_predictions_merged_shape_file.shp"

    # Output folder
    output_base_path = "/workspace/output/Cultivation_predictions/cultivation_prediction_shapefiles"

    threshold_min = "20"

    # Collect all .tif files except *_prob.tif
    tif_files = sorted(
        [f for f in glob.glob(os.path.join(predictions_dir, "*.tif"))
         if not f.endswith("_prob.tif")]
    )

    if not tif_files:
        print("❌ No TIFF files found for post-processing.")
        return

    print(f"🔍 Found {len(tif_files)} TIFF files for post-processing.\n")

    for tif_path in tif_files:
        file_name = os.path.basename(tif_path)

        print(f"\n🚀 Running post-processing for: {file_name}")
        print("------------------------------------------------------------")

        cmd = [
            "python",
            "/workspace/ML_Setup/post_processing/cultivation_post_processing/post_processing_percentage.py",
            "--prediction_tiff", tif_path,
            "--input_shapefile", input_shapefile,
            "--output_base_path", output_base_path,
            "--threshold_min", threshold_min,
        ]

        # Live logs streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            print(f"✅ Completed: {file_name}\n")
        else:
            print(f"❌ Failed: {file_name} (Exit code {process.returncode})\n")

        print("------------------------------------------------------------\n")

    print("🎉 All post-processing completed!")

if __name__ == "__main__":
    automate_post_processing()
