import os
import subprocess
import glob

def run_predictions():
    input_dir = "/workspace/input/Resampled_30cm"
    output_dir = "/workspace/input/Predictions/Building_predictions/pakka_house_predictions"
    # output_dir = "/workspace/output/Vegitation_predictions"
    # output_dir = '/workspace/input/Predictions/Cultivation_predictions'
    config_path = "/workspace/ML_Setup/config/config_v1.yaml"
    building_model_path = "dummy"

    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        print("❌ No TIFF files found in:", input_dir)
        return

    print(f"🔍 Found {len(tif_files)} TIFF files to process.\n")

    for tif in tif_files:
        file_name = os.path.basename(tif)
        print(f"\n🚀 Running prediction for: {file_name}")
        print("------------------------------------------------------------")

        cmd = [
            "python",
            "/workspace/ML_Setup/ensemble_triton_with_waterbody_test.py",
            "--input_image", tif,
            "--output_path", output_dir,
            "--config", config_path,
            "--building_model", building_model_path
        ]

        # cmd = [
        #     "python",
        #     "/workspace/ML_Setup/ensemble_triton_with_waterbody.py",
        #     "--input_image", tif,
        #     "--output_path", output_dir,
        #     "--config", config_path,
        #     "--building_model", building_model_path
        # ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # Stream logs live
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode == 0:
            print(f"✅ Completed: {file_name}")
        else:
            print(f"❌ Failed: {file_name} (Exit code {process.returncode})")

        print("------------------------------------------------------------\n")

    print("🎉 All files processed !!")

if __name__ == "__main__":
    run_predictions()
