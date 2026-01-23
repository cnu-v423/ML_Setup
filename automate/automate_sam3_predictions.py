import os
import subprocess
import glob

def run_predictions():
    input_dir = "/workspace/input/resampled_10cm_new"
    output_dir = "/workspace/input/Predictions/Vegitation_predictions"

    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        print("❌ No TIFF files found in:", input_dir)
        return

    print(f"🔍 Found {len(tif_files)} TIFF files to process.\n")

    for tif in tif_files:
        file_name = os.path.basename(tif)                 # e.g. image_01.tif
        name_no_ext = os.path.splitext(file_name)[0]      # e.g. image_01

        ALLOWED_FILES = {
            # "Kuragallu_10",
            # "Lingayapalem_10",
            # "Malkapuram_10",
            # "Mandadam_10",
            # "Mangalagiri_10",
            # "Nekkallu_10",
            # "Nelapadu_10",
            "Undavalli_10",
            "Velagapudi_10",
            "Venkatapalem_10"
        }

        if name_no_ext not in ALLOWED_FILES:
            print(f"Skipping prediction for {file_name} as this is already predicted")
            continue

        print(f"\n🚀 Running prediction for: {file_name}")
        print("------------------------------------------------------------")

        # 🔥 output_dir + file_name.tif
        output_tif_path = os.path.join(output_dir, f"{name_no_ext}.tif")


        cmd = [
            "python",
            "/workspace/ML_Setup/sam3_v1.py",
            "--input_path", tif,
            "--output_path", output_tif_path,
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
