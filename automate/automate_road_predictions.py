# run_predictions_for_corrected_inference.py
import os
import subprocess
import glob

def run_predictions(input_dir, output_dir, model_checkpoint, script_path, dry_run=False):
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    if not tif_files:
        print("❌ No TIFF files found in:", input_dir)
        return
    print(f"🔍 Found {len(tif_files)} TIFF files to process.\n")
    for tif in tif_files:
        file_name = os.path.basename(tif)
        print(f"\n🚀 Running prediction for: {file_name}")
        cmd = [
            "python", script_path,
            "--input_image", tif,
            "--output_dir", output_dir,
            "--model_checkpoint", model_checkpoint,
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode == 0:
            print(f"✅ Completed: {file_name}")
        else:
            print(f"❌ Failed: {file_name} (Exit code {process.returncode})")
    print("\n🎉 All files processed !!")

if __name__ == "__main__":
    input_dir = "/media/vassardigitalgpu/One Touch/Resampled_30cm"
    output_dir = "/media/vassardigitalgpu/One Touch/Predictions/Roads_predictions"
    model_checkpoint = "/mnt/external/APCRDA/ML/mahipal_road_detection/model_improve/fine_tune/latest_dataset_finetuned_best_model_150_epoch_v1.pth"
    script_path = "/mnt/external/APCRDA/ML/mahipal_road_detection/model_improve/corrected_inference_script.py"
    run_predictions(input_dir, output_dir, model_checkpoint, script_path, dry_run=True)