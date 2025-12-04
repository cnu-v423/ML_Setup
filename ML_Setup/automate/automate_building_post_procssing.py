import os
import glob
import subprocess

def automate_post_processing():
    # Base directory where TIFF files are stored
    predictions_dir = "/workspace/output/Building_predictions"

    # Input shapefile (constant as per your requirement)
    dsm_folder = "/media/srinivas/One Touch/DEM/"


    # Output folder
    output_base_path = "/workspace/output/Building_predictions/building_predictions_shapefile"


    # Collect all .tif files except *_prob.tif
    tif_files = sorted(
        [f for f in glob.glob(os.path.join(predictions_dir, "*.tif"))
         if not f.endswith("_prob.tif")]
    )

    dsm_files = glob.glob(os.path.join(dsm_folder, "*.tif"))


    if not tif_files:
        print("❌ No TIFF files found for post-processing.")
        return

    print(f"🔍 Found {len(tif_files)} TIFF files for post-processing.\n")

    for tif_path in tif_files:
        
        file_name = os.path.basename(tif_path).lower()

        print(f"\n🚀 Running post-processing for: {file_name}")
        print("------------------------------------------------------------")

        # -----------------------------------------
        # FIND MATCHING DSM FILE (CASE INSENSITIVE)
        # -----------------------------------------
        matched_dsm = None
        for dsm_path in dsm_files:
            dsm_name = os.path.basename(dsm_path).lower()

            if any(keyword in dsm_name for keyword in file_name.split("_")):
                matched_dsm = dsm_path
                break

        if matched_dsm is None:
            print(f"❌ No matching DSM found for: {file_name}")
            continue

        # print(f"✅ DSM Selected: {matched_dsm}")

        cmd = [
            "python",
            "/workspace/ML_Setup/post_processing/building/building_post_processing.py",
            "--predicted_tiff", tif_path,
            "--output_path", output_base_path,
            "--dsm_path", matched_dsm
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



############################## OLD CODE ##############################################


# import os
# import glob
# import subprocess

# def automate_post_processing():
#     # Base directory where TIFF files are stored
#     predictions_dir = "/workspace/output/Building_predictions"

#     # Input shapefile (constant as per your requirement)

#     # Output folder
#     output_base_path = "/workspace/output/Building_predictions/building_predictions_shapefile"


#     # Collect all .tif files except *_prob.tif
#     tif_files = sorted(
#         [f for f in glob.glob(os.path.join(predictions_dir, "*.tif"))
#          if not f.endswith("_prob.tif")]
#     )

#     if not tif_files:
#         print("❌ No TIFF files found for post-processing.")
#         return

#     print(f"🔍 Found {len(tif_files)} TIFF files for post-processing.\n")

#     for tif_path in tif_files:
#         file_name = os.path.basename(tif_path)

#         print(f"\n🚀 Running post-processing for: {file_name}")
#         print("------------------------------------------------------------")

#         cmd = [
#             "python",
#             "/workspace/ML_Setup/post_processing/building/building_post_processing.py",
#             "--predicted_tiff", tif_path,
#             "--output_path", output_base_path
#         ]

#         # Live logs streaming
#         process = subprocess.Popen(
#             cmd,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             universal_newlines=True
#         )

#         for line in process.stdout:
#             print(line, end="")

#         process.wait()

#         if process.returncode == 0:
#             print(f"✅ Completed: {file_name}\n")
#         else:
#             print(f"❌ Failed: {file_name} (Exit code {process.returncode})\n")

#         print("------------------------------------------------------------\n")

#     print("🎉 All post-processing completed!")

# if __name__ == "__main__":
#     automate_post_processing()
