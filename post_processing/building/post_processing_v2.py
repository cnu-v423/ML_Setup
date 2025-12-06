import rasterio
import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology
import argparse
import os

def process_building_mask_advanced(
    input_raster_path, 
    output_raster_path, 
    building_value=1, 
    close_kernel_size=3, 
    open_kernel_size=5, 
    min_object_size_pixels=50
):
    """
    Applies an advanced cleaning workflow to a building raster mask:
    1. Morphological Closing: To fill small gaps in buildings.
    2. Morphological Opening: To separate connected buildings.
    3. Small Object Removal: To eliminate noise.
    4. Hole Filling: To fill interior holes.

    Args:
        input_raster_path (str): Path to the input raster file.
        output_raster_path (str): Path to save the processed raster file.
        building_value (int): The pixel value representing buildings.
        close_kernel_size (int): Kernel size for the initial closing operation.
        open_kernel_size (int): Kernel size for the main opening (separation) operation.
        min_object_size_pixels (int): Minimum size in pixels for an object to be kept.
    """
    if not os.path.exists(input_raster_path):
        print(f"❌ Error: Input file not found at {input_raster_path}")
        return

    print("--- Starting Advanced Building Mask Processing ---")
    print(f"Reading raster: {input_raster_path}")

    try:
        with rasterio.open(input_raster_path) as src:
            original_mask = src.read(1)
            profile = src.profile.copy()

            print(f"Isolating pixels with value '{building_value}'...")
            binary_buildings = (original_mask == building_value).astype(np.uint8)
            initial_pixel_count = np.sum(binary_buildings)
            if initial_pixel_count == 0:
                print("⚠️ No building pixels found. Exiting.")
                return

            # --- Step 1: Morphological Closing (Solidify Shapes) ---
            print(f"Step 1: Closing gaps with a {close_kernel_size}x{close_kernel_size} kernel...")
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
            closed_mask = cv2.morphologyEx(binary_buildings, cv2.MORPH_CLOSE, close_kernel)

            # --- Step 2: Morphological Opening (Separate Buildings) ---
            print(f"Step 2: Separating buildings with a {open_kernel_size}x{open_kernel_size} kernel...")
            open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_size, open_kernel_size))
            opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, open_kernel)

            # --- Step 3: Remove Small Objects (Noise Removal) ---
            print(f"Step 3: Removing objects smaller than {min_object_size_pixels} pixels...")
            # skimage.morphology.remove_small_objects requires a boolean array
            cleaned_mask_bool = morphology.remove_small_objects(
                opened_mask.astype(bool), 
                min_size=min_object_size_pixels
            )
            cleaned_mask = cleaned_mask_bool.astype(np.uint8)

            # --- Step 4: Fill Interior Holes ---
            print("Step 4: Filling interior holes...")
            final_binary_mask = ndimage.binary_fill_holes(cleaned_mask).astype(np.uint8)
            
            # --- Reconstruct and Save ---
            print("Reconstructing final mask...")
            output_mask = original_mask.copy()
            output_mask[original_mask == building_value] = 0 # Clear old buildings
            output_mask[final_binary_mask == 1] = building_value # Add new ones
            
            profile.update(dtype=rasterio.uint8, compress='lzw')

            print(f"Saving processed mask to: {output_raster_path}")
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                dst.write(output_mask, 1)

            final_pixel_count = np.sum(final_binary_mask)
            print(f"✅ Processing complete! Pixel count changed from {initial_pixel_count:,} to {final_pixel_count:,}.")

    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced processing for building raster masks.")
    parser.add_argument("--input_raster", required=True, help="Input TIFF mask.")
    parser.add_argument("--output_raster", required=True, help="Output path for cleaned TIFF mask.")
    parser.add_argument("--building_value", type=int, default=1, help="Pixel value of buildings.")
    parser.add_argument("--open_kernel", type=int, default=5, help="Kernel size for separation. TRY TUNING THIS (e.g., 3, 5, 7).")
    parser.add_argument("--min_pixels", type=int, default=10, help="Minimum pixel area for a building. TRY TUNING THIS.")
    
    args = parser.parse_args()
    
    process_building_mask_advanced(
        input_raster_path=args.input_raster,
        output_raster_path=args.output_raster,
        building_value=args.building_value,
        open_kernel_size=args.open_kernel,
        min_object_size_pixels=args.min_pixels
    )