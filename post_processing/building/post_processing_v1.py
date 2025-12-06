import rasterio
import numpy as np
import cv2
from scipy import ndimage
import argparse
import os

def process_building_mask(input_raster_path, output_raster_path, building_value=1, kernel_size=3):
    """
    Applies post processing and fills interior holes for a specific
    pixel value in a raster mask.

    Args:
        input_raster_path (str): Path to the input raster file.
        output_raster_path (str): Path to save the processed raster file.
        building_value (int): The pixel value representing buildings.
        kernel_size (int): The size of the square kernel for the opening operation.
                           Should be an odd number (e.g., 3, 5).
    """
    if not os.path.exists(input_raster_path):
        print(f"❌ Error: Input file not found at {input_raster_path}")
        return

    print(f"--- Starting Building Mask Processing ---")
    print(f"Reading raster: {input_raster_path}")

    try:
        with rasterio.open(input_raster_path) as src:
            original_mask = src.read(1)
            profile = src.profile.copy()

            # --- Step 1: Create a binary mask for the specified building value ---
            # This isolates the pixels we want to work on (buildings) from all others.
            print(f"Isolating pixels with value '{building_value}' for processing...")
            binary_buildings = (original_mask == building_value).astype(np.uint8)
            
            initial_pixel_count = np.sum(binary_buildings)
            if initial_pixel_count == 0:
                print(f"⚠️ Warning: No pixels with value '{building_value}' found. Output will be same as input.")
                # If no buildings, we can just copy the file or exit
                # For simplicity, we'll continue and it will just write an identical file.
            else:
                print(f"Found {initial_pixel_count:,} building pixels to process.")

            # --- Step 2: Apply Morphological Opening ---
            # This is the core operation to separate slightly connected components.
            # Erosion removes thin connections, then Dilation restores the size.
            print(f"Applying morphological opening with a {kernel_size}x{kernel_size} kernel...")
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            opened_mask = cv2.morphologyEx(binary_buildings, cv2.MORPH_OPEN, kernel)

            # --- Step 3: Fill Holes Inside Buildings ---
            # scipy.ndimage.binary_fill_holes is perfect for this. By definition,
            # it only fills holes completely surrounded by `True` (or 1) values.
            # It will NOT fill indentations or gaps on the outer edges of a shape.
            print("Filling interior holes in building shapes...")
            final_binary_mask = ndimage.binary_fill_holes(opened_mask).astype(np.uint8)
            
            final_pixel_count = np.sum(final_binary_mask)
            print(f"Processing changed pixel count from {initial_pixel_count:,} to {final_pixel_count:,}.")

            # --- Step 4: Reconstruct the final multi-value mask ---
            # We create a new mask that preserves the original non-building values
            # and inserts our newly processed building pixels.
            print("Reconstructing final mask with original background values...")
            # Start with a copy of the original data
            output_mask = original_mask.copy()
            # First, set all original building locations to background (0)
            # This prevents old pixel fragments from remaining if shapes shrank.
            output_mask[original_mask == building_value] = 0
            # Now, "paste" the new, processed building pixels back onto the mask
            output_mask[final_binary_mask == 1] = building_value
            
            profile.update(dtype=rasterio.uint8, compress='lzw')

            print(f"Saving processed mask to: {output_raster_path}")
            with rasterio.open(output_raster_path, 'w', **profile) as dst:
                dst.write(output_mask, 1)

            print(f"✅ Processing complete!")

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a building raster mask by applying morphological opening and filling holes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_raster", required=True, help="Path to the input multi-value TIFF mask.")
    parser.add_argument("--output_raster", required=True, help="Path to save the processed TIFF mask.")
    parser.add_argument("--building_value", type=int, default=1, help="The integer pixel value that represents buildings.")
    parser.add_argument("--kernel_size", type=int, default=3, help="The size of the kernel for morphological opening (e.g., 3 for 3x3).")
    
    args = parser.parse_args()
    
    process_building_mask(
        input_raster_path=args.input_raster,
        output_raster_path=args.output_raster,
        building_value=args.building_value,
        kernel_size=args.kernel_size
    )