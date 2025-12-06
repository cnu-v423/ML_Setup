import numpy as np
import rasterio
from rasterio.transform import from_bounds
import cv2
from skimage import morphology, measure
from scipy import ndimage
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class BuildingSeparator:
    """
    Tool to separate connected buildings by shrinking (eroding) white pixels in binary images
    """
    
    def __init__(self):
        """Initialize the building separator"""
        pass
    
    def load_binary_raster(self, raster_path):
        """
        Load binary raster and ensure it's in proper format
        """
        print(f"Loading binary raster: {raster_path}")
        
        with rasterio.open(raster_path) as src:
            data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            profile = src.profile.copy()
            
            # Get pixel size for reference
            pixel_size_x = abs(transform[0])
            pixel_size_y = abs(transform[4])
            print(f"Pixel size: {pixel_size_x:.3f} x {pixel_size_y:.3f} units")
            
            # Ensure binary format (0 and 1)
            unique_values = np.unique(data)
            print(f"Unique values in input: {unique_values}")
            
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:
                binary_data = data.astype(np.uint8)
                print("✅ Perfect binary format (0=background, 1=buildings)")
            elif len(unique_values) == 2:
                min_val, max_val = unique_values
                binary_data = ((data == max_val) * 1).astype(np.uint8)
                print(f"✅ Converted to binary: {min_val}→0, {max_val}→1")
            else:
                # Handle multi-value or float data
                if data.max() <= 1.0:
                    binary_data = (data > 0.5).astype(np.uint8)
                else:
                    max_val = data.max()
                    binary_data = (data == max_val).astype(np.uint8)
                print(f"⚠️ Converted multi-value data to binary")
            
            print(f"Building pixels: {np.sum(binary_data):,} / {binary_data.size:,} ({np.sum(binary_data)/binary_data.size*100:.2f}%)")
            
        return binary_data, transform, crs, profile
    
    def create_erosion_kernel(self, kernel_size, kernel_shape='square'):
        """
        Create erosion kernel of specified size and shape
        
        Args:
            kernel_size (int): Size of the kernel (e.g., 3, 5, 7)
            kernel_shape (str): 'square', 'circle', 'cross', or 'diamond'
        """
        print(f"Creating {kernel_shape} kernel of size {kernel_size}x{kernel_size}")
        
        if kernel_shape == 'square':
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            
        elif kernel_shape == 'circle':
            kernel = np.zeros((kernel_size, kernel_size), np.uint8)
            center = kernel_size // 2
            radius = center
            y, x = np.ogrid[:kernel_size, :kernel_size]
            mask = (x - center)**2 + (y - center)**2 <= radius**2
            kernel[mask] = 1
            
        elif kernel_shape == 'cross':
            kernel = np.zeros((kernel_size, kernel_size), np.uint8)
            center = kernel_size // 2
            kernel[center, :] = 1  # Horizontal line
            kernel[:, center] = 1  # Vertical line
            
        elif kernel_shape == 'diamond':
            kernel = np.zeros((kernel_size, kernel_size), np.uint8)
            center = kernel_size // 2
            for i in range(kernel_size):
                for j in range(kernel_size):
                    if abs(i - center) + abs(j - center) <= center:
                        kernel[i, j] = 1
        else:
            raise ValueError("kernel_shape must be 'square', 'circle', 'cross', or 'diamond'")
        
        print(f"Kernel created with {np.sum(kernel)} active pixels")
        return kernel
    
    def smart_erosion(self, binary_data, target_erosion_px, kernel_shape='square', 
                     max_iterations=50, min_remaining_ratio=0.1):
        """
        Apply smart erosion that prevents complete pixel loss
        
        Args:
            binary_data: Input binary array
            target_erosion_px: Target erosion in pixels (e.g., 25 for 25-pixel shrinkage)
            kernel_shape: Shape of erosion kernel
            max_iterations: Maximum number of iterations to prevent infinite loops
            min_remaining_ratio: Minimum ratio of pixels to keep (default: 10%)
        """
        print(f"\n{'='*50}")
        print(f"SMART EROSION - TARGET: {target_erosion_px} PIXELS")
        print(f"{'='*50}")
        
        # Use smaller kernels with more iterations for controlled erosion
        if target_erosion_px <= 5:
            kernel_size = 3
            iterations = target_erosion_px
        elif target_erosion_px <= 15:
            kernel_size = 3
            iterations = target_erosion_px // 2 + target_erosion_px % 2
        elif target_erosion_px <= 30:
            kernel_size = 5
            iterations = target_erosion_px // 3 + 2
        else:
            kernel_size = 7
            iterations = target_erosion_px // 5 + 3
        
        # Limit iterations to prevent over-erosion
        iterations = min(iterations, max_iterations)
        
        print(f"Using kernel size: {kernel_size}x{kernel_size}, Max iterations: {iterations}")
        
        # Create erosion kernel
        kernel = self.create_erosion_kernel(kernel_size, kernel_shape)
        
        # Track progress
        eroded_data = binary_data.copy()
        original_pixels = np.sum(binary_data)
        min_pixels = int(original_pixels * min_remaining_ratio)
        
        print(f"Original building pixels: {original_pixels:,}")
        print(f"Minimum pixels to preserve: {min_pixels:,} ({min_remaining_ratio*100:.1f}%)")
        
        # Apply erosion iteratively with safety checks
        for i in range(iterations):
            # Test erosion first
            test_eroded = cv2.erode(eroded_data, kernel, iterations=1)
            test_pixels = np.sum(test_eroded)
            
            # Safety check: don't erode if too few pixels would remain
            if test_pixels < min_pixels:
                print(f"⚠️ Stopping erosion at iteration {i+1} to preserve {np.sum(eroded_data):,} pixels")
                break
            
            # Apply the erosion
            eroded_data = test_eroded
            current_pixels = np.sum(eroded_data)
            pixels_lost = original_pixels - current_pixels
            
            print(f"Iteration {i+1:2d}: {current_pixels:,} pixels remaining (lost: {pixels_lost:,})")
            
            # Stop if no change occurred
            if current_pixels == np.sum(eroded_data):
                print("ℹ️ No more pixels to erode")
                break
        
        final_pixels = np.sum(eroded_data)
        pixels_lost = original_pixels - final_pixels
        
        print(f"\n✅ Smart erosion completed!")
        print(f"Pixels lost: {pixels_lost:,} ({pixels_lost/original_pixels*100:.1f}%)")
        print(f"Remaining building pixels: {final_pixels:,}")
        
        return eroded_data
    
    def gentle_erosion(self, binary_data, erosion_px, kernel_shape='square'):
        """
        Apply gentle erosion with very small kernels
        
        Args:
            binary_data: Input binary array
            erosion_px: Erosion amount in pixels
            kernel_shape: Shape of erosion kernel
        """
        print(f"\n{'='*50}")
        print(f"GENTLE EROSION - {erosion_px} PIXELS")
        print(f"{'='*50}")
        
        # Always use 3x3 kernel for gentle erosion
        kernel = self.create_erosion_kernel(3, kernel_shape)
        
        original_pixels = np.sum(binary_data)
        print(f"Original building pixels: {original_pixels:,}")
        
        # Apply multiple iterations with 3x3 kernel
        eroded_data = binary_data.copy()
        
        for i in range(erosion_px):
            # Check if we still have pixels to erode
            if np.sum(eroded_data) == 0:
                print(f"⚠️ All pixels eroded at iteration {i+1}")
                break
                
            eroded_data = cv2.erode(eroded_data, kernel, iterations=1)
            current_pixels = np.sum(eroded_data)
            
            if i % 5 == 0 or i == erosion_px - 1:  # Print every 5 iterations
                print(f"Iteration {i+1:2d}: {current_pixels:,} pixels remaining")
        
        final_pixels = np.sum(eroded_data)
        pixels_lost = original_pixels - final_pixels
        
        print(f"✅ Gentle erosion completed!")
        print(f"Pixels lost: {pixels_lost:,} ({pixels_lost/original_pixels*100:.1f}%)")
        print(f"Remaining building pixels: {final_pixels:,}")
        
        return eroded_data
    
    def morphological_opening(self, binary_data, kernel_size=5, kernel_shape='square'):
        """
        Use morphological opening (erosion + dilation) to separate buildings
        This preserves building size while separating connected components
        """
        print(f"\n{'='*50}")
        print(f"MORPHOLOGICAL OPENING - KERNEL SIZE: {kernel_size}")
        print(f"{'='*50}")
        
        kernel = self.create_erosion_kernel(kernel_size, kernel_shape)
        
        original_pixels = np.sum(binary_data)
        print(f"Original building pixels: {original_pixels:,}")
        
        # Apply opening (erosion followed by dilation)
        opened_data = cv2.morphologyEx(binary_data, cv2.MORPH_OPEN, kernel)
        
        final_pixels = np.sum(opened_data)
        pixels_lost = original_pixels - final_pixels
        
        print(f"✅ Morphological opening completed!")
        print(f"Pixels lost: {pixels_lost:,} ({pixels_lost/original_pixels*100:.1f}%)")
        print(f"Remaining building pixels: {final_pixels:,}")
        
        return opened_data
    
    
    def save_eroded_raster(self, eroded_data, output_path, transform, crs, profile):
        """Save the eroded binary raster to file"""
        print(f"Saving eroded raster to: {output_path}")
        
        # Update profile for output
        profile.update({
            'dtype': 'uint8',
            'count': 1,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path + 'after_erosion.tif', 'w', **profile) as dst:
            dst.write(eroded_data, 1)
        
        print(f"✅ Eroded raster saved successfully!")
    
    def analyze_separation(self, original_data, eroded_data):
        """
        Analyze how well the erosion separated buildings
        """
        print(f"\n{'='*50}")
        print("SEPARATION ANALYSIS")
        print(f"{'='*50}")
        
        # Count connected components before and after
        original_labels = measure.label(original_data, connectivity=2)
        eroded_labels = measure.label(eroded_data, connectivity=2)
        
        original_count = np.max(original_labels)
        eroded_count = np.max(eroded_labels)
        
        print(f"Connected components BEFORE: {original_count}")
        print(f"Connected components AFTER:  {eroded_count}")
        
        if eroded_count > original_count:
            print(f"✅ Successfully separated {eroded_count - original_count} building groups!")
        elif eroded_count == original_count:
            print("ℹ️ No new separations (buildings may have been already separate)")
        else:
            print("⚠️ Some buildings may have been completely eroded away")
        
        return original_count, eroded_count
    
    def process_building_separation(self, input_raster_path, output_raster_path, 
                                  erosion_pixels=10, kernel_shape='square',
                                  method='smart', save=True):
        """
        Complete pipeline for building separation with safety measures
        
        Args:
            input_raster_path: Input binary TIFF file
            output_raster_path: Output eroded TIFF file  
            erosion_pixels: Amount of erosion in pixels (start with 5-15)
            kernel_shape: 'square', 'circle', 'cross', or 'diamond'
            method: 'smart', 'gentle', or 'opening'
            preview_result: Show before/after comparison
        """
        try:
            # Step 1: Load binary raster
            binary_data, transform, crs, profile = self.load_binary_raster(input_raster_path)
            
            # Safety check: ensure we have building pixels
            if np.sum(binary_data) == 0:
                print("No building pixels found in input raster!")
                return None
            
            # Step 2: Apply appropriate erosion method
            if method == 'smart':
                eroded_data = self.smart_erosion(binary_data, erosion_pixels, kernel_shape)
            elif method == 'gentle':
                eroded_data = self.gentle_erosion(binary_data, erosion_pixels, kernel_shape)
            elif method == 'opening':
                kernel_size = max(3, min(erosion_pixels, 15))  # Limit kernel size
                eroded_data = self.morphological_opening(binary_data, kernel_size, kernel_shape)
            else:
                raise ValueError("Method must be 'smart', 'gentle', or 'opening'")
            
            # Safety check: ensure we still have pixels after processing
            if np.sum(eroded_data) == 0:
                print("All building pixels were eroded away! Try smaller erosion_pixels value.")
                return None
            
            # Step 3: Analyze separation results
            self.analyze_separation(binary_data, eroded_data)
            
        
            # Step 4: Save eroded raster
            if (save):
                os.makedirs(os.path.dirname(output_raster_path), exist_ok=True)
                self.save_eroded_raster(eroded_data, output_raster_path, transform, crs, profile)
            
                print(f"\n🎉 Building separation completed successfully!")
                print(f"Input:  {input_raster_path}")
                print(f"Output: {output_raster_path}")
            
            return eroded_data
            
        except Exception as e:
            print(f"❌ Error in building separation: {e}")
            return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Separate connected buildings by erosion')
    parser.add_argument('--input_tiff', required=True, help='Input binary TIFF file')
    parser.add_argument('--output_tiff', required=True, help='Output eroded TIFF file')
    parser.add_argument('--erosion_pixels', type=int, default=15, 
                       help='Erosion amount in pixels (recommended: 5-15, default: 10)')
    parser.add_argument('--kernel_shape', choices=['square', 'circle', 'cross', 'diamond'],
                       default='square', help='Erosion kernel shape (default: square)')
    parser.add_argument('--method', choices=['smart', 'gentle', 'opening'],
                       default='smart', help='Erosion method (default: smart)')
    
    args = parser.parse_args()
    
    separator = BuildingSeparator()
    
    result = separator.process_building_separation(
        input_raster_path=args.input_tiff,
        output_raster_path=args.output_tiff,
        erosion_pixels=args.erosion_pixels,
        kernel_shape=args.kernel_shape,
        method=args.method,
        save=True
    )
    
    if result is not None:
        print("\nBuilding separation pipeline completed!")
    else:
        print("\nBuilding separation pipeline failed!")