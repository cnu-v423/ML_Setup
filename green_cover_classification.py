# import numpy as np
# import rasterio
# from rasterio.windows import Window
# from scipy import ndimage
# from scipy.ndimage import generic_filter
# from skimage.feature import graycomatrix, graycoprops
# from skimage.measure import label, regionprops
# from skimage.filters import sobel
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# import gc
# import psutil


# class GreenCoverClassifier:
#     """
#     Classifies green cover vs agriculture using RGB imagery, DEM, and height analysis
#     """

#     def __init__(self, rgb_path, dem_path, max_memory_gb=6.0, tile_size=1024):
#         """
#         Initialize with paths to RGB and DEM TIFFs

#         Parameters:
#         -----------
#         rgb_path : str
#             Path to RGB TIFF file
#         dem_path : str
#             Path to DEM TIFF file
#         max_memory_gb : float
#             Maximum memory to use in GB (default: 6.0)
#         tile_size : int
#             Size of tiles for processing large images (default: 1024)
#         """
#         self.rgb_path = rgb_path
#         self.dem_path = dem_path
#         self.max_memory_gb = max_memory_gb
#         self.tile_size = tile_size
#         self.use_tiling = False

#         # Check available memory
#         available_gb = psutil.virtual_memory().available / (1024 ** 3)
#         print(f"Available RAM: {available_gb:.2f} GB")
#         print(f"Max memory allocation: {max_memory_gb:.2f} GB")

#     def load_data(self):
#         """Load RGB and DEM data with memory optimization"""
#         print("Loading RGB data...")
#         with rasterio.open(self.rgb_path) as src:
#             self.rgb_meta = src.meta
#             self.transform = src.transform
#             self.height = src.height
#             self.width = src.width

#             # Estimate memory requirement
#             pixels = self.height * self.width
#             # RGB (3 bands) + DEM + intermediate arrays
#             estimated_gb = (pixels * 4 * 10) / (1024 ** 3)  # 10 arrays, 4 bytes each

#             print(f"Image size: {self.height} x {self.width}")
#             print(f"Estimated memory needed: {estimated_gb:.2f} GB")

#             if estimated_gb > self.max_memory_gb:
#                 print(f"WARNING: Image too large for available memory!")
#                 print(f"Enabling tiled processing with tile size: {self.tile_size}")
#                 self.use_tiling = True
#                 # Don't load full image, will process in tiles
#                 self.rgb = None
#             else:
#                 # Load full image
#                 self.rgb = src.read().astype(np.float32)  # Use float32 instead of float64
#                 if self.rgb.max() > 1:
#                     self.rgb = self.rgb / 255.0

#         print("Loading DEM data...")
#         with rasterio.open(self.dem_path) as src:
#             self.dem_meta = src.meta
#             if not self.use_tiling:
#                 self.dem = src.read(1).astype(np.float32)
#             else:
#                 self.dem = None

#         if not self.use_tiling:
#             print(f"RGB shape: {self.rgb.shape}")
#             print(f"DEM shape: {self.dem.shape}")

#         # Force garbage collection
#         gc.collect()

#     def calculate_vegetation_mask(self, threshold=0.1):
#         """
#         Calculate vegetation mask using Excess Green Index (ExG)
#         Memory-efficient version
#         """
#         print("Calculating vegetation mask...")
#         r = self.rgb[0]
#         g = self.rgb[1]
#         b = self.rgb[2]

#         # Excess Green Index - calculate in place to save memory
#         exg = 2 * g - r - b

#         # Create vegetation mask
#         self.veg_mask = (exg > threshold).astype(np.uint8)
#         print(f"Vegetation pixels: {np.sum(self.veg_mask)} ({100 * np.mean(self.veg_mask):.2f}%)")

#         del exg
#         gc.collect()

#         return self.veg_mask

#     def calculate_texture_variance(self, window_size=7):
#         """
#         Calculate texture variance using sliding window
#         Memory-efficient version
#         """
#         print(f"Calculating texture variance (window={window_size})...")

#         # Use green band for texture analysis
#         green = self.rgb[1]

#         # Use uniform filter for mean (more memory efficient)
#         mean = ndimage.uniform_filter(green, size=window_size, mode='reflect')
#         sqr_mean = ndimage.uniform_filter(green ** 2, size=window_size, mode='reflect')
#         texture_var = sqr_mean - mean ** 2

#         self.texture_variance = texture_var.astype(np.float32)
#         print(f"Texture variance range: {texture_var.min():.4f} to {texture_var.max():.4f}")

#         # Clean up
#         del mean, sqr_mean
#         gc.collect()

#         return texture_var

#     def calculate_glcm_texture(self, window_size=15, sample_rate=0.1):
#         """
#         Calculate GLCM texture features (contrast, homogeneity)
#         Uses sampling to reduce memory usage

#         Parameters:
#         -----------
#         window_size : int
#             Window size for GLCM calculation
#         sample_rate : float
#             Fraction of pixels to sample (0.1 = 10%)
#         """
#         print(f"Calculating GLCM texture features (sampling {sample_rate * 100}% of pixels)...")

#         green = self.rgb[1]
#         green_uint = (green * 255).astype(np.uint8)

#         h, w = green_uint.shape
#         contrast = np.zeros_like(green_uint, dtype=np.float32)
#         homogeneity = np.zeros_like(green_uint, dtype=np.float32)

#         half_win = window_size // 2

#         # Sample pixels to process
#         step = int(1 / sample_rate)

#         for i in range(half_win, h - half_win, step):
#             for j in range(half_win, w - half_win, step):
#                 window = green_uint[i - half_win:i + half_win + 1,
#                 j - half_win:j + half_win + 1]

#                 if window.size > 0:
#                     glcm = graycomatrix(window, [1], [0], 256,
#                                         symmetric=True, normed=True)
#                     c_val = graycoprops(glcm, 'contrast')[0, 0]
#                     h_val = graycoprops(glcm, 'homogeneity')[0, 0]

#                     # Fill neighborhood with same values
#                     contrast[max(0, i - step // 2):min(h, i + step // 2),
#                     max(0, j - step // 2):min(w, j + step // 2)] = c_val
#                     homogeneity[max(0, i - step // 2):min(h, i + step // 2),
#                     max(0, j - step // 2):min(w, j + step // 2)] = h_val

#         self.glcm_contrast = contrast
#         self.glcm_homogeneity = homogeneity

#         gc.collect()
#         return contrast, homogeneity

#     def calculate_slope(self):
#         """Calculate slope from DEM (in degrees)"""
#         print("Calculating slope from DEM...")

#         # Calculate gradients
#         dy, dx = np.gradient(self.dem)

#         # Calculate slope in degrees
#         slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))

#         self.slope = slope
#         print(f"Slope range: {slope.min():.2f}° to {slope.max():.2f}°")

#         return slope

#     def calculate_height_features(self):
#         """
#         Calculate vegetation height features from DEM
#         Memory-efficient version
#         """
#         print("Calculating height features...")

#         # Local height variation (roughness) - use uniform filter
#         mean = ndimage.uniform_filter(self.dem, size=9, mode='reflect')
#         sqr_mean = ndimage.uniform_filter(self.dem ** 2, size=9, mode='reflect')
#         height_std = np.sqrt(sqr_mean - mean ** 2).astype(np.float32)

#         # Relative height (deviation from local mean)
#         local_mean = ndimage.uniform_filter(self.dem, size=15, mode='reflect')
#         relative_height = (self.dem - local_mean).astype(np.float32)

#         self.height_std = height_std
#         self.relative_height = relative_height

#         print(f"Height std range: {height_std.min():.2f} to {height_std.max():.2f}")
#         print(f"Relative height range: {relative_height.min():.2f} to {relative_height.max():.2f}")

#         # Clean up
#         del mean, sqr_mean, local_mean
#         gc.collect()

#         return height_std, relative_height

#     def calculate_shape_metrics(self, max_regions=500):
#         """
#         Calculate shape regularity of vegetation patches
#         Memory-efficient version with region limit
#         """
#         print("Calculating shape metrics...")

#         # Label connected vegetation regions
#         labeled = label(self.veg_mask)
#         regions = regionprops(labeled)

#         # Sort by area and keep only largest regions to save memory
#         regions = sorted(regions, key=lambda x: x.area, reverse=True)[:max_regions]

#         # Create shape regularity map
#         shape_irregularity = np.zeros_like(self.veg_mask, dtype=np.float32)

#         for region in regions:
#             if region.area < 100:  # Skip very small regions
#                 continue

#             # Solidity and extent calculations
#             solidity = region.solidity
#             extent = region.extent

#             # Combine metrics
#             irregularity = 1 - (solidity * extent)

#             # Fill the region
#             coords = region.coords
#             shape_irregularity[coords[:, 0], coords[:, 1]] = irregularity

#         self.shape_irregularity = shape_irregularity
#         print(f"Shape irregularity calculated for {len(regions)} regions")

#         del labeled, regions
#         gc.collect()

#         return shape_irregularity

#     def classify_green_cover(self,
#                              texture_weight=0.30,
#                              shape_weight=0.25,
#                              height_weight=0.25,
#                              slope_weight=0.20,
#                              forest_threshold=0.5):
#         """
#         Classify green cover using weighted combination of features
#         Memory-efficient version
#         """
#         print("\nClassifying green cover...")

#         # Normalize all features to 0-1 range
#         def normalize(arr):
#             veg_values = arr[self.veg_mask > 0]
#             if len(veg_values) == 0:
#                 return np.zeros_like(arr, dtype=np.float32)
#             arr_min = np.percentile(veg_values, 1)  # Use percentile to avoid outliers
#             arr_max = np.percentile(veg_values, 99)
#             if arr_max - arr_min == 0:
#                 return np.zeros_like(arr, dtype=np.float32)
#             normalized = np.clip((arr - arr_min) / (arr_max - arr_min), 0, 1)
#             return normalized.astype(np.float32)

#         # Normalize features
#         print("Normalizing features...")
#         texture_norm = normalize(self.texture_variance)
#         shape_norm = self.shape_irregularity.astype(np.float32)
#         height_norm = normalize(self.height_std)
#         slope_norm = normalize(self.slope)

#         # Calculate combined score
#         print("Computing classification score...")
#         green_cover_score = (
#                 texture_weight * texture_norm +
#                 shape_weight * shape_norm +
#                 height_weight * height_norm +
#                 slope_weight * slope_norm
#         ).astype(np.float32)

#         # Apply only to vegetation areas
#         green_cover_score = green_cover_score * self.veg_mask

#         # Classify
#         self.green_cover_score = green_cover_score
#         self.green_cover_mask = ((green_cover_score > forest_threshold) & (self.veg_mask > 0)).astype(np.uint8)
#         self.agriculture_mask = ((green_cover_score <= forest_threshold) & (self.veg_mask > 0)).astype(np.uint8)

#         # Calculate statistics
#         total_veg = np.sum(self.veg_mask)
#         green_cover_pixels = np.sum(self.green_cover_mask)
#         agriculture_pixels = np.sum(self.agriculture_mask)

#         print(f"\nClassification Results:")
#         print(f"Total vegetation pixels: {total_veg}")
#         print(f"Green cover (forest/natural): {green_cover_pixels} ({100 * green_cover_pixels / total_veg:.1f}%)")
#         print(f"Agriculture: {agriculture_pixels} ({100 * agriculture_pixels / total_veg:.1f}%)")

#         # Clean up
#         del texture_norm, shape_norm, height_norm, slope_norm
#         gc.collect()

#         return self.green_cover_mask, self.agriculture_mask

#     def visualize_results(self, save_path=None, dpi=150):
#         """
#         Visualize classification results
#         Lower DPI for memory efficiency
#         """
#         print("\nGenerating visualization...")

#         # Downsample for visualization if image is too large
#         max_display_size = 2000
#         if self.height > max_display_size or self.width > max_display_size:
#             scale = max_display_size / max(self.height, self.width)
#             display_h = int(self.height * scale)
#             display_w = int(self.width * scale)
#             print(f"Downsampling for display: {display_h} x {display_w}")

#             from scipy.ndimage import zoom
#             zoom_factor = (display_h / self.height, display_w / self.width)

#             rgb_display = np.transpose(zoom(self.rgb, (1, zoom_factor[0], zoom_factor[1])), (1, 2, 0))
#             dem_display = zoom(self.dem, zoom_factor)
#             veg_display = zoom(self.veg_mask, zoom_factor, order=0)
#             texture_display = zoom(self.texture_variance, zoom_factor)
#             slope_display = zoom(self.slope, zoom_factor)
#             height_display = zoom(self.height_std, zoom_factor)
#             shape_display = zoom(self.shape_irregularity, zoom_factor)
#             score_display = zoom(self.green_cover_score, zoom_factor)
#             gc_display = zoom(self.green_cover_mask, zoom_factor, order=0)
#             ag_display = zoom(self.agriculture_mask, zoom_factor, order=0)
#         else:
#             rgb_display = np.transpose(self.rgb, (1, 2, 0))
#             dem_display = self.dem
#             veg_display = self.veg_mask
#             texture_display = self.texture_variance
#             slope_display = self.slope
#             height_display = self.height_std
#             shape_display = self.shape_irregularity
#             score_display = self.green_cover_score
#             gc_display = self.green_cover_mask
#             ag_display = self.agriculture_mask

#         fig, axes = plt.subplots(3, 3, figsize=(15, 15))

#         # RGB composite
#         axes[0, 0].imshow(rgb_display)
#         axes[0, 0].set_title('RGB Composite')
#         axes[0, 0].axis('off')

#         # DEM
#         im1 = axes[0, 1].imshow(dem_display, cmap='terrain')
#         axes[0, 1].set_title('DEM (Elevation)')
#         axes[0, 1].axis('off')
#         plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

#         # Vegetation mask
#         axes[0, 2].imshow(veg_display, cmap='Greens')
#         axes[0, 2].set_title('Vegetation Mask')
#         axes[0, 2].axis('off')

#         # Texture variance
#         im2 = axes[1, 0].imshow(texture_display, cmap='viridis')
#         axes[1, 0].set_title('Texture Variance')
#         axes[1, 0].axis('off')
#         plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

#         # Slope
#         im3 = axes[1, 1].imshow(slope_display, cmap='YlOrRd')
#         axes[1, 1].set_title('Slope (degrees)')
#         axes[1, 1].axis('off')
#         plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

#         # Height variation
#         im4 = axes[1, 2].imshow(height_display, cmap='plasma')
#         axes[1, 2].set_title('Height Variation')
#         axes[1, 2].axis('off')
#         plt.colorbar(im4, ax=axes[1, 2], fraction=0.046)

#         # Shape irregularity
#         im5 = axes[2, 0].imshow(shape_display * veg_display, cmap='coolwarm')
#         axes[2, 0].set_title('Shape Irregularity')
#         axes[2, 0].axis('off')
#         plt.colorbar(im5, ax=axes[2, 0], fraction=0.046)

#         # Green cover score
#         im6 = axes[2, 1].imshow(score_display, cmap='RdYlGn')
#         axes[2, 1].set_title('Green Cover Score')
#         axes[2, 1].axis('off')
#         plt.colorbar(im6, ax=axes[2, 1], fraction=0.046)

#         # Final classification
#         classification = np.zeros_like(veg_display, dtype=int)
#         classification[ag_display > 0] = 1
#         classification[gc_display > 0] = 2

#         cmap = ListedColormap(['black', 'yellow', 'darkgreen'])
#         axes[2, 2].imshow(classification, cmap=cmap, vmin=0, vmax=2)
#         axes[2, 2].set_title('Final Classification\n(Black=Non-veg, Yellow=Agri, Green=Forest)')
#         axes[2, 2].axis('off')

#         plt.tight_layout()

#         if save_path:
#             plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
#             print(f"Visualization saved to: {save_path}")

#         plt.show()
#         plt.close()

#         # Clean up
#         gc.collect()

#     def save_classification(self, output_path):
#         """Save classification result as GeoTIFF"""
#         print(f"\nSaving classification to: {output_path}")

#         # Create classification raster
#         # 0 = Non-vegetation, 1 = Agriculture, 2 = Green cover
#         classification = np.zeros_like(self.veg_mask, dtype=np.uint8)
#         classification[self.agriculture_mask] = 1
#         classification[self.green_cover_mask] = 2

#         # Update metadata
#         meta = self.rgb_meta.copy()
#         meta.update({
#             'count': 1,
#             'dtype': 'uint8',
#             'nodata': 0
#         })

#         # Write to file
#         with rasterio.open(output_path, 'w', **meta) as dst:
#             dst.write(classification, 1)

#         print("Classification saved successfully!")

#     def run_full_analysis(self, output_path='green_cover_classification.tif',
#                           viz_path='classification_results.png',
#                           skip_glcm=True):
#         """
#         Run complete analysis pipeline

#         Parameters:
#         -----------
#         skip_glcm : bool
#             Skip GLCM texture analysis to save memory (default: True)
#         """
#         print("=" * 60)
#         print("GREEN COVER CLASSIFICATION PIPELINE")
#         print(f"Memory limit: {self.max_memory_gb} GB")
#         print("=" * 60)

#         # Load data
#         self.load_data()

#         if self.use_tiling:
#             print("\nERROR: Image too large for available memory!")
#             print("Please either:")
#             print("1. Increase max_memory_gb parameter")
#             print("2. Clip/subsample your input images to a smaller area")
#             print("3. Use a machine with more RAM")
#             return None, None

#         # Calculate all features
#         self.calculate_vegetation_mask()
#         self.calculate_texture_variance()
#         self.calculate_slope()
#         self.calculate_height_features()
#         self.calculate_shape_metrics()

#         # Skip GLCM if memory is tight
#         if not skip_glcm:
#             self.calculate_glcm_texture(sample_rate=0.1)

#         # Classify
#         self.classify_green_cover()

#         # Visualize
#         self.visualize_results(save_path=viz_path)

#         # Save results
#         self.save_classification(output_path)

#         print("\n" + "=" * 60)
#         print("ANALYSIS COMPLETE!")
#         print("=" * 60)

#         return self.green_cover_mask, self.agriculture_mask


# # Example usage
# if __name__ == "__main__":
#     # Initialize classifier with 6GB memory limit
#     classifier = GreenCoverClassifier(
#         rgb_path='/workspace/input/Uddandarayanipalem.tif',
#         dem_path='/workspace/output/RB023_Uddandarayanipalem DEM_2.tif',
#         max_memory_gb=16.0
#     )

#     # Run full analysis
#     green_cover, agriculture = classifier.run_full_analysis(
#         output_path='/workspace/output/Green_Cover/green_cover_classification.tif',
#         viz_path='/workspace/output/Green_Cover/classification_results.png'
#     )

#     # Or run step by step with custom parameters
#     """
#     classifier.load_data()
#     classifier.calculate_vegetation_mask(threshold=0.15)
#     classifier.calculate_texture_variance(window_size=9)
#     classifier.calculate_slope()
#     classifier.calculate_height_features()
#     classifier.calculate_shape_metrics()

#     # Custom weights for classification
#     classifier.classify_green_cover(
#         texture_weight=0.35,
#         shape_weight=0.25,
#         height_weight=0.25,
#         slope_weight=0.15,
#         forest_threshold=0.45
#     )

#     classifier.visualize_results()
#     classifier.save_classification('output.tif')
#     """

















import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gc
import psutil


class HeightBasedGreenCoverClassifier:
    """
    Classifies green cover based on vegetation height derived from DEM analysis
    Memory-optimized for large files
    """

    def __init__(self, rgb_path, dem_path, max_memory_gb=6.0, tile_size=512):
        """
        Initialize with paths to RGB and DEM TIFFs

        Parameters:
        -----------
        rgb_path : str
            Path to RGB TIFF file
        dem_path : str
            Path to DEM TIFF file
        max_memory_gb : float
            Maximum memory to use in GB (default: 6.0)
        tile_size : int
            Size of tiles for processing large images (default: 512)
        """
        self.rgb_path = rgb_path
        self.dem_path = dem_path
        self.max_memory_gb = max_memory_gb
        self.tile_size = tile_size
        self.use_tiling = False

        # Check available memory
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"Available RAM: {available_gb:.2f} GB")
        print(f"Max memory allocation: {max_memory_gb:.2f} GB")
        
        self.needs_resampling = False  # Will be set in load_data_info()

    def load_data_info(self):
        """Load metadata without loading full arrays"""
        print("Analyzing data dimensions...")
        
        with rasterio.open(self.rgb_path) as src:
            self.rgb_meta = src.meta
            self.rgb_transform = src.transform
            self.rgb_height = src.height
            self.rgb_width = src.width
            self.rgb_crs = src.crs
            
        with rasterio.open(self.dem_path) as src:
            self.dem_meta = src.meta
            self.dem_transform = src.transform
            self.dem_height = src.height
            self.dem_width = src.width
            self.dem_crs = src.crs
            
        print(f"RGB size: {self.rgb_height} x {self.rgb_width}")
        print(f"DEM size: {self.dem_height} x {self.dem_width}")
        
        # Check if dimensions match
        if (self.rgb_height != self.dem_height) or (self.rgb_width != self.dem_width):
            print("\n⚠️  WARNING: RGB and DEM dimensions don't match!")
            print(f"   RGB: {self.rgb_height} x {self.rgb_width}")
            print(f"   DEM: {self.dem_height} x {self.dem_width}")
            print("   Will resample DEM to match RGB...")
            self.needs_resampling = True
            # Use RGB dimensions as reference
            self.height = self.rgb_height
            self.width = self.rgb_width
            self.transform = self.rgb_transform
        else:
            self.needs_resampling = False
            self.height = self.rgb_height
            self.width = self.rgb_width
            self.transform = self.rgb_transform
        
        # Estimate memory requirement (in GB)
        pixels = self.height * self.width
        # RGB (3 bands) + DEM + 8 intermediate arrays
        estimated_gb = (pixels * 4 * 12) / (1024 ** 3)
        
        print(f"\nProcessing dimensions: {self.height} x {self.width}")
        print(f"Estimated memory needed: {estimated_gb:.2f} GB")
        
        if estimated_gb > self.max_memory_gb * 0.7:  # Use 70% threshold for safety
            print(f"⚠️  Enabling tiled processing (tile size: {self.tile_size})")
            self.use_tiling = True
        else:
            print("✓ Loading full images into memory")
            self.use_tiling = False

    def load_data_full(self):
        """Load full data into memory (for smaller files)"""
        print("\nLoading RGB data...")
        with rasterio.open(self.rgb_path) as src:
            self.rgb = src.read().astype(np.float32)
            if self.rgb.max() > 1:
                self.rgb = self.rgb / 255.0
        
        print("Loading DEM data...")
        with rasterio.open(self.dem_path) as src:
            if self.needs_resampling:
                print(f"  Resampling DEM from {self.dem_height}x{self.dem_width} to {self.height}x{self.width}...")
                from rasterio.enums import Resampling
                # Read with output shape to trigger resampling
                self.dem = src.read(
                    1,
                    out_shape=(self.height, self.width),
                    resampling=Resampling.bilinear
                ).astype(np.float32)
                print("  ✓ DEM resampled successfully")
            else:
                self.dem = src.read(1).astype(np.float32)
        
        print(f"✓ RGB shape: {self.rgb.shape}")
        print(f"✓ DEM shape: {self.dem.shape}")
        
        # Verify shapes match
        if self.rgb.shape[1:] != self.dem.shape:
            raise ValueError(f"Shape mismatch after loading: RGB {self.rgb.shape[1:]} vs DEM {self.dem.shape}")
        
        gc.collect()

    def process_tile(self, window, rgb_src, dem_src):
        """Process a single tile"""
        # Read RGB tile
        rgb_tile = rgb_src.read(window=window).astype(np.float32)
        if rgb_tile.max() > 1:
            rgb_tile = rgb_tile / 255.0
        
        # Read DEM tile with resampling if needed
        if self.needs_resampling:
            from rasterio.enums import Resampling
            # Calculate corresponding DEM window
            # Transform from RGB coordinates to DEM coordinates
            rgb_transform = self.rgb_transform
            dem_transform = self.dem_transform
            
            # Get bounds of RGB window
            col_off, row_off = window.col_off, window.row_off
            width, height = window.width, window.height
            
            # Use RGB window directly and let rasterio handle resampling
            dem_tile = dem_src.read(
                1,
                window=window,
                out_shape=(height, width),
                resampling=Resampling.bilinear
            ).astype(np.float32)
        else:
            dem_tile = dem_src.read(1, window=window).astype(np.float32)
        
        return rgb_tile, dem_tile

    def calculate_vegetation_mask(self, rgb=None, threshold=0.1):
        """
        Calculate vegetation mask using Excess Green Index (ExG)
        """
        if rgb is None:
            rgb = self.rgb
            
        r, g, b = rgb[0], rgb[1], rgb[2]
        
        # Excess Green Index
        exg = 2 * g - r - b
        veg_mask = (exg > threshold).astype(np.uint8)
        
        return veg_mask

    def calculate_height_metrics(self, dem, window_sizes=[9, 15, 25]):
        """
        Calculate comprehensive height-based metrics for green cover classification
        Memory-optimized version
        
        Parameters:
        -----------
        dem : numpy array
            Digital Elevation Model
        window_sizes : list
            Different window sizes for multi-scale analysis
        
        Returns:
        --------
        dict with height metrics
        """
        metrics = {}
        
        print(f"  Computing elevation metrics...")
        
        # 1. Absolute elevation (normalized)
        metrics['elevation'] = dem
        
        # 2. Multi-scale height variation (roughness)
        print(f"  Computing height variation (scales: {window_sizes})...")
        height_stds = []
        for i, ws in enumerate(window_sizes):
            mean = ndimage.uniform_filter(dem, size=ws, mode='reflect')
            sqr_mean = ndimage.uniform_filter(dem ** 2, size=ws, mode='reflect')
            std = np.sqrt(np.maximum(sqr_mean - mean ** 2, 0)).astype(np.float32)
            height_stds.append(std)
            del mean, sqr_mean
            gc.collect()
            
        metrics['height_std_fine'] = height_stds[0]    # Fine scale
        metrics['height_std_medium'] = height_stds[1]  # Medium scale
        metrics['height_std_coarse'] = height_stds[2]  # Coarse scale
        
        # 3. Relative height (deviation from local mean)
        print(f"  Computing relative heights...")
        local_mean_small = ndimage.uniform_filter(dem, size=15, mode='reflect')
        metrics['relative_height_local'] = (dem - local_mean_small).astype(np.float32)
        del local_mean_small
        
        local_mean_large = ndimage.uniform_filter(dem, size=51, mode='reflect')
        metrics['relative_height_regional'] = (dem - local_mean_large).astype(np.float32)
        del local_mean_large
        gc.collect()
        
        # 4. Height range in local neighborhood
        print(f"  Computing height range...")
        local_min = ndimage.minimum_filter(dem, size=15, mode='reflect')
        local_max = ndimage.maximum_filter(dem, size=15, mode='reflect')
        metrics['height_range'] = (local_max - local_min).astype(np.float32)
        del local_min, local_max
        gc.collect()
        
        # 5. Topographic Position Index (TPI)
        tpi_mean = ndimage.uniform_filter(dem, size=25, mode='reflect')
        metrics['tpi'] = (dem - tpi_mean).astype(np.float32)
        del tpi_mean
        gc.collect()
        
        # 6. Surface roughness (standard deviation of slope)
        print(f"  Computing slope roughness...")
        dy, dx = np.gradient(dem)
        slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2))).astype(np.float32)
        slope_mean = ndimage.uniform_filter(slope, size=9, mode='reflect')
        slope_sqr_mean = ndimage.uniform_filter(slope ** 2, size=9, mode='reflect')
        slope_std = np.sqrt(np.maximum(slope_sqr_mean - slope_mean ** 2, 0)).astype(np.float32)
        metrics['slope_roughness'] = slope_std
        
        del dy, dx, slope, slope_mean, slope_sqr_mean, slope_std
        gc.collect()
        
        print(f"  ✓ Height metrics computed")
        return metrics

    def classify_by_height(self, veg_mask, height_metrics, 
                          method='combined',
                          low_veg_max_height=2.0,      # meters
                          medium_veg_max_height=5.0,   # meters
                          tall_veg_threshold=5.0):      # meters
        """
        Classify vegetation into categories based on height
        
        Categories:
        - 0: Non-vegetation
        - 1: Low vegetation (grass, crops) - typically 0-2m
        - 2: Medium vegetation (shrubs, young trees) - typically 2-5m
        - 3: Tall vegetation (mature trees, forest) - typically >5m
        
        Parameters:
        -----------
        method : str
            'height_range' - use local height range
            'height_std' - use height variation
            'relative_height' - use relative elevation
            'combined' - use weighted combination (recommended)
        """
        print(f"\nClassifying vegetation by height (method: {method})...")
        
        h, w = veg_mask.shape
        classification = np.zeros((h, w), dtype=np.uint8)
        
        if method == 'height_range':
            # Use local height range as proxy for vegetation height
            height_proxy = height_metrics['height_range']
            
        elif method == 'height_std':
            # Use height standard deviation
            height_proxy = height_metrics['height_std_medium']
            
        elif method == 'relative_height':
            # Use relative height (good for distinguishing tree crowns)
            height_proxy = np.abs(height_metrics['relative_height_local'])
            
        elif method == 'combined':
            # Weighted combination of multiple metrics
            # Normalize each metric to 0-1
            def normalize_metric(arr):
                veg_values = arr[veg_mask > 0]
                if len(veg_values) == 0:
                    return np.zeros_like(arr)
                p1, p99 = np.percentile(veg_values, [1, 99])
                if p99 - p1 == 0:
                    return np.zeros_like(arr)
                return np.clip((arr - p1) / (p99 - p1), 0, 1)
            
            # Combine metrics with weights
            height_proxy = (
                0.30 * normalize_metric(height_metrics['height_range']) +
                0.25 * normalize_metric(height_metrics['height_std_medium']) +
                0.20 * normalize_metric(np.abs(height_metrics['relative_height_local'])) +
                0.15 * normalize_metric(height_metrics['height_std_coarse']) +
                0.10 * normalize_metric(height_metrics['slope_roughness'])
            )
        
        # Store the height proxy for visualization
        self.height_proxy = height_proxy
        
        # Classify based on height proxy
        # These thresholds should be calibrated based on your specific data
        veg_values = height_proxy[veg_mask > 0]
        if len(veg_values) > 0:
            p33 = np.percentile(veg_values, 33)
            p66 = np.percentile(veg_values, 66)
            
            print(f"Height proxy percentiles: 33rd={p33:.3f}, 66th={p66:.3f}")
            
            # Classify
            classification[veg_mask > 0] = 1  # Default: low vegetation
            classification[(veg_mask > 0) & (height_proxy > p33)] = 2  # Medium
            classification[(veg_mask > 0) & (height_proxy > p66)] = 3  # Tall
        
        # Calculate statistics
        total_veg = np.sum(veg_mask)
        low_veg = np.sum(classification == 1)
        medium_veg = np.sum(classification == 2)
        tall_veg = np.sum(classification == 3)
        
        print(f"\nClassification Results:")
        print(f"Total vegetation pixels: {total_veg:,}")
        print(f"Low vegetation (crops/grass): {low_veg:,} ({100*low_veg/total_veg:.1f}%)")
        print(f"Medium vegetation (shrubs): {medium_veg:,} ({100*medium_veg/total_veg:.1f}%)")
        print(f"Tall vegetation (trees/forest): {tall_veg:,} ({100*tall_veg/total_veg:.1f}%)")
        
        return classification

    def process_full_image(self, method='combined', veg_threshold=0.1):
        """Process full image (non-tiled)"""
        print("\n" + "="*60)
        print("Processing full image...")
        print("="*60)
        
        # Calculate vegetation mask
        print("\nStep 1: Calculating vegetation mask...")
        self.veg_mask = self.calculate_vegetation_mask(threshold=veg_threshold)
        veg_percent = 100 * np.mean(self.veg_mask)
        print(f"Vegetation coverage: {veg_percent:.2f}%")
        
        # Calculate height metrics
        print("\nStep 2: Calculating height metrics...")
        self.height_metrics = self.calculate_height_metrics(self.dem)
        
        # Classify
        print("\nStep 3: Classifying by height...")
        self.classification = self.classify_by_height(
            self.veg_mask, 
            self.height_metrics,
            method=method
        )
        
        gc.collect()
        return self.classification

    def process_tiled(self, method='combined', veg_threshold=0.1):
        """Process large image using tiles"""
        print("\n" + "="*60)
        print("Processing with tiling...")
        print("="*60)
        
        # Initialize output arrays
        self.veg_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        self.classification = np.zeros((self.height, self.width), dtype=np.uint8)
        self.height_proxy = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Open files
        rgb_src = rasterio.open(self.rgb_path)
        dem_src = rasterio.open(self.dem_path)
        
        # Calculate number of tiles
        n_tiles_h = int(np.ceil(self.height / self.tile_size))
        n_tiles_w = int(np.ceil(self.width / self.tile_size))
        total_tiles = n_tiles_h * n_tiles_w
        
        print(f"Processing {n_tiles_h} x {n_tiles_w} = {total_tiles} tiles")
        print(f"Tile size: {self.tile_size} x {self.tile_size}")
        
        tile_count = 0
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                tile_count += 1
                
                # Define window with overlap
                row_start = i * self.tile_size
                col_start = j * self.tile_size
                row_end = min(row_start + self.tile_size, self.height)
                col_end = min(col_start + self.tile_size, self.width)
                
                window = Window(col_start, row_start, 
                              col_end - col_start, 
                              row_end - row_start)
                
                if tile_count % 10 == 0 or tile_count == total_tiles:
                    print(f"Processing tile {tile_count}/{total_tiles}...")
                
                # Process tile
                rgb_tile, dem_tile = self.process_tile(window, rgb_src, dem_src)
                
                # Vegetation mask
                veg_mask_tile = self.calculate_vegetation_mask(rgb_tile, veg_threshold)
                
                # Height metrics
                height_metrics_tile = self.calculate_height_metrics(dem_tile)
                
                # Classify
                classification_tile = self.classify_by_height(
                    veg_mask_tile, 
                    height_metrics_tile,
                    method=method
                )
                
                # Store results
                self.veg_mask[row_start:row_end, col_start:col_end] = veg_mask_tile
                self.classification[row_start:row_end, col_start:col_end] = classification_tile
                
                # Clean up
                del rgb_tile, dem_tile, veg_mask_tile, height_metrics_tile, classification_tile
                gc.collect()
        
        rgb_src.close()
        dem_src.close()
        
        # Calculate final statistics
        total_veg = np.sum(self.veg_mask)
        low_veg = np.sum(self.classification == 1)
        medium_veg = np.sum(self.classification == 2)
        tall_veg = np.sum(self.classification == 3)
        
        print(f"\nFinal Results:")
        print(f"Total vegetation: {total_veg:,} pixels ({100*total_veg/(self.height*self.width):.2f}%)")
        print(f"Low vegetation: {low_veg:,} ({100*low_veg/total_veg:.1f}%)")
        print(f"Medium vegetation: {medium_veg:,} ({100*medium_veg/total_veg:.1f}%)")
        print(f"Tall vegetation: {tall_veg:,} ({100*tall_veg/total_veg:.1f}%)")
        
        return self.classification

    def visualize_results(self, save_path=None, dpi=150):
        """Visualize classification results"""
        print("\nGenerating visualization...")
        
        # Downsample for display if needed
        max_display = 2000
        if self.height > max_display or self.width > max_display:
            scale = max_display / max(self.height, self.width)
            display_h = int(self.height * scale)
            display_w = int(self.width * scale)
            print(f"Downsampling for display: {display_h} x {display_w}")
            
            from scipy.ndimage import zoom
            zoom_factor = (display_h / self.height, display_w / self.width)
            
            if not self.use_tiling:
                rgb_display = np.transpose(zoom(self.rgb, (1, zoom_factor[0], zoom_factor[1])), (1, 2, 0))
                dem_display = zoom(self.dem, zoom_factor)
            else:
                # For tiled processing, read downsampled versions
                with rasterio.open(self.rgb_path) as src:
                    data = src.read(out_shape=(src.count, display_h, display_w))
                    rgb_display = np.transpose(data.astype(np.float32) / 255.0, (1, 2, 0))
                with rasterio.open(self.dem_path) as src:
                    dem_display = src.read(1, out_shape=(display_h, display_w)).astype(np.float32)
            
            veg_display = zoom(self.veg_mask, zoom_factor, order=0)
            class_display = zoom(self.classification, zoom_factor, order=0)
            
            if hasattr(self, 'height_metrics') and not self.use_tiling:
                height_range_display = zoom(self.height_metrics['height_range'], zoom_factor)
                height_std_display = zoom(self.height_metrics['height_std_medium'], zoom_factor)
                rel_height_display = zoom(self.height_metrics['relative_height_local'], zoom_factor)
            else:
                height_range_display = np.zeros((display_h, display_w))
                height_std_display = np.zeros((display_h, display_w))
                rel_height_display = np.zeros((display_h, display_w))
        else:
            rgb_display = np.transpose(self.rgb, (1, 2, 0)) if not self.use_tiling else None
            dem_display = self.dem if not self.use_tiling else None
            veg_display = self.veg_mask
            class_display = self.classification
            height_range_display = self.height_metrics.get('height_range', np.zeros_like(self.veg_mask))
            height_std_display = self.height_metrics.get('height_std_medium', np.zeros_like(self.veg_mask))
            rel_height_display = self.height_metrics.get('relative_height_local', np.zeros_like(self.veg_mask))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # RGB
        if rgb_display is not None:
            axes[0, 0].imshow(rgb_display)
            axes[0, 0].set_title('RGB Composite', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # DEM
        if dem_display is not None:
            im1 = axes[0, 1].imshow(dem_display, cmap='terrain')
            axes[0, 1].set_title('DEM (Elevation)', fontsize=12, fontweight='bold')
            plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        axes[0, 1].axis('off')
        
        # Vegetation mask
        axes[0, 2].imshow(veg_display, cmap='Greens')
        axes[0, 2].set_title('Vegetation Mask', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Height range
        im2 = axes[1, 0].imshow(height_range_display * veg_display, cmap='YlOrRd')
        axes[1, 0].set_title('Height Range (Local)', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
        axes[1, 0].axis('off')
        
        # Height variation
        im3 = axes[1, 1].imshow(height_std_display * veg_display, cmap='plasma')
        axes[1, 1].set_title('Height Variation', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
        axes[1, 1].axis('off')
        
        # Final classification
        cmap = ListedColormap(['black', 'yellow', 'orange', 'darkgreen'])
        im4 = axes[1, 2].imshow(class_display, cmap=cmap, vmin=0, vmax=3)
        axes[1, 2].set_title('Height-Based Classification', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label='Non-vegetation'),
            Patch(facecolor='yellow', label='Low veg (0-2m)'),
            Patch(facecolor='orange', label='Medium veg (2-5m)'),
            Patch(facecolor='darkgreen', label='Tall veg (>5m)')
        ]
        axes[1, 2].legend(handles=legend_elements, loc='upper right', 
                         fontsize=8, framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        plt.close()
        gc.collect()

    def save_classification(self, output_path):
        """Save classification as GeoTIFF"""
        print(f"\nSaving classification to: {output_path}")
        
        meta = self.rgb_meta.copy()
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(self.classification, 1)
        
        print("Classification saved successfully!")

    def run_analysis(self, output_path=None, viz_path=None, 
                    method='combined', veg_threshold=0.1):
        """
        Run complete height-based classification
        
        Parameters:
        -----------
        method : str
            Classification method ('combined', 'height_range', 'height_std', 'relative_height')
        veg_threshold : float
            Threshold for vegetation detection (default: 0.1)
        """
        print("="*60)
        print("HEIGHT-BASED GREEN COVER CLASSIFICATION")
        print("="*60)
        
        # Load metadata
        self.load_data_info()
        
        # Process based on size
        if self.use_tiling:
            self.process_tiled(method=method, veg_threshold=veg_threshold)
        else:
            self.load_data_full()
            self.process_full_image(method=method, veg_threshold=veg_threshold)
        
        # Visualize
        if viz_path:
            self.visualize_results(save_path=viz_path)
        
        # Save
        if output_path:
            self.save_classification(output_path)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        
        return self.classification


# Example usage
if __name__ == "__main__":
    classifier = HeightBasedGreenCoverClassifier(
        rgb_path='/workspace/input/Uddandarayanipalem.tif',
        dem_path='/workspace/output/RB023_Uddandarayanipalem DEM_2.tif',
        max_memory_gb=16.0,  # Adjust based on your system (you have 28GB available)
        tile_size=512        # Smaller tiles for very large files
    )
    
    # Run analysis
    classification = classifier.run_analysis(
        output_path='/workspace/output/Green_Cover/height_based_classification.tif',
        viz_path='/workspace/output/Green_Cover/height_classification_results.png',
        method='combined',      # Options: 'combined', 'height_range', 'height_std', 'relative_height'
        veg_threshold=0.1       # Adjust for vegetation detection sensitivity
    )
    
    # Classification legend:
    # 0 = Non-vegetation
    # 1 = Low vegetation (grass, crops) - 0-2m height
    # 2 = Medium vegetation (shrubs, young trees) - 2-5m height
    # 3 = Tall vegetation (mature trees, forest) - >5m height