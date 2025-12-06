from skimage import io
from osgeo import gdal
import ntpath
import numpy as np
from scipy import ndimage
import os
import glob
import rasterio
import time


def change_projection(inputfile, referencefile):
    file = inputfile
    outfile = inputfile
    ds = gdal.Open(file)
    # band = ds.GetRasterBand(1)
    arr = ds.ReadAsArray()
    # print(arr.shape)
    if len(arr.shape) == 3:
        num = arr.shape[0]
        ds_ref = gdal.Open(referencefile)
        geotrans = ds_ref.GetGeoTransform()
        proj = ds_ref.GetProjection()
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outfile, arr.shape[2], arr.shape[1], arr.shape[0], gdal.GDT_Float32)
        outdata.SetGeoTransform(geotrans)  ##sets same geotransform as input
        outdata.SetProjection(proj)  ##sets same projection as input
        # outdata.GetRasterBand(1).WriteArray(arr)
        for i in range(1, int(num) + 1):
            # print(arr.shape)
            outdata.GetRasterBand(i).WriteArray(arr[i - 1, :, :])
            outdata.GetRasterBand(i).SetNoDataValue(10000)
        # outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache()  ##saves to disk!!
        outdata = None
        band = None
        ds = None
    elif len(arr.shape) == 2:
        [cols, rows] = arr.shape
        ds_ref = gdal.Open(referencefile)	
        geotrans = ds_ref.GetGeoTransform()
        proj = ds_ref.GetProjection()
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
        outdata.SetGeoTransform(geotrans)  ##sets same geotransform as input
        outdata.SetProjection(proj)  ##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(arr)
        outdata.GetRasterBand(1).SetNoDataValue(10000)  ##if you want these values transparent
        outdata.FlushCache()  ##saves to disk!!
        outdata = None
        band = None
        ds = None
def minMax(band):
    # print('mmband', band.min(), band.max())
    # minimum = 1
    # minimum =0.0
    # maximum = 255.0
    band = np.float32(band)
    band = (band - band.min()) / (band.max() - band.min())
    # band = (band - minimum) / (maximum - minimum)
    return (band)

def image_scaler(raw_ras, out_dir):
    in_ras = rasterio.open(raw_ras)
    out_meta = in_ras.meta

    rd_ras = in_ras.read()

    dra_img = []

    #for band in range(rd_ras.shape[0]-1):
    for band in range(rd_ras.shape[0]):

        arr = rd_ras[band]

        arr1 = arr.copy()
        # thr1 = round(np.percentile(arr[arr > 0], 2.5))
        # thr2 = round(np.percentile(arr[arr > 0], 99))
        # arr1[arr1 < thr1] = thr1
        # arr1[arr1 > thr2] = thr2
        # arr1 = (arr1 - thr1) / (thr2 - thr1)
        # arr1 = arr1 * 255.
        # arr1 = np.uint8(arr1)
        # arr1[arr1 == 0] = 1.

        if np.any(arr1 > 0):
            thr1 = round(np.percentile(arr[arr > 0], 2.5))
            thr2 = round(np.percentile(arr[arr > 0], 99))
            arr1[arr1 < thr1] = thr1
            arr1[arr1 > thr2] = thr2
            arr1 = (arr1 - thr1) / (thr2 - thr1) if (thr2 - thr1) > 0 else np.zeros_like(arr1)
            arr1 = arr1 * 255.
            arr1 = np.uint8(arr1)
            arr1[arr1 == 0] = 1.
        else:
            arr1 = np.zeros_like(arr1, dtype=np.uint8)

        dra_img.append(arr1)

    foo = np.stack(dra_img)
    out_meta.update({'count':3,'dtype':'uint8'})
    base_name = os.path.splitext(os.path.basename(raw_ras))[0]
    output_dir = out_dir
    # rgb_path = os.path.join(output_dir, f"{base_name}_rgb.tif")
    rgb_path = os.path.join(output_dir, f"{base_name}.tif")


    with rasterio.open(rgb_path,'w',**out_meta) as dst:
        dst.write(foo)

    return rgb_path

def image_scaler_histogram_fixed(raw_ras, out_dir, bins=1024):
    """
    Memory-safe raster scaling using histogram-based percentile calculation,
    with correct geospatial metadata handling.
    """
    with rasterio.open(raw_ras) as in_ras:
        out_meta = in_ras.meta.copy()
        num_bands = in_ras.count
        base_name = os.path.splitext(os.path.basename(raw_ras))[0]
        rgb_path = os.path.join(out_dir, f"{base_name}_scaled.tif")

        print(f"\n[START] Processing {os.path.basename(raw_ras)} ({num_bands} bands)")

        # Pass 1: build histogram
        hist = np.zeros(bins, dtype=np.int64)
        min_val, max_val = None, None
       
        all_blocks = list(in_ras.block_windows(1)) # Use the first band for blocks
        total_blocks = len(all_blocks)
       
        print(f"  Total blocks: {total_blocks}")
        print("  Pass 1: Calculating histogram...")
        start_time = time.time()
        for i, (_, window) in enumerate(all_blocks, start=1):
            arr = in_ras.read(1, window=window)
            arr = arr[arr > 0]
            if arr.size > 0:
                if min_val is None:
                    min_val, max_val = arr.min(), arr.max()
                else:
                    min_val = min(min_val, arr.min())
                    max_val = max(max_val, arr.max())
               
                if max_val - min_val > 0:
                    block_hist, _ = np.histogram(arr, bins=bins, range=(min_val, max_val))
                    hist += block_hist
           
            if i % 50 == 0 or i == total_blocks:
                elapsed = time.time() - start_time
                pct = (i / total_blocks) * 100
                eta = (elapsed / i) * (total_blocks - i) if i > 0 else 0
                print(f"  Pass 1: {i}/{total_blocks} blocks ({pct:.1f}%) - ETA {eta:.1f}s", end="\r")

        if hist.sum() == 0:
            thr1 = thr2 = 0
        else:
            cdf = np.cumsum(hist)
            total = cdf[-1]
            thr1_idx = np.searchsorted(cdf, total * 0.025)
            thr2_idx = np.searchsorted(cdf, total * 0.99)
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            thr1, thr2 = bin_edges[thr1_idx], bin_edges[thr2_idx]
       
        print(f"\n  Thresholds: thr1={thr1:.2f}, thr2={thr2:.2f}")

        # Update output metadata to reflect new data type and count
        out_meta.update({
            'count': num_bands,
            'dtype': 'uint8',
            'nodata': 0 # Setting nodata to 0 is common for 8-bit images
        })

        # Pass 2: process and write
        print("  Pass 2: Scaling & writing...")
        start_time = time.time()
        with rasterio.open(rgb_path, 'w', **out_meta) as dst:
            for band_idx in range(1, num_bands + 1):
                for i, (_, window) in enumerate(in_ras.block_windows(band_idx), start=1):
                    arr = in_ras.read(band_idx, window=window)
                   
                    if np.any(arr > 0) and (thr2 - thr1) > 0:
                        arr = np.clip(arr, thr1, thr2)
                        arr = (arr - thr1) / (thr2 - thr1)
                        arr = (arr * 255.0).astype(np.uint8)
                        arr[arr == 0] = 1 # Set nodata to a small value if 0 is used as nodata
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint8)
                   
                    dst.write(arr, band_idx, window=window)
               
                print(f"  Band {band_idx}: Done ✅")

        print(f"\n[DONE] Saved -> {rgb_path}")
        return rgb_path

def image_scaler_memory_efficient(raw_ras, out_dir, per_block_sample=500):
    import numpy as np, os, rasterio

    with rasterio.open(raw_ras) as in_ras:
        out_meta = in_ras.meta.copy()
        num_bands = in_ras.count
        base_name = os.path.splitext(os.path.basename(raw_ras))[0]
        rgb_path = os.path.join(out_dir, f"{base_name}.tif")

        print(f"\n[START] Processing {os.path.basename(raw_ras)} ({num_bands} bands)")

        out_meta.update({'count': num_bands, 'dtype': 'uint8'})
        with rasterio.open(rgb_path, 'w', **out_meta) as dst:
            for band_idx in range(1, num_bands + 1):
                print(f"  Band {band_idx}: Calculating thresholds...")
                sampled_pixels = []

                # Pass 1: Threshold calc with per-block sampling
                for _, window in in_ras.block_windows(band_idx):
                    arr = in_ras.read(band_idx, window=window)
                    vals = arr[arr > 0]
                    if vals.size > per_block_sample:
                        vals = np.random.choice(vals, size=per_block_sample, replace=False)
                    if vals.size > 0:
                        sampled_pixels.extend(vals)

                if sampled_pixels:
                    sampled_pixels = np.array(sampled_pixels, dtype=np.float64)
                    thr1 = np.percentile(sampled_pixels, 2.5)
                    thr2 = np.percentile(sampled_pixels, 99)
                else:
                    thr1 = thr2 = 0

                print(f"    Thresholds: thr1={thr1:.2f}, thr2={thr2:.2f}")

                # Pass 2: Process & write
                for _, window in in_ras.block_windows(band_idx):
                    arr = in_ras.read(band_idx, window=window).astype(np.float64)
                    if np.any(arr > 0) and (thr2 - thr1) > 0:
                        arr = np.clip(arr, thr1, thr2)
                        arr = (arr - thr1) / (thr2 - thr1)
                        arr = (arr * 255.0).astype(np.uint8)
                        arr[arr == 0] = 1
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint8)
                    dst.write(arr, band_idx, window=window)

                print(f"  Band {band_idx}: Done ✅")

        print(f"[DONE] Saved -> {rgb_path}")
        return rgb_path
    
def image_scaler_histogram(raw_ras, out_dir, bins=1024):
    """
    Memory-safe raster scaling using histogram-based percentile calculation
    with progress and ETA logging.
    """
    with rasterio.open(raw_ras) as in_ras:
        out_meta = in_ras.meta.copy()
        num_bands = in_ras.count
        base_name = os.path.splitext(os.path.basename(raw_ras))[0]
        rgb_path = os.path.join(out_dir, f"{base_name}.tif")

        print(f"\n[START] Processing {os.path.basename(raw_ras)} ({num_bands} bands)")

        out_meta.update({'count': num_bands, 'dtype': 'uint8'})
        with rasterio.open(rgb_path, 'w', **out_meta) as dst:
            for band_idx in range(1, num_bands + 1):
                print(f"\n  Band {band_idx}: Calculating thresholds via histogram...")
                
                # Count total blocks for progress
                all_blocks = list(in_ras.block_windows(band_idx))
                total_blocks = len(all_blocks)
                print(f"    Total blocks: {total_blocks}")

                # Pass 1: build histogram
                hist = np.zeros(bins, dtype=np.int64)
                min_val, max_val = None, None

                start_time = time.time()
                for i, (_, window) in enumerate(all_blocks, start=1):
                    arr = in_ras.read(band_idx, window=window)
                    arr = arr[arr > 0]
                    if arr.size > 0:
                        if min_val is None:
                            min_val, max_val = arr.min(), arr.max()
                        else:
                            min_val = min(min_val, arr.min())
                            max_val = max(max_val, arr.max())
                        block_hist, _ = np.histogram(arr, bins=bins, range=(min_val, max_val))
                        hist += block_hist
                    
                    # Progress log
                    if i % 50 == 0 or i == total_blocks:
                        elapsed = time.time() - start_time
                        pct = (i / total_blocks) * 100
                        eta = (elapsed / i) * (total_blocks - i)
                        print(f"    Pass 1: {i}/{total_blocks} blocks ({pct:.1f}%) - ETA {eta:.1f}s", end="\r")

                if hist.sum() == 0:
                    thr1 = thr2 = 0
                else:
                    cdf = np.cumsum(hist)
                    total = cdf[-1]
                    thr1_idx = np.searchsorted(cdf, total * 0.025)
                    thr2_idx = np.searchsorted(cdf, total * 0.99)
                    bin_edges = np.linspace(min_val, max_val, bins)
                    thr1, thr2 = bin_edges[thr1_idx], bin_edges[thr2_idx]

                print(f"\n    Thresholds: thr1={thr1:.2f}, thr2={thr2:.2f}")

                # Pass 2: process and write
                print(f"  Band {band_idx}: Scaling & writing...")
                start_time = time.time()
                for i, (_, window) in enumerate(all_blocks, start=1):
                    arr = in_ras.read(band_idx, window=window).astype(np.float64)
                    if np.any(arr > 0) and (thr2 - thr1) > 0:
                        arr = np.clip(arr, thr1, thr2)
                        arr = (arr - thr1) / (thr2 - thr1)
                        arr = (arr * 255.0).astype(np.uint8)
                        arr[arr == 0] = 1
                    else:
                        arr = np.zeros_like(arr, dtype=np.uint8)
                    dst.write(arr, band_idx, window=window)

                    # Progress log
                    if i % 50 == 0 or i == total_blocks:
                        elapsed = time.time() - start_time
                        pct = (i / total_blocks) * 100
                        eta = (elapsed / i) * (total_blocks - i)
                        print(f"    Pass 2: {i}/{total_blocks} blocks ({pct:.1f}%) - ETA {eta:.1f}s", end="\r")

                print(f"\n  Band {band_idx}: Done ✅")

        print(f"\n[DONE] Saved -> {rgb_path}")
        return rgb_path


def assing_projection(rd_ras,inputfile,reference_file):
    # image = rio.open(inputfile)

    out_meta = rasterio.open(reference_file).meta

    # rd_ras = image.read()
    band_count = rd_ras.shape[0]

    out_meta.update({'count':band_count,'nodata': 10000,'dtype': 'float32'})

    with rasterio.open(inputfile,'w',**out_meta) as dst:
        dst.write(rd_ras)


def assing_projection_new(rd_ras, inputfile, reference_file):
    """
    Correctly transfer projection information from reference file to output file.
    
    Args:
        rd_ras: The raster data array
        inputfile: Path to the output file being created
        reference_file: Path to the reference file containing valid geospatial information
    """
    try:
        # Open the reference file to get complete metadata
        with rasterio.open(reference_file) as src:
            # Get the complete transformation information
            transform = src.transform
            crs = src.crs
            
            # Update metadata for output
            out_meta = src.meta.copy()
            band_count = rd_ras.shape[0]
            out_meta.update({
                'count': band_count,
                'nodata': 10000,
                'dtype': 'float32',
                'transform': transform,  # Ensure transform is explicitly set
                'crs': crs              # Ensure CRS is explicitly set
            })
            
            # Write the output with correct geospatial information
            with rasterio.open(inputfile, 'w', **out_meta) as dst:
                dst.write(rd_ras)
                
            print(f"Successfully wrote geospatial data to {inputfile}")
            print(f"CRS: {crs}")
            print(f"Transform: {transform}")
            
    except Exception as e:
        print(f"Error in assing_projection: {e}")

def add_nir_sobel_filter(image, outPath):
    print("In Lut")
    try:
        # Open image with rasterio to preserve CRS information
        with rasterio.open(image) as src:
            # Read image data
            img_data = src.read()
            # Get metadata for later use
            meta = src.meta.copy()
            
            # Convert to format expected by rest of code
            img = np.moveaxis(img_data, 0, -1)  # Convert from (bands, height, width) to (height, width, bands)
            
        # Also open with GDAL to maintain compatibility with existing code
        img2 = gdal.Open(image)
        arr = img2.ReadAsArray()
        
        fName = ntpath.basename(image)
        dra_img = []
        OutDir = outPath
        
        for band in range(img.shape[2]):
            arr1 = img[:, :, band]  # Changed index logic to be more direct
            arr2 = arr1.copy()
            arr2[arr2 > 0] = 1
            
            # More robust percentile calculation
            valid_pixels = arr1[arr1 > 0]
            if len(valid_pixels) > 0:
                thr1 = np.percentile(valid_pixels, 2.5)
                thr2 = np.percentile(valid_pixels, 99)
            else:
                thr1, thr2 = 0, 1  # Default values if no valid pixels
                
            arr1 = np.clip(arr1, thr1, thr2)
            arr1 = (arr1 - thr1) / (thr2 - thr1) if (thr2 - thr1) > 0 else np.zeros_like(arr1)
            arr1 = arr1 * 255.
            arr1 = np.uint8(arr1)
            arr1[arr1 == 0] = 1.
            arr2 = np.uint8(arr2)
            arr1 = arr1 * arr2
            dra_img.append(arr1)
            
        dra_img = np.array(dra_img)
        dra_img = np.rollaxis(dra_img, 0, 3)
        dra_img = np.uint8(dra_img)
        print('dra_img shape: ', dra_img.shape)
        
        # Extract bands
        b1 = dra_img[:, :, 0]
        b2 = dra_img[:, :, 1]
        b3 = dra_img[:, :, 2]
        b4 = dra_img[:, :, 3]

        b1 = minMax(b1)
        b2 = minMax(b2)
        b3 = minMax(b3)
        b4 = minMax(b4)

        # Compute sobel filter using only the first three bands (RGB)
        grey1 = (b1 + b2 + b3) / 3

        grey2 = grey1.copy()
        grey2[grey2 > 0] = 1
        grey = grey1.copy()
        grey = grey[grey != 0]
        
        if len(grey) > 0:
            thr1 = grey.mean() - 2 * grey.std()
            thr2 = grey.mean() + 2 * grey.std()
            grey1[grey1 < thr1] = thr1
            grey1[grey1 > thr2] = thr2
            grey1 = (grey1 - thr1) / (thr2 - thr1) if (thr2 - thr1) > 0 else np.zeros_like(grey1)
        else:
            grey1 = np.zeros_like(grey1)
            
        sobelx = ndimage.sobel(grey1, 0)
        sobely = ndimage.sobel(grey1, 1)
        sobel = np.hypot(sobelx, sobely)
        print('Standard Div : ', sobel.std())
        print('Min : ', sobel.min(), 'Max : ', sobel.max())
        
        sobelMinStd = (sobel - sobel.min()) / (sobel.std()) if sobel.std() > 0 else np.zeros_like(sobel)
        sobelMinStd = sobelMinStd * grey2
        sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))

        b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
        b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
        b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
        b4 = np.reshape(b4, (b1.shape[0], b1.shape[1], 1))

        # Include the 4th band in the concatenation
        img = np.concatenate((b3, b2, b1, b4), axis=-1)
        #img = np.concatenate((b1, b2, b3, sobelMinStd), axis=-1)
        rgbsobelMinStd = np.concatenate((img, sobelMinStd), axis=-1)
        #rgbsobelMinStd = np.concatenate((img, b4), axis=-1)
        rgbsobelMinStd = np.rollaxis(rgbsobelMinStd, 2, 0)  # Convert to (bands, height, width)

        base_name = os.path.splitext(os.path.basename(image))[0]
        output_path = os.path.join(OutDir, f"{base_name}.tif")
        
        # Update metadata for new output
        meta.update({
            'count': rgbsobelMinStd.shape[0],
            'dtype': 'float32',
            'nodata': 10000
        })
        
        # Write directly with rasterio to preserve geospatial info
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(rgbsobelMinStd)
            
        print(f"Applied NIR Sobel Filter and saved to {output_path} with spatial reference preserved")
        return output_path
        
    except Exception as e:
        print(f"Error in add_nir_sobel_filter: {e}")
        import traceback
        traceback.print_exc()
        return None


def add_sobel_nir_filter(image, outPath):
    print("In Lut")
    try:
        # Open image with rasterio to preserve CRS information
        with rasterio.open(image) as src:
            # Read image data
            img_data = src.read()
            # Get metadata for later use
            meta = src.meta.copy()
            
            # Convert to format expected by rest of code
            img = np.moveaxis(img_data, 0, -1)  # Convert from (bands, height, width) to (height, width, bands)
            
        # Also open with GDAL to maintain compatibility with existing code
        img2 = gdal.Open(image)
        arr = img2.ReadAsArray()
        
        fName = ntpath.basename(image)
        dra_img = []
        OutDir = outPath
        
        for band in range(img.shape[2]):
            arr1 = img[:, :, band]  # Changed index logic to be more direct
            arr2 = arr1.copy()
            arr2[arr2 > 0] = 1
            
            # More robust percentile calculation
            valid_pixels = arr1[arr1 > 0]
            if len(valid_pixels) > 0:
                thr1 = np.percentile(valid_pixels, 2.5)
                thr2 = np.percentile(valid_pixels, 99)
            else:
                thr1, thr2 = 0, 1  # Default values if no valid pixels
                
            arr1 = np.clip(arr1, thr1, thr2)
            arr1 = (arr1 - thr1) / (thr2 - thr1) if (thr2 - thr1) > 0 else np.zeros_like(arr1)
            arr1 = arr1 * 255.
            arr1 = np.uint8(arr1)
            arr1[arr1 == 0] = 1.
            arr2 = np.uint8(arr2)
            arr1 = arr1 * arr2
            dra_img.append(arr1)
            
        dra_img = np.array(dra_img)
        dra_img = np.rollaxis(dra_img, 0, 3)
        dra_img = np.uint8(dra_img)
        print('dra_img shape: ', dra_img.shape)
        
        # Extract bands
        b1 = dra_img[:, :, 0]
        b2 = dra_img[:, :, 1]
        b3 = dra_img[:, :, 2]
        b4 = dra_img[:, :, 3]

        b1 = minMax(b1)
        b2 = minMax(b2)
        b3 = minMax(b3)
        b4 = minMax(b4)

        # Compute sobel filter using only the first three bands (RGB)
        grey1 = (b1 + b2 + b3) / 3

        grey2 = grey1.copy()
        grey2[grey2 > 0] = 1
        grey = grey1.copy()
        grey = grey[grey != 0]
        
        if len(grey) > 0:
            thr1 = grey.mean() - 2 * grey.std()
            thr2 = grey.mean() + 2 * grey.std()
            grey1[grey1 < thr1] = thr1
            grey1[grey1 > thr2] = thr2
            grey1 = (grey1 - thr1) / (thr2 - thr1) if (thr2 - thr1) > 0 else np.zeros_like(grey1)
        else:
            grey1 = np.zeros_like(grey1)
            
        sobelx = ndimage.sobel(grey1, 0)
        sobely = ndimage.sobel(grey1, 1)
        sobel = np.hypot(sobelx, sobely)
        print('Standard Div : ', sobel.std())
        print('Min : ', sobel.min(), 'Max : ', sobel.max())
        
        # First compute the sobelMinStd as in the original code
        sobelMinStd = (sobel - sobel.min()) / (sobel.std()) if sobel.std() > 0 else np.zeros_like(sobel)
        sobelMinStd = sobelMinStd * grey2
        
        # Apply percentile normalization to the Sobel band (similar to other bands)
        valid_sobel_pixels = sobelMinStd[sobelMinStd > 0]
        if len(valid_sobel_pixels) > 0:
            thr1_sobel = np.percentile(valid_sobel_pixels, 2.5)
            thr2_sobel = np.percentile(valid_sobel_pixels, 99)
            sobelMinStd = np.clip(sobelMinStd, thr1_sobel, thr2_sobel)
            sobelMinStd = (sobelMinStd - thr1_sobel) / (thr2_sobel - thr1_sobel) if (thr2_sobel - thr1_sobel) > 0 else np.zeros_like(sobelMinStd)
        
        sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))

        b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
        b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
        b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
        b4 = np.reshape(b4, (b1.shape[0], b1.shape[1], 1))

        # Include the 4th band in the concatenation
        img = np.concatenate((b1, b2, b3, b4), axis=-1)
        rgbsobelMinStd = np.concatenate((img, sobelMinStd), axis=-1)
        rgbsobelMinStd = np.rollaxis(rgbsobelMinStd, 2, 0)  # Convert to (bands, height, width)

        base_name = os.path.splitext(os.path.basename(image))[0]
        output_path = os.path.join(OutDir, f"{base_name}.tif")
        
        # Update metadata for new output
        meta.update({
            'count': rgbsobelMinStd.shape[0],
            'dtype': 'float32',
            'nodata': 10000
        })
        
        # Write directly with rasterio to preserve geospatial info
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(rgbsobelMinStd)
            
        print(f"Applied NIR Sobel Filter and saved to {output_path} with spatial reference preserved")
        return output_path
        
    except Exception as e:
        print(f"Error in add_sobel_nir_filter: {e}")
        import traceback
        traceback.print_exc()
        return None

def add_sobel_filter(image, imgPath):
    print("In Lut")
    img = io.imread(image)
    img2 = gdal.Open(image)
    arr = img2.ReadAsArray()
    fName = ntpath.basename(image)
    OutDir = imgPath
    dra_img = []

    for band in range(img.shape[2]):
        arr1 = img[:, :,  band]
        arr2 = arr1.copy()
        arr2[arr2 > 0] = 1
        thr1 = round(np.percentile(arr1[arr1 > 0], 2.5))
        thr2 = round(np.percentile(arr1[arr1 > 0], 99))
        arr1[arr1 < thr1] = thr1
        arr1[arr1 > thr2] = thr2
        arr1 = (arr1 - thr1) / (thr2 - thr1)
        arr1 = arr1 * 255.
        arr1 = np.uint8(arr1)
        arr1[arr1 == 0] = 1.
        arr2 = np.uint8(arr2)
        arr1 = arr1 * arr2
        dra_img.append(arr1)
    dra_img = np.array(dra_img)
    dra_img = np.rollaxis(dra_img, 0, 3)
    dra_img = np.uint8(dra_img)
    print('dra_img shape: ', dra_img.shape)
    b1 = dra_img[:, :, 0]
    b2 = dra_img[:, :, 1]
    b3 = dra_img[:, :, 2]
    b4 = dra_img[:, :, 3]

    b1 = minMax(b1)
    b2 = minMax(b2)
    b3 = minMax(b3)
    b4 = minMax(b4)

    grey1 = (b1 + b2 + b3) / 3
    # grey1 = ndimage.gaussian_filter(grey1, sigma=1)


    grey2 = grey1.copy()
    grey2[grey2 > 0] = 1
    grey = grey1.copy()
    grey = grey[grey != 0]
    thr1 = grey.mean() - 2 * grey.std()
    thr2 = grey.mean() + 2 * grey.std()
    grey1[grey1 < thr1] = thr1
    grey1[grey1 > thr2] = thr2
    grey1 = (grey1 - thr1) / (thr2 - thr1)
    sobelx = ndimage.sobel(grey1, 0)
    sobely = ndimage.sobel(grey1, 1)
    sobel = np.hypot(sobelx, sobely)
    print('Standard Div : ', sobel.std())
    print('Min : ', sobel.min(), 'Max : ', sobel.max())
    sobelMinStd = (sobel - sobel.min()) / (sobel.std())
    sobelMinStd = sobelMinStd * grey2
    sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))
    # io.imsave(OutDir + 'Sobel_Lut/' + fName, sobelMinStd)

    b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
    b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
    b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
    b4 = np.reshape(b4, (b1.shape[0], b1.shape[1], 1))


    img = np.concatenate((b1, b2, b3,b4), axis=-1)
    rgbsobelMinStd = np.concatenate((img, sobelMinStd), axis=-1)
    rgbsobelMinStd = np.rollaxis(rgbsobelMinStd,2,0)
    OutDir = os.path.dirname(image)
    base_name = os.path.splitext(os.path.basename(image))[0]
    output_path = os.path.join(OutDir, f"{base_name}.tif")
    io.imsave(output_path, rgbsobelMinStd )

    assing_projection(rgbsobelMinStd,output_path, image)



def add_old_sobel_filter(image, imgPath):
    print("In Lut")
    img = io.imread(image)
    img2 = gdal.Open(image)
    arr = img2.ReadAsArray()
    fName = ntpath.basename(image)
    OutDir = imgPath
    dra_img = []
    for band in range(3):
        arr1 = img[:, :, band]
        arr2 = arr1.copy()
        arr2[arr2 > 0] = 1
        thr1 = round(np.percentile(arr1[arr1 > 0], 2.5))
        thr2 = round(np.percentile(arr1[arr1 > 0], 99))
        arr1[arr1 < thr1] = thr1
        arr1[arr1 > thr2] = thr2
        arr1 = (arr1 - thr1) / (thr2 - thr1)
        arr1 = arr1 * 255.
        arr1 = np.uint8(arr1)
        arr1[arr1 == 0] = 1.
        arr2 = np.uint8(arr2)
        arr1 = arr1 * arr2
        dra_img.append(arr1)
    dra_img = np.array(dra_img)
    dra_img = np.rollaxis(dra_img, 0, 3)
    dra_img = np.uint8(dra_img)
    print('dra_img shape: ', dra_img.shape)
    b1 = dra_img[:, :, 0]
    b2 = dra_img[:, :, 1]
    b3 = dra_img[:, :, 2]
    b1 = minMax(b1)
    b2 = minMax(b2)
    b3 = minMax(b3)

    grey1 = (b1 + b2 + b3) / 3

    grey2 = grey1.copy()
    grey2[grey2 > 0] = 1
    grey = grey1.copy()
    grey = grey[grey != 0]
    thr1 = grey.mean() - 2 * grey.std()
    thr2 = grey.mean() + 2 * grey.std()
    grey1[grey1 < thr1] = thr1
    grey1[grey1 > thr2] = thr2
    grey1 = (grey1 - thr1) / (thr2 - thr1)
    sobelx = ndimage.sobel(grey1, 0)
    sobely = ndimage.sobel(grey1, 1)
    sobel = np.hypot(sobelx, sobely)
    print('Standard Div : ', sobel.std())
    print('Min : ', sobel.min(), 'Max : ', sobel.max())
    sobelMinStd = (sobel - sobel.min()) / (sobel.std())
    sobelMinStd = sobelMinStd * grey2
    sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))
    # io.imsave(OutDir + 'Sobel_Lut/' + fName, sobelMinStd)
    b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
    b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
    b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
    img = np.concatenate((b1, b2, b3), axis=-1)
    rgbsobelMinStd = np.concatenate((img, sobelMinStd), axis=-1)
    
    # Save output
    base_name = os.path.splitext(os.path.basename(image))[0]
    output_path = os.path.join(OutDir, f"{base_name}.tif")
    io.imsave(output_path, rgbsobelMinStd )
    change_projection(output_path, image)
    
    print(f"Processing complete. Output saved as: {output_path}")


def minMax(band):
    """Min-Max normalization to [0, 255]."""
    band = band.astype(np.float32)
    min_val = np.percentile(band[band > 0], 2.5)
    max_val = np.percentile(band[band > 0], 99)
    band = np.clip((band - min_val) / (max_val - min_val), 0, 1)
    return (band * 255).astype(np.uint8)

def add_sobel_filter_nir_sobel_min_max(image, outDir):
    try:
        img = io.imread(image)
        img2 = gdal.Open(image)
        arr = img2.ReadAsArray()

        # Ensure we have 4 channels (RGB + NIR)
        if arr.shape[0] != 4:
            raise ValueError("Input image must have 4 channels (RGB + NIR).")

        dra_img = []
        
        # Extract RGB channels and normalize them
        r, g, b = arr[0], arr[1], arr[2]
        b1 = minMax(r)
        b2 = minMax(g)
        b3 = minMax(b)

        # Create grayscale image from RGB for Sobel filtering
        grey1 = (b1 + b2 + b3) / 3

        # Sobel filter application  
        sobelx = ndimage.sobel(grey1, axis=0)
        sobely = ndimage.sobel(grey1, axis=1)
        sobel = np.hypot(sobelx, sobely)

        # Normalize Sobel output to range [0, 1]
        sobelMinStd = (sobel - sobel.min()) / (sobel.max() - sobel.min())
        
        

        # Process NIR channel with percentile normalization
        nir_channel = img[:, :, 3]
        
        # Percentile normalization for NIR channel
        nir_thr1 = round(np.percentile(nir_channel[nir_channel > 0], 2.5))
        nir_thr2 = round(np.percentile(nir_channel[nir_channel > 0], 97.5))
        nir_channel[nir_channel < nir_thr1] = nir_thr1
        nir_channel[nir_channel > nir_thr2] = nir_thr2
        nir_channel_normalized = (nir_channel - nir_thr1) / (nir_thr2 - nir_thr1)

        r = b1
        g = b2
        b = b3
        nir = nir_channel_normalized
        sobel_norm = sobelMinStd
        assert r.shape == g.shape == b.shape == nir.shape == sobel_norm.shape, "Shape mismatch among input channels!"

        # Concatenate RGB + NIR + Sobel to create a new image with all channels
        img_combined = np.stack([r, g, b, nir, sobel_norm], axis=-1)

        print("$$$$$",img_combined.shape)

        base_name = os.path.splitext(os.path.basename(image))[0]
        output_dir = outDir
        
        rgb_sobel_path = os.path.join(output_dir, f"{base_name}.tif")
        
        # io.imsave(rgb_sobel_path, img_combined.astype(np.float32))  # Save as float32 for precision
        save_multichannel_image(img_combined, rgb_sobel_path, image)
        
        change_projection(rgb_sobel_path, image)
        
        print("Applied Sobel Filter and Saved at ", str(rgb_sobel_path))
        
        return rgb_sobel_path
    
    except Exception as e:
        print(f"Error in add_sobel_filter_nir_sobel: {e}")
        return None



def save_multichannel_image(image, output_path, reference_path):
    """
    Save a multi-channel image properly with 5 channels.

    Args:
        image (np.ndarray): Multi-channel image (H, W, 5)
        output_path (str): Path to save the image.
        reference_path (str): Path to the reference image for metadata.
    """
    # Read metadata from reference image
    with rasterio.open(reference_path) as src:
        meta = src.meta.copy()

    # Update metadata to reflect 5 channels
    meta.update({
        'count': image.shape[-1],  # Set channel count to 5
        'dtype': 'float32'         # Ensure data type consistency
    })

    # Save the multi-channel image
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i in range(image.shape[-1]):
            dst.write(image[:, :, i], i + 1)

    print(f"✅ Multi-channel image saved at: {output_path}")

def minMax(arr):
    """Normalize the array to range [0, 1]."""
    return (arr - arr.min()) / (arr.max() - arr.min())




def process_folder(input_folder, output_folder):
    """
    Process all TIF files in an input folder
    
    Parameters:
    -----------
    input_folder : str
        Folder containing input TIF files
    output_folder : str
        Folder to save processed images
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all TIF files
    tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
    
    print(f"Found {len(tif_files)} TIF files to process")
    
    # Process each file
    for tif_file in tif_files:
        #add_old_sobel_filter(tif_file, output_folder)
        # print(tif_file)
        if (tif_file == '/workspace/original_tiff/part6_cog.tif'):
            print(f'processing {tif_file}')
            image_scaler(tif_file, output_folder)

        else : 
            print(f'not processing {tif_file}')
        # image_scaler_histogram_fixed(tif_file, output_folder)
        #add_nir_sobel_filter(tif_file, output_folder)
        #add_sobel_filter_nir_sobel_min_max(tif_file, output_folder)
    
    print("Processing complete!")

def main():
    # Prompt for preprocessing.pyinput and output folders
    input_folder = input("Enter input folder path: ").strip()
    output_folder = input("Enter output folder path: ").strip()
    
    #input_folder = "/sharedDisk/vinay/waterbody/512_dataset_training/512_NIR_tiles"
    #output_folder = "/sharedDisk/vinay/waterbody/512_dataset_training/512_NIR_tiles_sobel_added"
    
    # Validate paths
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    
    # Confirm with user
    print(f"\nInput folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    
    confirm = input("\nProceed with processing? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled by user.")
        return
    
    # Process the folder
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()

