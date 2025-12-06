
import argparse
from ensemble_triton_pytorch_predictor_with_waterbody_test import EnsemblePredictor
import os
from skimage import io
from osgeo import gdal
import numpy as np
from scipy import ndimage
from pathlib import Path
import sys
import argparse
import rasterio
from clip_raster_with_tiles import clip_raster_with_tiles
from clip_raster_with_shapefile import clip_raster_with_shapefile
from merge_tif_with_shp import merge_with_shapefile

import rasterio
import numpy as np
from scipy import ndimage
from skimage import io
import gc
import rasterio.windows


def add_sobel_filter(image):
    try:
        # Open the image using rasterio for memory-efficient reading
        with rasterio.open(image) as src:
            profile = src.profile
            
            # Calculate optimal block size based on available memory
            block_size = int(1024 * 1024 * 1024)  # 1GB in bytes
            dtype_size = 4  # float32 size in bytes
            bands = src.count
            
            # Calculate rows per block
            pixels_per_block = block_size // (dtype_size * bands * src.width)
            block_height = min(pixels_per_block, src.height)
            
            # Initialize output arrays
            base_name = os.path.splitext(os.path.basename(image))[0]
            output_dir = os.path.dirname(image)
            rgb_sobel_path = os.path.join(output_dir, f"{base_name}_rgb_sobel.tif")
            
            # Create output file with same profile
            profile.update(count=4, dtype='float32')  # 3 bands + sobel
            
            with rasterio.open(rgb_sobel_path, 'w', **profile) as dst:
                # Process image in blocks
                for y in range(0, src.height, block_height):
                    # Calculate actual block height (handle last block)
                    current_height = min(block_height, src.height - y)
                    
                    # Read block
                    window = rasterio.windows.Window(0, y, src.width, current_height)
                    block_data = src.read(window=window)
                    
                    # Process block using original logic
                    dra_img = []
                    # Match original version's band order (2 - band)
                    for band in range(block_data.shape[0] - 1):
                        arr1 = block_data[2 - band].copy()
                        arr2 = arr1.copy()
                        arr2[arr2 > 0] = 1
                        
                        # Calculate percentiles for non-zero values
                        valid_data = arr1[arr1 > 0]
                        if len(valid_data) > 0:
                            thr1 = round(np.percentile(valid_data, 2.5))
                            thr2 = round(np.percentile(valid_data, 99))
                            
                            arr1[arr1 < thr1] = thr1
                            arr1[arr1 > thr2] = thr2
                            arr1 = (arr1 - thr1) / (thr2 - thr1)
                            arr1 = arr1 * 255.
                            
                        arr1 = np.uint8(arr1)
                        arr1[arr1 == 0] = 1
                        arr2 = np.uint8(arr2)
                        arr1 = arr1 * arr2
                        dra_img.append(arr1)
                    
                    # Convert to proper format matching original version
                    dra_img = np.array(dra_img)
                    dra_img = np.rollaxis(dra_img, 0, 3)
                    dra_img = np.uint8(dra_img)
                    
                    # Process exactly like original version
                    b1 = minMax(dra_img[:, :, 0])
                    b2 = minMax(dra_img[:, :, 1])
                    b3 = minMax(dra_img[:, :, 2])

                    grey1 = (b1 + b2 + b3) / 3
                    grey2 = grey1.copy()
                    grey2[grey2 > 0] = 1
                    
                    grey = grey1[grey1 != 0]
                    if len(grey) > 0:
                        thr1 = grey.mean() - 2 * grey.std()
                        thr2 = grey.mean() + 2 * grey.std()
                        
                        grey1[grey1 < thr1] = thr1
                        grey1[grey1 > thr2] = thr2
                        grey1 = (grey1 - thr1) / (thr2 - thr1)
                    
                    sobelx = ndimage.sobel(grey1, 0)
                    sobely = ndimage.sobel(grey1, 1)
                    sobel = np.hypot(sobelx, sobely)
                    
                    sobelMinStd = (sobel - sobel.min()) / (sobel.std()) if sobel.std() != 0 else sobel
                    sobelMinStd = sobelMinStd * grey2
                    
                    # Stack bands in same order as original
                    b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
                    b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
                    b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
                    sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))
                    
                    output_block = np.concatenate([b1, b2, b3, sobelMinStd], axis=-1)
                    output_block = np.moveaxis(output_block, -1, 0)
                    
                    # Write block
                    dst.write(output_block, window=window)
                    
                    # Force memory cleanup
                    del block_data, dra_img, output_block
                    gc.collect()
            
            # Change projection
            change_projection(rgb_sobel_path, image)
            
            print("Applied Sobel Filter and Saved at ", str(rgb_sobel_path))
            return rgb_sobel_path
    
    except Exception as e:
        print(sys.exc_info(), Path(__file__), sys._getframe().f_code.co_name, e)
        return 1


def add_sobel_filter_new(image):
    try:
        # Open the image using rasterio for memory-efficient reading
        with rasterio.open(image) as src:
            profile = src.profile
            
            # Calculate optimal block size based on available memory
            # Aim to process ~1GB at a time
            block_size = int(1024 * 1024 * 1024)  # 1GB in bytes
            dtype_size = 4  # float32 size in bytes
            bands = src.count
            
            # Calculate rows per block
            pixels_per_block = block_size // (dtype_size * bands * src.width)
            block_height = min(pixels_per_block, src.height)
            
            # Initialize output arrays
            base_name = os.path.splitext(os.path.basename(image))[0]
            output_dir = os.path.dirname(image)
            rgb_sobel_path = os.path.join(output_dir, f"{base_name}_rgb_sobel.tif")
            
            # Create output file with same profile
            profile.update(count=4, dtype='float32')  # 3 bands + sobel
            
            with rasterio.open(rgb_sobel_path, 'w', **profile) as dst:
                # Process image in blocks
                for y in range(0, src.height, block_height):
                    # Calculate actual block height (handle last block)
                    current_height = min(block_height, src.height - y)
                    
                    # Read block
                    window = rasterio.windows.Window(0, y, src.width, current_height)
                    block_data = src.read(window=window)
                    
                    # Process block using original logic
                    dra_img = []
                    for band in range(block_data.shape[0]):
                        arr1 = block_data[band].copy()
                        arr2 = arr1.copy()
                        arr2[arr2 > 0] = 1
                        
                        # Calculate percentiles for non-zero values
                        valid_data = arr1[arr1 > 0]
                        if len(valid_data) > 0:
                            thr1 = np.percentile(valid_data, 2.5)
                            thr2 = np.percentile(valid_data, 99)
                            
                            arr1[arr1 < thr1] = thr1
                            arr1[arr1 > thr2] = thr2
                            arr1 = (arr1 - thr1) / (thr2 - thr1)
                            arr1 = arr1 * 255.
                            
                        arr1 = np.uint8(arr1)
                        arr1[arr1 == 0] = 1
                        arr2 = np.uint8(arr2)
                        arr1 = arr1 * arr2
                        dra_img.append(arr1)
                    
                    dra_img = np.array(dra_img)
                    dra_img = np.rollaxis(dra_img, 0, 3)
                    dra_img = np.uint8(dra_img)
                    
                    # Apply minMax and calculate sobel
                    b1 = minMax(dra_img[:, :, 0])
                    b2 = minMax(dra_img[:, :, 1])
                    b3 = minMax(dra_img[:, :, 2])
                    
                    grey1 = (b1 + b2 + b3) / 3
                    grey2 = grey1.copy()
                    grey2[grey2 > 0] = 1
                    
                    grey = grey1[grey1 != 0]
                    if len(grey) > 0:
                        thr1 = grey.mean() - 2 * grey.std()
                        thr2 = grey.mean() + 2 * grey.std()
                        
                        grey1[grey1 < thr1] = thr1
                        grey1[grey1 > thr2] = thr2
                        grey1 = (grey1 - thr1) / (thr2 - thr1)
                    
                    sobelx = ndimage.sobel(grey1, 0)
                    sobely = ndimage.sobel(grey1, 1)
                    sobel = np.hypot(sobelx, sobely)
                    
                    sobelMinStd = (sobel - sobel.min()) / (sobel.std()) if sobel.std() != 0 else sobel
                    sobelMinStd = sobelMinStd * grey2
                    
                    # Prepare output block
                    output_block = np.stack([b1, b2, b3, sobelMinStd])
                    
                    # Write block
                    dst.write(output_block, window=window)
                    
                    # Force memory cleanup
                    del block_data, dra_img, output_block
                    gc.collect()
            
            # Change projection
            change_projection(rgb_sobel_path, image)
            
            print("Applied Sobel Filter and Saved at ", str(rgb_sobel_path))
            return rgb_sobel_path
    
    except Exception as e:
        print(sys.exc_info(), Path(__file__), sys._getframe().f_code.co_name, e)
        return 1

def minMax(band):
    #print('mmband', band.min(), band.max())
    #minimum = 1
    #minimum =0.0
    #maximum = 255.0
    band = np.float32(band)       
    band = (band - band.min()) / (band.max() - band.min())  
    #band = (band - minimum) / (maximum - minimum)
    return(band) 

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

def add_sobel_filter_old(image):
    try :
        img = io.imread(image)
        img2 = gdal.Open(image)
        arr = img2.ReadAsArray()

        dra_img = []
        for band in range(img.shape[2] - 1):
            arr1 = img[:, :,2 - band]
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
        sobelMinStd = (sobel - sobel.min()) / (sobel.std())
        sobelMinStd = sobelMinStd * grey2
        sobelMinStd = np.reshape(sobelMinStd, (b1.shape[0], b1.shape[1], 1))
        base_name = os.path.splitext(os.path.basename(image))[0]
        output_dir = os.path.dirname(image)
        rgb_sobel_path = os.path.join(output_dir, f"{base_name}_rgb_sobel.tif")
        b1 = np.reshape(b1, (b1.shape[0], b1.shape[1], 1))
        b2 = np.reshape(b2, (b1.shape[0], b1.shape[1], 1))
        b3 = np.reshape(b3, (b1.shape[0], b1.shape[1], 1))
        img = np.concatenate((b1, b2, b3), axis=-1)
        rgbsobelMinStd = np.concatenate((img, sobelMinStd), axis=-1)
        io.imsave(rgb_sobel_path, rgbsobelMinStd)
        change_projection(rgb_sobel_path, image)
        print("Applied Sobel Filter and Saved at ", str(rgb_sobel_path))
        return rgb_sobel_path
    except Exception as e:
        print(sys.exc_info(), Path(__file__), sys._getframe().f_code.co_name, e)
        return 1
def set_gpu():
    # List available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')

    if len(physical_devices) > 1:
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
    else:
        print("Only one GPU available")


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction for multi-class segmentation')
    
    parser.add_argument('--input_image', required=True,
                      help='Path to input image file')
    parser.add_argument('--output_path', required=True,
                      help='Path for output classification TIF')
    parser.add_argument('--building_model', required=False,
                      help='Path to building detection model weights')
    parser.add_argument('--vegetation_model', required=False,
                      help='Path to vegetation detection model weights')
    parser.add_argument('--water_model', required=False,
                      help='Path to water body detection model weights')
    parser.add_argument('--config', default='/home/gpuserver/vinay/attention_unet/config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--shapefile_path', required=False,
                      help='Path to shapefile for clipping')
    
    args = parser.parse_args()
    
    
    # Initialize predictor
    predictor = EnsemblePredictor(config_path=args.config)
    
    # Load all models with their respective thresholds
    model_paths = {
        'building': ("TEMP_v2", 0.343, "input","output"),
        #'building': ("UNET_SENET154", 0.343, "input","sigmoid"),    
        #'building': ("UNET_Bld", 0.444, "input_1","conv2d_41"),
        'vegetation': ("UNET_New_Veg",0.25, "input_3","model_1"),
        'water': ("UNET_WB", 0.9, "input_1","conv2d_41")
    }

    predictor.load_models(model_paths)
    
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    # Clip input image before processing
    input_filename = os.path.splitext(os.path.basename(args.input_image))[0]
    clipped_input = os.path.join(os.path.dirname(args.output_path), f"{input_filename}_clipped.tif")
    print(f"Clipping input raster with shapefile...")
    #clip_raster_with_tiles(
    #    args.input_image,
    #    args.shapefile_path,
    #    clipped_input,
    #    tile_size=256,
    #    compress=False
    #)

    # Add sobel filter to clipped input
    # sobel_input = add_sobel_filter_old(clipped_input)
    sobel_input = args.input_image
    # sobel_input = args.input_image
    # Run prediction
    stats = predictor.predict(sobel_input, args.output_path)
    
    # prediction_output = os.path.join(os.path.dirname(args.output_path), f"{input_filename}_mutliclass.tif")
    prediction_output = os.path.join(os.path.dirname(args.output_path),f"{os.path.splitext(os.path.basename(args.output_path))[0]}_multiclass.tif")
    #month = input_filename[:2]
    #shp_dir = os.path.dirname(os.path.dirname(args.input_image))
    #if(int(month) < 7):
     #   waterbody_shp_path = os.path.join(shp_dir, "01.shp")
    #else:
     #   waterbody_shp_path = os.path.join(shp_dir, "06.shp")

    #merged_path = os.path.join(os.path.dirname(args.output_path), f"{input_filename}_merged.tif")
    #merged_output = merge_with_shapefile(prediction_output, waterbody_shp_path, merged_path)

    # Clip the output prediction
    output_filename = os.path.splitext(os.path.basename(args.output_path))[0]
    final_output = os.path.join(os.path.dirname(args.output_path), f"{output_filename}.tif")
    print(f"Clipping output prediction with shapefile...")
    #clip_raster_with_shapefile(
     #   prediction_output,
     #   args.shapefile_path,
     #   final_output,
     #   chunk_size=4096,
     #   compress=False
    #)
    
    print("\nPrediction completed successfully!")
    print(f"Final clipped output saved to: {final_output}")
    print("\nClass Statistics:")
    print(f"Buildings: {stats['building_percentage']:.2f}%")
    print(f"Vegetation: {stats['vegetation_percentage']:.2f}%")
    print(f"Water: {stats['water_percentage']:.2f}%")

if __name__ == "__main__":
    #set_gpu()
    main()
