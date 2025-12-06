import sys
import os
import re
from osgeo import gdal
from shapely.geometry import box
import json
import logging
import numpy as np
import rasterio as rio
from rasterio.errors import RasterioIOError
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import argparse
import matplotlib.pyplot as plt
# import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def log_raster_info(raster_path, label=""):
    """Log detailed information about a raster file."""
    with rio.open(raster_path) as src:
        logging.info(f"{label} Raster Info:")
        logging.info(f"  CRS: {src.crs}")
        logging.info(f"  Bounds: {src.bounds}")
        logging.info(f"  Resolution: {src.res}")
        logging.info(f"  Size: {src.width}x{src.height}")

def check_raster(file_path):
    """Validate and load a raster file."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File does not exist: {file_path}")
        raster_ds = rio.open(file_path)
        if raster_ds.count != 1:
            raise ValueError(f"Expected a single-band raster, but found {raster_ds.count} bands.")
        return file_path
    except (RasterioIOError, FileNotFoundError, ValueError) as e:
        raise ValueError(f"Error validating raster file {file_path}: {e}")

def reproject_raster(pred_file, gt_file):
    """Reproject prediction raster to match ground truth CRS and resolution."""
    try:
        with rio.open(gt_file) as gt_ras, rio.open(pred_file) as pred_ras:
            # Log initial states
            logging.info("Before reprojection:")
            log_raster_info(pred_file, "Prediction")
            log_raster_info(gt_file, "Ground Truth")
            
            output_path = re.sub(r'\.tif$', '_reprojected.tif', pred_file)
            
            # Calculate destination transform and dimensions 
            dst_transform, dst_width, dst_height = calculate_default_transform(
                pred_ras.crs, gt_ras.crs, 
                pred_ras.width, pred_ras.height, 
                *pred_ras.bounds
            )
            
            # Prepare metadata for output
            meta = pred_ras.meta.copy()
            meta.update({
                'crs': gt_ras.crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'driver': 'GTiff'
            })
            
            # Remove any unsupported options
            meta.pop('resolution', None)
            
            with rio.open(output_path, 'w', **meta) as dst:
                reproject(
                    source=rio.band(pred_ras, 1),
                    destination=rio.band(dst, 1),
                    src_transform=pred_ras.transform,
                    src_crs=pred_ras.crs,
                    dst_transform=dst_transform,
                    dst_crs=gt_ras.crs,
                    resampling=Resampling.nearest
                )
            
            # Log final state
            logging.info("After reprojection:")
            log_raster_info(output_path, "Reprojected Prediction")
            
            return output_path
    except Exception as e:
        raise RuntimeError(f"Reprojection failed: {str(e)}")

def threshold_mask(mask):
    """Apply a threshold to convert values to binary."""
    return np.where(mask > 0, 1, 0)

def raster_read(ras_file):
    """Read and threshold a raster file."""
    try:
        with rio.open(ras_file) as in_ras:
            data = in_ras.read(1)
            # Ensure array is in correct orientation
            data = data[::-1] if in_ras.transform.e > 0 else data
            return threshold_mask(data)
    except Exception as e:
        raise RuntimeError(f"Failed to read raster file: {e}")

def calculate_iou(predicted_mask, ground_truth_mask):
    """Calculate Intersection over Union (IoU)."""
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    union = np.logical_or(predicted_mask, ground_truth_mask).sum()
    return intersection / union if union != 0 else 0.0

def calculate_dice(predicted_mask, ground_truth_mask):
    """Calculate Dice coefficient."""
    intersection = np.logical_and(predicted_mask, ground_truth_mask).sum()
    return (2.0 * intersection) / (predicted_mask.sum() + ground_truth_mask.sum()) if (predicted_mask.sum() + ground_truth_mask.sum()) != 0 else 0.0

def calculate_f1_score(predicted_mask, ground_truth_mask):
    """Calculate F1 score."""
    true_positive = np.logical_and(predicted_mask, ground_truth_mask).sum()
    false_positive = np.logical_and(predicted_mask, np.logical_not(ground_truth_mask)).sum()
    false_negative = np.logical_and(np.logical_not(predicted_mask), ground_truth_mask).sum()

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0

    return (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.0

def cleanup_files(files):
    """Clean up temporary files."""
    for file in files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {file}: {e}")

def calculate_accuracy_metrics(predicted_mask, ground_truth_mask):
    """Calculate all accuracy metrics."""
    # if predicted_mask.shape != ground_truth_mask.shape:
    #         raise ValueError(f"Shape mismatch: predicted {predicted_mask.shape} vs ground truth {ground_truth_mask.shape}")
    iou = calculate_iou(predicted_mask, ground_truth_mask)
    dice = calculate_dice(predicted_mask, ground_truth_mask)
    f1 = calculate_f1_score(predicted_mask, ground_truth_mask)
     
    # Calculate additional metrics for detailed analysis
    true_positive = np.logical_and(predicted_mask, ground_truth_mask).sum()
    false_positive = np.logical_and(predicted_mask, np.logical_not(ground_truth_mask)).sum()
    false_negative = np.logical_and(np.logical_not(predicted_mask), ground_truth_mask).sum()
    true_negative = np.logical_and(np.logical_not(predicted_mask), np.logical_not(ground_truth_mask)).sum()
    
    total = true_positive + true_negative + false_positive + false_negative
    
    metrics = {
        "IoU": round(iou, 4),
        "Dice_Coefficient": round(dice, 4),
        "F1_Score": round(f1, 4),
        "True_Positive": int(true_positive),
        "False_Positive": int(false_positive),
        "False_Negative": int(false_negative),
        "True_Negative": int(true_negative),
        "Total_Pixels": int(total)
    }
    
    return metrics

def projection_and_resample_check(pred_file, gt_file):
    """Ensure the prediction raster has the same CRS and resolution as the ground truth."""
    try:
        with rio.open(pred_file) as pred_ras, rio.open(gt_file) as gt_ras:
            pred_crs = pred_ras.crs
            gt_crs = gt_ras.crs
            pred_res = pred_ras.res
            gt_res = gt_ras.res

            if pred_crs != gt_crs or pred_res != gt_res:
                logging.info("Reprojecting and resampling predicted raster")
                pred_crs_file = re.sub(r'\.tif$', '_aligned.tif', pred_file)
                gdal.Warp(
                    pred_crs_file, pred_file,
                    dstSRS=f"EPSG:{gt_crs.to_epsg()}",
                    xRes=gt_res[0], yRes=gt_res[1],
                    resampleAlg="bilinear",
                    options=["COMPRESS=DEFLATE"]
                )
                return pred_crs_file
            return pred_file
    except Exception as e:
        raise RuntimeError(f"Projection and resampling check failed: {e}")

def intersec_check(gt_ras_path, pred_ras_path):
    """Check the intersection percentage and handle cases where prediction extends beyond ground truth."""
    try:
        with rio.open(gt_ras_path) as gt_ras, rio.open(pred_ras_path) as pred_ras:
            gt_box = box(*gt_ras.bounds)
            pred_box = box(*pred_ras.bounds)

            if gt_box.intersects(pred_box):
                intersection_area = gt_box.intersection(pred_box).area
                gt_area = gt_box.area
                pred_area = pred_box.area

                # Calculate intersection percentages correctly
                inter_gt_percentage = (intersection_area / gt_area) * 100
                inter_pred_percentage = (intersection_area / pred_area) * 100

                # Log the corrected percentages
                logging.info(f"Percentage of Ground Truth covered by Prediction: {inter_gt_percentage:.2f}%")
                logging.info(f"Percentage of Prediction covered by Ground Truth: {inter_pred_percentage:.2f}%")

                return inter_gt_percentage, inter_pred_percentage
            else:
                logging.warning("The bounds of the rasters do not intersect.")
                return 0.0, 0.0
    except Exception as e:
        raise RuntimeError(f"Intersection check failed: {e}")


def resolution_check(pred_file, gt_file):
    try:
        with rio.open(pred_file) as pred_ras, rio.open(gt_file) as gt_ras:
            pred_res = pred_ras.res
            gt_res = gt_ras.res
            
            if pred_res != gt_res:
                logging.info(
                    f"Resolution mismatch detected: Predicted resolution {pred_res}, Ground truth resolution {gt_res}"
                )
                
                resampled_gt_file = re.sub(r'\.tif$', '_resampled.tif', gt_file)
                
                # Get exact dimensions from prediction raster
                dst_shape = (pred_ras.height, pred_ras.width)
                
                meta = gt_ras.meta.copy()
                meta.update({
                    "height": dst_shape[0],
                    "width": dst_shape[1],
                    "transform": pred_ras.transform
                })
                
                with rio.open(resampled_gt_file, "w", **meta) as dst:
                    reproject(
                        source=rio.band(gt_ras, 1),
                        destination=rio.band(dst, 1),
                        src_transform=gt_ras.transform,
                        src_crs=gt_ras.crs,
                        dst_transform=pred_ras.transform,
                        dst_crs=pred_ras.crs,
                        dst_shape=dst_shape,  # Force exact shape match
                        resampling=Resampling.nearest
                    )
                
                logging.info(f"Resampled Ground Truth saved at: {resampled_gt_file}")
                return resampled_gt_file
            else:
                return gt_file
                
    except Exception as e:
        raise RuntimeError(f"Resolution check failed: {e}")

def clip_raster(base_raster, target_raster):
    try:
        with rio.open(base_raster) as base, rio.open(target_raster) as target:
            # Get target shape
            target_shape = (target.height, target.width)
            
            # Calculate window from bounds
            window = rio.windows.from_bounds(
                *target.bounds,
                transform=base.transform
            )
            
            # Update metadata
            meta = base.meta.copy()
            meta.update({
                "height": target_shape[0],
                "width": target_shape[1],
                "transform": target.transform
            })
            
            clipped_raster_path = re.sub(r'\.tif$', '_clipped.tif', base_raster)
            
            with rio.open(clipped_raster_path, "w", **meta) as out_raster:
                reproject(
                    source=rio.band(base, 1),
                    destination=rio.band(out_raster, 1),
                    src_transform=base.transform,
                    src_crs=base.crs,
                    dst_transform=target.transform,
                    dst_crs=target.crs,
                    dst_shape=target_shape,
                    resampling=Resampling.nearest
                )
            
            logging.info(f"Clipped raster saved at: {clipped_raster_path}")
            return clipped_raster_path
            
    except Exception as e:
        raise RuntimeError(f"Error clipping raster: {e}")

def align_rasters_strict(pred_file, gt_file):
    """
    Strictly align the ground truth raster to match prediction raster dimensions.
    """
    try:
        with rio.open(pred_file) as pred_ras:
            pred_shape = (pred_ras.height, pred_ras.width)
            pred_bounds = pred_ras.bounds
            pred_transform = pred_ras.transform
            pred_crs = pred_ras.crs

        aligned_gt_file = re.sub(r'\.tif$', '_aligned.tif', gt_file)
        
        with rio.open(gt_file) as gt_ras:
            meta = gt_ras.meta.copy()
            meta.update({
                'height': pred_shape[0],
                'width': pred_shape[1],
                'transform': pred_transform,
                'crs': pred_crs
            })
            
            with rio.open(aligned_gt_file, 'w', **meta) as dst:
                reproject(
                    source=rio.band(gt_ras, 1),
                    destination=rio.band(dst, 1),
                    src_transform=gt_ras.transform,
                    src_crs=gt_ras.crs,
                    dst_transform=pred_transform,
                    dst_crs=pred_crs,
                    dst_shape=pred_shape,
                    resampling=Resampling.nearest
                )
        
        logging.info(f"Aligned ground truth saved at: {aligned_gt_file}")
        return aligned_gt_file
        
    except Exception as e:
        raise RuntimeError(f"Error in strict alignment: {e}")


def clip_raster_old(base_raster, target_raster):
    """
    Clip the base raster to match the bounds of the target raster.

    Args:
        base_raster (str): Path to the base raster (ground truth).
        target_raster (str): Path to the target raster (prediction).

    Returns:
        str: Path to the clipped raster.
    """
    try:
        with rio.open(base_raster) as base, rio.open(target_raster) as target:
            # Convert raster bounds to Shapely geometries
            base_box = box(*base.bounds)
            target_box = box(*target.bounds)

            # Calculate intersection
            intersection_box = base_box.intersection(target_box)

            if intersection_box.is_empty:
                raise ValueError("The rasters do not overlap.")

            # Convert the intersection box back to raster bounds
            intersection_bounds = (
                intersection_box.bounds[0],  # min_x
                intersection_box.bounds[1],  # min_y
                intersection_box.bounds[2],  # max_x
                intersection_box.bounds[3]   # max_y
            )

            # Calculate the window for the intersection
            window = rio.windows.from_bounds(
                *intersection_bounds,
                transform=base.transform
            )

            # Update metadata for the clipped raster
            meta = base.meta.copy()
            meta.update({
                "width": window.width,
                "height": window.height,
                "transform": rio.windows.transform(window, base.transform)
            })

            # Save the clipped raster
            clipped_raster_path = re.sub(r'\.tif$', '_clipped.tif', base_raster)
            with rio.open(clipped_raster_path, "w", **meta) as out_raster:
                out_raster.write(base.read(window=window))

            logging.info(f"Clipped raster saved at: {clipped_raster_path}")
            return clipped_raster_path
    except Exception as e:
        raise RuntimeError(f"Error clipping raster: {e}")

def align_rasters(reference_raster, target_raster):
    """
    Modified version to handle shape mismatches
    """
    try:
        with rio.open(reference_raster) as ref, rio.open(target_raster) as tgt:
            # Force exact dimensions
            dst_shape = (ref.height, ref.width)
            
            # Update metadata
            meta = ref.meta.copy()
            aligned_raster_path = re.sub(r'\.tif$', '_aligned.tif', target_raster)
            
            with rio.open(aligned_raster_path, "w", **meta) as dst:
                reproject(
                    source=rio.band(tgt, 1),
                    destination=rio.band(dst, 1),
                    src_transform=tgt.transform,
                    src_crs=tgt.crs,
                    dst_transform=ref.transform,
                    dst_crs=ref.crs,
                    dst_shape=dst_shape,
                    resampling=Resampling.nearest
                )
            return aligned_raster_path
    except Exception as e:
        raise RuntimeError(f"Error aligning rasters: {e}")  
    

# ---------------- Confusion Matrix Function ---------------- #
def save_confusion_matrix(tp, fp, fn, tn, ground_truth_path, filename="confusion_matrix.png"):
    """Save the confusion matrix as an image file in the ground truth folder."""

    cm = np.array([[tn, fp],
                   [fn, tp]])

    labels = ["Negative", "Positive"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save inside ground truth folder
    gt_folder = os.path.dirname(ground_truth_path)
    save_path = os.path.join(gt_folder, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Confusion matrix saved as {save_path}")


def crop_to_overlap(pred_mask, gt_mask):
    """
    Crop prediction and ground truth masks to their common overlapping area.
    Assumes both arrays have the same height but slightly different widths.
    """
    min_height = min(pred_mask.shape[0], gt_mask.shape[0])
    min_width = min(pred_mask.shape[1], gt_mask.shape[1])

    pred_cropped = pred_mask[:min_height, :min_width]
    gt_cropped = gt_mask[:min_height, :min_width]

    return pred_cropped, gt_cropped

def main():
    # if len(sys.argv) != 3:
    #     logging.error("Invalid number of arguments. Usage: python script.py <predicted_file> <ground_truth_file>")
    #     print(json.dumps({"error": "Invalid number of arguments"}))
    #     return
    
    parser = argparse.ArgumentParser(description='Ensemble prediction for multi-class segmentation')
    
    parser.add_argument('--predicted_file', required=True,
                      help='Path to predicted tiff file')
    parser.add_argument('--ground_truth_file', required=True,
                      help='Path for groundtruth classification TIF')

    args = parser.parse_args()

    try:

        pred_file = check_raster(args.predicted_file)
        gt_file = check_raster(args.ground_truth_file)

        pred_file = projection_and_resample_check(pred_file, gt_file)

        pred_file = reproject_raster(pred_file, gt_file)

        predicted_mask = raster_read(pred_file)
        ground_truth_mask = raster_read(gt_file)

        predicted_mask, ground_truth_mask = crop_to_overlap(predicted_mask, ground_truth_mask)

        metrics = calculate_accuracy_metrics(predicted_mask, ground_truth_mask)
        print(json.dumps(metrics, indent=2))

        # save_confusion_matrix(
        #     metrics["True_Positive"], metrics["False_Positive"],
        #     metrics["False_Negative"], metrics["True_Negative"],
        #     ground_truth_path=args.ground_truth_file,
        #     filename="building_detection_confusion_matrix.png"
        # )

        # Check intersection
        inter_gt_percentage, inter_pred_percentage = intersec_check(gt_file, pred_file)
        logging.info(f"Percentage of Ground Truth covered by Prediction: {inter_gt_percentage:.2f}%")
        logging.info(f"Percentage of Prediction covered by Ground Truth: {inter_pred_percentage:.2f}%")

        if inter_gt_percentage < 30:
            logging.warning("Low ground truth coverage. Results may not be reliable.")

        # # Strict alignment to prediction raster
        # aligned_gt_file = align_rasters_strict(pred_file, gt_file)
        
        # # Read and verify shapes
        # with rio.open(pred_file) as pred_ras, rio.open(aligned_gt_file) as gt_ras:
        #     if (pred_ras.height, pred_ras.width) != (gt_ras.height, gt_ras.width):
        #         raise ValueError(f"Shape mismatch after alignment: pred {(pred_ras.height, pred_ras.width)} vs gt {(gt_ras.height, gt_ras.width)}")
        
        # # Calculate metrics
        # predicted_mask = raster_read(pred_file)
        # ground_truth_mask = raster_read(aligned_gt_file)
        
        # metrics = calculate_accuracy_metrics(predicted_mask, ground_truth_mask)
        # print(json.dumps(metrics))

        # # Cleanup
        # if os.path.exists(aligned_gt_file):
        #     os.remove(aligned_gt_file)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()

