#!/usr/bin/env python3
"""
VEGETATION DETECTION - MODEL COMPARISON & ENSEMBLE
Compare predictions from multiple models and create ensemble predictions
"""

import os
import sys
import argparse
import numpy as np
import rasterio
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class VegetationEnsemble:
    """Ensemble multiple vegetation detection predictions"""
    
    def __init__(self, output_dir="./ensemble_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_prediction(self, pred_path):
        """Load a vegetation prediction from TIFF"""
        with rasterio.open(pred_path) as src:
            # Last band is usually the prediction
            pred = src.read(src.count)
            return pred.astype(np.float32)
    
    def ensemble_average(self, prediction_paths, output_path, method='mean'):
        """
        Create ensemble prediction by averaging multiple models
        
        Args:
            prediction_paths: List of paths to predictions
            output_path: Output path for ensemble result
            method: 'mean', 'median', 'weighted', 'max', 'min'
        """
        print(f"\n🎯 Ensemble Prediction ({method})")
        print(f"{'='*60}")
        print(f"📊 Combining {len(prediction_paths)} predictions")
        
        # Load all predictions
        predictions = []
        profile = None
        
        for pred_path in tqdm(prediction_paths, desc="Loading predictions"):
            pred = self.load_prediction(pred_path)
            predictions.append(pred)
            
            # Get profile from first file
            if profile is None:
                with rasterio.open(pred_path) as src:
                    profile = src.profile
        
        predictions = np.array(predictions)
        print(f"   Shape of each prediction: {predictions[0].shape}")
        
        # Compute ensemble
        if method == 'mean':
            ensemble = np.mean(predictions, axis=0)
        elif method == 'median':
            ensemble = np.median(predictions, axis=0)
        elif method == 'max':
            ensemble = np.max(predictions, axis=0)
        elif method == 'min':
            ensemble = np.min(predictions, axis=0)
        elif method == 'weighted':
            # Weight by average confidence (higher confidence models weighted more)
            weights = np.array([np.mean(p) for p in predictions])
            weights = weights / weights.sum()
            ensemble = np.average(predictions, axis=0, weights=weights)
            print(f"   Weights: {weights}")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Save ensemble
        profile_out = profile.copy()
        profile_out.update(count=1, dtype='float32')
        
        with rasterio.open(output_path, 'w', **profile_out) as dst:
            dst.write(ensemble, 1)
        
        print(f"✅ Ensemble prediction saved: {output_path}")
        
        # Statistics
        print(f"\n📊 Ensemble Statistics:")
        print(f"   Min: {ensemble.min():.4f}")
        print(f"   Max: {ensemble.max():.4f}")
        print(f"   Mean: {ensemble.mean():.4f}")
        print(f"   Median: {np.median(ensemble):.4f}")
        print(f"   Std Dev: {ensemble.std():.4f}")
        
        return ensemble
    
    def create_binary_maps(self, prediction_path, thresholds=[0.3, 0.5, 0.7]):
        """
        Create binary vegetation maps at different thresholds
        
        Args:
            prediction_path: Path to ensemble prediction
            thresholds: List of thresholds to try
        """
        print(f"\n📊 Creating Binary Maps")
        print(f"{'='*60}")
        
        ensemble = self.load_prediction(prediction_path)
        
        with rasterio.open(prediction_path) as src:
            profile = src.profile
        
        profile_out = profile.copy()
        profile_out.update(count=1, dtype='uint8')
        
        results = {}
        
        for threshold in thresholds:
            binary = (ensemble > threshold).astype(np.uint8) * 255
            
            out_path = str(prediction_path).replace(
                '.tif',
                f'_binary_t{threshold:.2f}.tif'
            )
            
            with rasterio.open(out_path, 'w', **profile_out) as dst:
                dst.write(binary, 1)
            
            coverage = (binary > 0).sum() / binary.size * 100
            results[threshold] = {
                'path': out_path,
                'coverage': coverage
            }
            
            print(f"   Threshold {threshold}: {coverage:.2f}% coverage → {out_path}")
        
        return results
    
    def create_confidence_map(self, prediction_path):
        """
        Create color-coded confidence map
        Red = low confidence, Green = high confidence
        """
        print(f"\n🎨 Creating Confidence Visualization")
        print(f"{'='*60}")
        
        ensemble = self.load_prediction(prediction_path)
        
        # Create RGB confidence map
        height, width = ensemble.shape
        confidence_rgb = np.zeros((3, height, width), dtype=np.uint8)
        
        # Red channel: inverse confidence (high confidence = low red)
        confidence_rgb[0] = (255 * (1 - ensemble)).astype(np.uint8)
        
        # Green channel: confidence
        confidence_rgb[1] = (255 * ensemble).astype(np.uint8)
        
        # Blue channel: neutral
        confidence_rgb[2] = 128
        
        with rasterio.open(prediction_path) as src:
            profile = src.profile
        
        profile_out = profile.copy()
        profile_out.update(count=3, dtype='uint8')
        
        out_path = str(prediction_path).replace('.tif', '_confidence_map.tif')
        
        with rasterio.open(out_path, 'w', **profile_out) as dst:
            dst.write(confidence_rgb)
        
        print(f"✅ Confidence map saved: {out_path}")
        return out_path
    
    def compare_predictions(self, prediction_paths, output_csv=None):
        """
        Compare statistics across multiple predictions
        
        Args:
            prediction_paths: List of prediction paths
            output_csv: Path to save comparison CSV
        """
        print(f"\n📊 Comparing Predictions")
        print(f"{'='*60}")
        
        results = []
        
        for pred_path in prediction_paths:
            pred = self.load_prediction(pred_path)
            
            stats = {
                'model': Path(pred_path).stem,
                'min': float(pred.min()),
                'max': float(pred.max()),
                'mean': float(pred.mean()),
                'median': float(np.median(pred)),
                'std': float(pred.std()),
                'coverage_50': float((pred > 0.5).sum() / pred.size * 100),
                'coverage_30': float((pred > 0.3).sum() / pred.size * 100),
                'coverage_70': float((pred > 0.7).sum() / pred.size * 100),
            }
            
            results.append(stats)
            
            print(f"\n📌 {stats['model']}:")
            print(f"   Mean Confidence: {stats['mean']:.4f}")
            print(f"   Coverage (T=0.5): {stats['coverage_50']:.2f}%")
            print(f"   Coverage (T=0.3): {stats['coverage_30']:.2f}%")
            print(f"   Coverage (T=0.7): {stats['coverage_70']:.2f}%")
        
        # Save comparison
        if output_csv:
            import csv
            with open(output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n✅ Comparison saved to: {output_csv}")
        
        return results
    
    def difference_analysis(self, pred_path1, pred_path2, output_path):
        """
        Analyze difference between two predictions
        
        Args:
            pred_path1: First prediction path
            pred_path2: Second prediction path
            output_path: Path to save difference map
        """
        print(f"\n📊 Difference Analysis")
        print(f"{'='*60}")
        
        pred1 = self.load_prediction(pred_path1)
        pred2 = self.load_prediction(pred_path2)
        
        if pred1.shape != pred2.shape:
            print("❌ Predictions have different shapes")
            return None
        
        # Compute differences
        diff = np.abs(pred1 - pred2)
        
        with rasterio.open(pred_path1) as src:
            profile = src.profile
        
        profile_out = profile.copy()
        profile_out.update(count=1, dtype='float32')
        
        with rasterio.open(output_path, 'w', **profile_out) as dst:
            dst.write(diff, 1)
        
        print(f"\n📊 Difference Statistics:")
        print(f"   Mean Difference: {diff.mean():.4f}")
        print(f"   Max Difference: {diff.max():.4f}")
        print(f"   Agreement (diff < 0.1): {(diff < 0.1).sum() / diff.size * 100:.2f}%")
        
        print(f"\n✅ Difference map saved: {output_path}")
        return diff


def main():
    parser = argparse.ArgumentParser(
        description='Ensemble and compare vegetation detection predictions'
    )
    
    parser.add_argument('--predictions', nargs='+', required=True,
                       help='Paths to prediction TIFFs')
    parser.add_argument('--output_dir', default='./ensemble_results',
                       help='Output directory')
    parser.add_argument('--method', choices=['mean', 'median', 'max', 'min', 'weighted'],
                       default='mean', help='Ensemble method')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.3, 0.5, 0.7],
                       help='Thresholds for binary maps')
    parser.add_argument('--compare_only', action='store_true',
                       help='Only compare predictions, do not ensemble')
    
    args = parser.parse_args()
    
    # Validate input files
    for pred_path in args.predictions:
        if not os.path.exists(pred_path):
            print(f"❌ Prediction not found: {pred_path}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("🌳 VEGETATION DETECTION - ENSEMBLE & COMPARISON")
    print("="*70)
    
    ensemble = VegetationEnsemble(args.output_dir)
    
    if args.compare_only:
        # Compare only
        csv_path = os.path.join(args.output_dir, 'comparison.csv')
        ensemble.compare_predictions(args.predictions, csv_path)
    else:
        # Ensemble predictions
        ensemble_path = os.path.join(args.output_dir, 'ensemble_prediction.tif')
        ensemble_pred = ensemble.ensemble_average(
            args.predictions,
            ensemble_path,
            method=args.method
        )
        
        # Compare all predictions
        csv_path = os.path.join(args.output_dir, 'comparison.csv')
        ensemble.compare_predictions(args.predictions, csv_path)
        
        # Create binary maps
        ensemble.create_binary_maps(ensemble_path, args.thresholds)
        
        # Create confidence visualization
        ensemble.create_confidence_map(ensemble_path)
    
    print("\n" + "="*70)
    print("✅ Ensemble processing completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
