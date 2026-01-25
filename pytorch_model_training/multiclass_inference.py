"""
Multi-Class Road Segmentation Inference Script
Predict road types on new tiles using trained model
"""

import torch
import torch.nn.functional as F
import numpy as np
import rasterio
from pathlib import Path
from pytorch_backbone_model_v2 import build_unet_resnet50
from tqdm import tqdm
import argparse


class MultiClassRoadSegmentor:
    """Inference pipeline for multi-class road segmentation"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.class_names = ['Background', 'Thar Road', 'CC Road', 'Mud/Gravel Road']
        
    def _load_model(self, model_path):
        """Load trained model"""
        print(f"🔄 Loading model from {model_path}")
        model = build_unet_resnet50(num_classes=4, input_size=256, freeze_backbone=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print("✓ Model loaded successfully")
        return model
    
    def preprocess_image(self, image_path, input_size=256):
        """Load and preprocess satellite image"""
        with rasterio.open(image_path) as src:
            image = src.read()  # (C, H, W)
            
            # Normalize each channel
            for j in range(src.count):
                channel = image[j]
                min_val = np.percentile(channel, 2)
                max_val = np.percentile(channel, 98)
                
                if max_val > min_val:
                    image[j] = np.clip((channel - min_val) / (max_val - min_val + 1e-7), 0, 1)
                else:
                    image[j] = np.zeros_like(channel)
            
            image = image.astype(np.float32)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)
        return image_tensor.to(self.device)
    
    def predict(self, image_path, return_confidence=True):
        """
        Predict road classes for image
        
        Returns:
            - predictions: (H, W) array with class indices [0, 1, 2, 3]
            - confidence: (H, W) array with max probability per pixel (optional)
        """
        image_tensor = self.preprocess_image(image_path)
        
        with torch.no_grad():
            logits = self.model(image_tensor)  # (1, 4, H, W)
            
            # Get class predictions
            predictions = torch.argmax(logits, dim=1)[0].cpu().numpy()  # (H, W)
            
            if return_confidence:
                # Get confidence scores
                probabilities = F.softmax(logits, dim=1)[0].cpu()  # (4, H, W)
                confidence = torch.max(probabilities, dim=0)[0].numpy()  # (H, W)
                return predictions, confidence
        
        return predictions
    
    def predict_batch(self, image_dir, output_dir=None, threshold=0.5):
        """
        Predict on multiple images in directory
        
        Args:
            image_dir: Directory containing .tif files
            output_dir: Save predictions here (optional)
            threshold: Confidence threshold for output
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.tif'))
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for image_path in tqdm(image_files, desc="Predicting"):
            predictions, confidence = self.predict(str(image_path))
            
            # Save results
            results[image_path.stem] = {
                'predictions': predictions,
                'confidence': confidence
            }
            
            # Save prediction GeoTIFF if output_dir specified
            if output_dir:
                self._save_prediction(
                    predictions, 
                    confidence,
                    str(image_path),
                    str(output_dir / f"{image_path.stem}_pred.tif")
                )
                
                # Save confidence map
                self._save_confidence(
                    confidence,
                    str(output_dir / f"{image_path.stem}_conf.tif")
                )
        
        return results
    
    def _save_prediction(self, predictions, confidence, ref_image_path, output_path):
        """Save predictions as GeoTIFF preserving georeference"""
        with rasterio.open(ref_image_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.uint8, count=1)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(predictions.astype(np.uint8), 1)
        
        print(f"  ✓ Saved: {output_path}")
    
    def _save_confidence(self, confidence, output_path):
        """Save confidence map"""
        profile = {
            'driver': 'GTiff',
            'dtype': rasterio.float32,
            'width': confidence.shape[1],
            'height': confidence.shape[0],
            'count': 1,
            'crs': 'EPSG:3857'
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write((confidence * 255).astype(np.uint8), 1)
    
    def get_class_distribution(self, predictions):
        """Get pixel count per class"""
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = {}
        
        for class_id, count in zip(unique, counts):
            if class_id < len(self.class_names):
                distribution[self.class_names[class_id]] = count
        
        return distribution
    
    def print_statistics(self, predictions, confidence):
        """Print prediction statistics"""
        print(f"\n📊 Prediction Statistics:")
        print(f"{'='*50}")
        
        # Class distribution
        dist = self.get_class_distribution(predictions)
        total_pixels = predictions.size
        
        print(f"\n🗺️  Class Distribution:")
        for class_name, count in dist.items():
            percentage = (count / total_pixels) * 100
            print(f"  • {class_name}: {count:,} pixels ({percentage:.2f}%)")
        
        # Confidence statistics
        print(f"\n🎯 Confidence Statistics:")
        print(f"  • Mean: {confidence.mean():.4f}")
        print(f"  • Std:  {confidence.std():.4f}")
        print(f"  • Min:  {confidence.min():.4f}")
        print(f"  • Max:  {confidence.max():.4f}")
        
        # Pixels with high confidence (>0.9)
        high_conf = (confidence > 0.9).sum()
        print(f"  • High Confidence (>0.9): {high_conf:,} ({high_conf/total_pixels*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Multi-class road segmentation inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--image_path', help='Single image for prediction')
    parser.add_argument('--image_dir', help='Directory of images for batch prediction')
    parser.add_argument('--output_dir', help='Directory to save predictions')
    parser.add_argument('--device', default='cuda', help='Device: cuda or cpu')
    args = parser.parse_args()
    
    # Initialize segmentor
    segmentor = MultiClassRoadSegmentor(args.model_path, device=args.device)
    
    if args.image_path:
        # Single image prediction
        print(f"\n🖼️  Predicting single image: {args.image_path}")
        predictions, confidence = segmentor.predict(args.image_path)
        segmentor.print_statistics(predictions, confidence)
        
        # Optionally save
        if args.output_dir:
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            segmentor._save_prediction(
                predictions, 
                confidence, 
                args.image_path,
                str(output_path / "prediction.tif")
            )
    
    elif args.image_dir:
        # Batch prediction
        print(f"\n📂 Predicting batch from: {args.image_dir}")
        results = segmentor.predict_batch(args.image_dir, args.output_dir)
        print(f"\n✓ Processed {len(results)} images")
    
    else:
        print("Specify either --image_path or --image_dir")


if __name__ == "__main__":
    main()
