#!/usr/bin/env python3
"""
Triton Model Export Script
Converts PyTorch vegetation detection model to Triton-compatible format
"""

import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np


def create_vegetation_model():
    """Create U-Net++ model with 7-channel adapter"""
    model = smp.UnetPlusPlus(
        encoder_name='senet154',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
        decoder_channels=(256, 128, 64, 32, 16),
        activation='sigmoid'
    )
    
    # Add channel adapter for 7 channels
    adapter = nn.Conv2d(7, 3, kernel_size=1, padding=0)
    
    class VegetationModel(nn.Module):
        def __init__(self, model, adapter):
            super().__init__()
            self.adapter = adapter
            self.model = model
        
        def forward(self, x):
            x = self.adapter(x)
            return self.model(x)
    
    return VegetationModel(model, adapter)


def export_to_triton(model_path, output_dir, model_name="vegetation_detector"):
    """
    Export PyTorch model to Triton format
    
    Args:
        model_path: Path to trained PyTorch model
        output_dir: Output directory for Triton model repository
        model_name: Name of model in Triton
    """
    print(f"🔄 Exporting model from {model_path}...")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vegetation_model().to(device)
    
    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✅ Loaded model weights")
    else:
        print(f"⚠️  Model path not found, using random weights for demonstration")
    
    # Create version directory
    version_dir = os.path.join(output_dir, model_name, "1")
    os.makedirs(version_dir, exist_ok=True)
    
    # Export to TorchScript
    model.eval()
    
    # Create dummy input (batch_size=1, 7 channels, 512x512)
    dummy_input = torch.randn(1, 7, 512, 512).to(device)
    
    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save model
    model_file = os.path.join(version_dir, "model.pt")
    traced_model.save(model_file)
    print(f"✅ Model exported to {model_file}")
    
    # Create config
    config_file = os.path.join(output_dir, model_name, "config.pbtxt")
    config_content = '''name: "vegetation_detector"
platform: "pytorch_libtorch"
max_batch_size: 8
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 100
}

input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [7, 512, 512]
  }
]

output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1, 512, 512]
  }
]

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]

optimization {
  execution_accelerators {
    gpu_execution_accelerator {
      using_managed_memory: true
    }
  }
}
'''
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Config saved to {config_file}")
    print(f"\n🚀 Model ready for Triton deployment at: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export model to Triton format")
    parser.add_argument('--model_path', help="Path to PyTorch model")
    parser.add_argument('--output_dir', default='./triton_models', help="Output directory")
    parser.add_argument('--model_name', default='vegetation_detector', help="Model name")
    
    args = parser.parse_args()
    
    export_to_triton(args.model_path, args.output_dir, args.model_name)
