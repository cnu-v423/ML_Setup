# PyTorch to ONNX Conversion Guide

## Overview
This guide explains how to convert your trained PyTorch multi-class segmentation model to ONNX format for deployment.

## Features
- ✅ Multi-class segmentation support (4 classes: Background, Thar, CC, Mud/Gravel)
- ✅ Automatic model discovery if path not specified
- ✅ Model validation against PyTorch output
- ✅ Configurable ONNX opset version
- ✅ Clear progress and error reporting

## Prerequisites
```bash
# Core requirements
pip install torch torchvision
pip install onnx
pip install onnxruntime

# Optional (for model validation)
pip install numpy
```

## Usage

### 1. Basic Conversion (Automatic Model Discovery)
```bash
cd pytorch_model_training
python pytorch_model_to_onnx.py --config_path ../config/config_v1.yaml
```

This will automatically find the trained model in the current directory.

### 2. Conversion with Specific Model Path
```bash
python pytorch_model_to_onnx.py \
    --model_path multiclass_road_segmentation_final.pt \
    --output_path ./models/multiclass_model.onnx
```

### 3. Advanced Usage with Custom Settings
```bash
python pytorch_model_to_onnx.py \
    --model_path multiclass_road_segmentation_final.pt \
    --output_path ./onnx_models/model_v1.onnx \
    --config_path ../config/config_v1.yaml \
    --opset_version 14
```

### 4. Conversion Without Validation (Faster)
```bash
python pytorch_model_to_onnx.py \
    --model_path multiclass_road_segmentation_final.pt \
    --no-validate
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Auto-detect | Path to trained PyTorch model (.pt file) |
| `--output_path` | Auto | Path to save ONNX model |
| `--config_path` | `../config/config_v1.yaml` | Path to configuration file |
| `--opset_version` | `13` | ONNX opset version |
| `--no-validate` | False | Skip ONNX validation |

## Supported Model Files
The script automatically searches for models in this priority order:
1. `multiclass_road_segmentation_final.pt`
2. `best_model_stage2.pt`
3. `best_model_stage1.pt`
4. `final_model.pt`
5. Any `.pt` file in subdirectories matching 'multiclass' or 'final'

## Expected Output

```
================================================================================
🚀 PyTorch to ONNX Conversion Tool (Multi-Class Segmentation)
================================================================================
✓ Configuration loaded
  • Input Size: 256x256
  • Num Classes: 4
  • Device: cuda

🔍 Searching for trained models...
  ✓ Found: multiclass_road_segmentation_final.pt

📦 Loading PyTorch model from: multiclass_road_segmentation_final.pt
  • Loading weights...
✓ Model loaded successfully
  • Total Parameters: 24,743,682
  • Trainable Parameters: 24,743,682
  • Model Size: 94.43 MB

🔄 Converting model to ONNX...
  • Input Size: (1, 3, 256, 256)
  • Output Size: (1, 4, 256, 256)
  • Opset Version: 13
✓ Model exported to ONNX
  • Output path: multiclass_road_segmentation_final.onnx

📊 ONNX Model Information:
  • Input: input - [1, 3, 256, 256]
  • Output: output - [1, 4, 256, 256]

🔍 Validating ONNX model...
✓ ONNX model validated
  • Output Shape: (1, 4, 256, 256)
  • Max Difference: 0.000123
  • Mean Difference: 0.000045
✓ Outputs match within acceptable tolerance

================================================================================
✅ CONVERSION COMPLETED SUCCESSFULLY!
================================================================================
📦 ONNX Model: multiclass_road_segmentation_final.onnx
💾 File size: 94.32 MB
📋 Config:
   • Input Size: 256x256
   • Classes: 4
   • Device: cuda
================================================================================
```

## Programmatic Usage

### Using as a Python Module
```python
from pytorch_model_to_onnx import convert_model_to_onnx
import yaml

# Load config
with open('../config/config_v1.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['num_classes'] = 4

# Convert model
onnx_path = convert_model_to_onnx(
    config,
    model_path='multiclass_road_segmentation_final.pt',
    output_path='model.onnx',
    validate=True,
    opset_version=13
)

print(f"ONNX model saved to: {onnx_path}")
```

### Using the Converter Class
```python
from pytorch_model_to_onnx import ONNXModelConverter
import yaml
import torch

# Load config
with open('../config/config_v1.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['data']['num_classes'] = 4

# Create converter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
converter = ONNXModelConverter(config, 'multiclass_road_segmentation_final.pt', device)

# Load model
pytorch_model = converter.load_model()

# Convert to ONNX
onnx_path = converter.convert_to_onnx(pytorch_model, 'model.onnx')

# Validate
converter.validate_onnx(onnx_path, pytorch_model)
```

## Inference with ONNX Model

### Using ONNX Runtime
```python
import onnxruntime as ort
import numpy as np

# Create session
sess = ort.InferenceSession('multiclass_road_segmentation_final.onnx')

# Prepare input
input_data = np.random.randn(1, 3, 256, 256).astype(np.float32)

# Run inference
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: input_data})

# Get predictions
logits = output[0]  # Shape: (1, 4, 256, 256)
predictions = np.argmax(logits, axis=1)  # Get class predictions
```

## Troubleshooting

### Model Not Found
**Error:** `FileNotFoundError: No trained model found`

**Solution:** Specify model path explicitly:
```bash
python pytorch_model_to_onnx.py --model_path /path/to/model.pt
```

### ONNX Validation Failed
**Error:** `Validation failed` or large output differences

**Solution:** 
1. Try with `--no-validate` first to check if export works
2. Check if model architecture matches between PyTorch and ONNX
3. Verify input/output shapes are correct

### ONNX Runtime Not Available
**Error:** `ImportError: No module named 'onnxruntime'`

**Solution:** Install ONNX Runtime:
```bash
pip install onnxruntime
# For GPU support:
pip install onnxruntime-gpu
```

### Out of Memory During Conversion
**Solution:** Reduce input size or use CPU:
```bash
python pytorch_model_to_onnx.py --model_path model.pt --device cpu
```

## Model Configuration

The conversion uses these settings from config:
```yaml
data:
  input_size: 256  # Model input size
  num_classes: 4   # Number of output classes
```

## Performance Considerations

1. **Input Size**: Larger input sizes result in larger ONNX models
2. **Opset Version**: Higher versions may have better performance but less compatibility
3. **Batch Size**: ONNX models support dynamic batch sizes (set to 1 during export)

## Opset Versions

Common opset versions:
- **11**: Basic operations, good compatibility
- **12**: Better shape inference, improved performance
- **13**: Better flow control, recommended (default)
- **14**: New operators, better optimization
- **17+**: Latest features, requires newer ONNX Runtime

## Next Steps

After conversion:
1. Test the ONNX model with inference code
2. Deploy to production using ONNX Runtime
3. Consider using Triton Inference Server for serving
4. Benchmark performance against PyTorch

## Additional Resources

- [ONNX Documentation](https://onnx.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
