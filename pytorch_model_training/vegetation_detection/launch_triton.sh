#!/bin/bash
# Launch Triton Inference Server with vegetation detection model
# Usage: ./launch_triton.sh [--gpu 0] [--port 8000] [--model_repo ./triton_models]

GPU_ID=0
PORT=8000
MODEL_REPO="./triton_models"
DOCKER_IMAGE="nvcr.io/nvidia/tritonserver:23.12-py3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --model_repo) MODEL_REPO="$2"; shift 2 ;;
        --docker_image) DOCKER_IMAGE="$2"; shift 2 ;;
        --help) 
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --gpu GPU_ID           GPU device ID (default: 0)"
            echo "  --port PORT            Port for Triton server (default: 8000)"
            echo "  --model_repo PATH      Path to model repository (default: ./triton_models)"
            echo "  --docker_image IMAGE   Docker image (default: nvcr.io/nvidia/tritonserver:23.12-py3)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "════════════════════════════════════════════════════════════════"
echo "🚀 TRITON INFERENCE SERVER LAUNCHER"
echo "════════════════════════════════════════════════════════════════"
echo "📊 GPU Device: $GPU_ID"
echo "🔌 Port: $PORT"
echo "📁 Model Repository: $MODEL_REPO"
echo "📦 Docker Image: $DOCKER_IMAGE"
echo "════════════════════════════════════════════════════════════════"

# Check if model repository exists
if [ ! -d "$MODEL_REPO" ]; then
    echo "❌ Model repository not found: $MODEL_REPO"
    echo "   Create it with: python export_to_triton.py --model_path <path> --output_dir $MODEL_REPO"
    exit 1
fi

# Check if vegetation_detector model exists
if [ ! -d "$MODEL_REPO/vegetation_detector" ]; then
    echo "❌ vegetation_detector model not found in $MODEL_REPO"
    exit 1
fi

echo "✅ Model repository verified"
echo ""

# Check if Docker is available
if command -v docker &> /dev/null; then
    echo "🐳 Docker found. Launching Triton server in container..."
    echo ""
    
    # Get absolute path
    ABS_MODEL_REPO=$(cd "$MODEL_REPO" && pwd)
    
    # Launch Docker container
    docker run --gpus '"'device=$GPU_ID'"' \
        -p $PORT:8000 \
        -p 8001:8001 \
        -p 8002:8002 \
        -v "$ABS_MODEL_REPO:/models" \
        $DOCKER_IMAGE \
        tritonserver --model-repository=/models --log-verbose=1
else
    echo "⚠️  Docker not found. Trying direct Triton server launch..."
    echo "   Ensure tritoniserver is installed: pip install nvidia-pytriton"
    echo ""
    
    # Try direct launch
    tritonserver --model-repository="$MODEL_REPO" \
                 --log-verbose=1 \
                 --grpc-port 8001 \
                 --http-port $PORT \
                 --metrics-port 8002
fi
