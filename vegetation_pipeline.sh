#!/bin/bash
# VEGETATION DETECTION PIPELINE - QUICK START SCRIPT
# This script runs the complete vegetation detection pipeline

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🌳 VEGETATION DETECTION PIPELINE 🌳${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration
INPUT_TIFF="${1:-}"
VEGETATION_SHP="${2:-}"
OUTPUT_DIR="${3:-./vegetation_output}"
MODEL_WEIGHTS="${4:-}"

# Validate inputs
if [ -z "$INPUT_TIFF" ] || [ -z "$VEGETATION_SHP" ]; then
    echo -e "${YELLOW}Usage: ./vegetation_pipeline.sh <input_tiff> <vegetation_shapefile> [output_dir] [model_weights]${NC}"
    echo ""
    echo "Example:"
    echo "  ./vegetation_pipeline.sh data/ortho.tif data/trees_shrubs.shp output/ weights/best_model.pt"
    exit 1
fi

if [ ! -f "$INPUT_TIFF" ]; then
    echo -e "${YELLOW}❌ Input TIFF not found: $INPUT_TIFF${NC}"
    exit 1
fi

if [ ! -f "$VEGETATION_SHP" ]; then
    echo -e "${YELLOW}❌ Vegetation shapefile not found: $VEGETATION_SHP${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Create Tiles
echo -e "\n${GREEN}Step 1: Creating vegetation tiles...${NC}"
python create_vegetation_tiles.py \
    --input_tif "$INPUT_TIFF" \
    --input_shp "$VEGETATION_SHP" \
    --output_dir "$OUTPUT_DIR/tiles_masks" \
    --tile_size 512 \
    --overlap 0.25

TILES_DIR="$OUTPUT_DIR/tiles_masks/tiles_veg"
MASKS_DIR="$OUTPUT_DIR/tiles_masks/masks_veg"

# Check if tiles were created
if [ ! -d "$TILES_DIR" ] || [ -z "$(ls -A $TILES_DIR)" ]; then
    echo -e "${YELLOW}❌ No tiles were created. Check input data.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Tiles created successfully${NC}"

# Step 2: Train Model (if model weights not provided)
if [ -z "$MODEL_WEIGHTS" ]; then
    echo -e "\n${GREEN}Step 2: Training vegetation detection model...${NC}"
    
    MODEL_OUTPUT="$OUTPUT_DIR/model"
    mkdir -p "$MODEL_OUTPUT"
    
    cd pytorch_model_training
    python vegetation_detection_training.py \
        --input_tiles_dir "$TILES_DIR" \
        --input_masks_dir "$MASKS_DIR" \
        --model_path "$MODEL_OUTPUT"
    cd ..
    
    # Find the best model
    MODEL_WEIGHTS=$(find "$MODEL_OUTPUT" -name "*best*.pt" -type f | head -1)
    
    if [ -z "$MODEL_WEIGHTS" ]; then
        echo -e "${YELLOW}❌ Model training failed or no best model found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Model trained successfully${NC}"
else
    if [ ! -f "$MODEL_WEIGHTS" ]; then
        echo -e "${YELLOW}❌ Provided model weights not found: $MODEL_WEIGHTS${NC}"
        exit 1
    fi
    echo -e "${GREEN}Using provided model: $MODEL_WEIGHTS${NC}"
fi

# Step 3: Run Inference
echo -e "\n${GREEN}Step 3: Running vegetation detection inference...${NC}"

OUTPUT_PREDICTION="$OUTPUT_DIR/predictions/vegetation_prediction.tif"
mkdir -p "$OUTPUT_DIR/predictions"

python vegetation_inference.py \
    --input_image "$INPUT_TIFF" \
    --output_path "$OUTPUT_PREDICTION" \
    --model_path "$MODEL_WEIGHTS" \
    --tile_size 512 \
    --overlap 64 \
    --threshold 0.5 \
    --device cuda

echo -e "${GREEN}✅ Inference completed successfully${NC}"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${GREEN}Output Files:${NC}"
echo "  📊 Tiles:              $TILES_DIR"
echo "  🎭 Masks:              $MASKS_DIR"
echo "  🧠 Model Weights:      $MODEL_WEIGHTS"
echo "  🌳 Vegetation Map:     $OUTPUT_PREDICTION"
echo "  🔲 Binary Map:         ${OUTPUT_PREDICTION%.*}_binary.tif"
echo -e "\n${BLUE}Next steps:${NC}"
echo "  1. Review vegetation predictions in GIS software"
echo "  2. Fine-tune threshold if needed (currently: 0.5)"
echo "  3. Post-process if necessary (morphological operations)"
echo -e "\n"
