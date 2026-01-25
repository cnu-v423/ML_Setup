#!/bin/bash
# Quick Start Script for Multi-Class Road Segmentation Training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Multi-Class Road Segmentation - Quick Start                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

# Check if directories are provided
if [ $# -lt 2 ]; then
    echo -e "${RED}Usage:${NC} ./quickstart.sh <tiles_dir> <masks_dir> [model_output_dir] [weights_path]"
    echo ""
    echo "Example:"
    echo "  ./quickstart.sh ../data/tiles ../data/masks ./models/roads_v1"
    echo ""
    echo "With pretrained weights:"
    echo "  ./quickstart.sh ../data/tiles ../data/masks ./models/roads_v1 ./pretrained.pt"
    exit 1
fi

TILES_DIR=$1
MASKS_DIR=$2
MODEL_OUTPUT=${3:-./trained_models/multiclass_roads}
WEIGHTS_PATH=${4:-}

echo -e "\n${YELLOW}📋 Configuration:${NC}"
echo "  Tiles directory:  $TILES_DIR"
echo "  Masks directory:  $MASKS_DIR"
echo "  Output directory: $MODEL_OUTPUT"
if [ -n "$WEIGHTS_PATH" ]; then
    echo "  Pretrained weights: $WEIGHTS_PATH"
fi

# Step 1: Validate data
echo -e "\n${BLUE}Step 1/3: Validating Data${NC}"
echo "=================================================="

if python validate_multiclass_data.py \
    --tiles_dir "$TILES_DIR" \
    --masks_dir "$MASKS_DIR"; then
    echo -e "${GREEN}✅ Data validation passed!${NC}"
else
    echo -e "${RED}❌ Data validation failed!${NC}"
    echo "Please fix the issues and try again."
    exit 1
fi

# Step 2: Create output directory
echo -e "\n${BLUE}Step 2/3: Preparing Output Directory${NC}"
echo "=================================================="
mkdir -p "$MODEL_OUTPUT"
echo -e "${GREEN}✅ Output directory: $MODEL_OUTPUT${NC}"

# Step 3: Start training
echo -e "\n${BLUE}Step 3/3: Starting Training${NC}"
echo "=================================================="
echo -e "${YELLOW}This will take 1-3 hours depending on GPU${NC}"
echo ""

if [ -n "$WEIGHTS_PATH" ]; then
    python enhanced_pytorch_backbone_training_multiclass.py \
        --input_tiles_dir "$TILES_DIR" \
        --input_masks_dir "$MASKS_DIR" \
        --model_path "$MODEL_OUTPUT" \
        --weights_path "$WEIGHTS_PATH"
else
    python enhanced_pytorch_backbone_training_multiclass.py \
        --input_tiles_dir "$TILES_DIR" \
        --input_masks_dir "$MASKS_DIR" \
        --model_path "$MODEL_OUTPUT"
fi

# Summary
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Training Complete!                                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}📦 Output Files:${NC}"
echo "  • Best model: $MODEL_OUTPUT/best_model_stage2.pt"
echo "  • Final model: $MODEL_OUTPUT/multiclass_road_segmentation_final.pt"
echo "  • Logs: $MODEL_OUTPUT/logs/"

echo -e "\n${GREEN}🎯 Next Steps:${NC}"
echo "  1. Evaluate model on test set"
echo "  2. Use for predictions:"
echo "     python multiclass_inference.py \\"
echo "       --model_path $MODEL_OUTPUT/best_model_stage2.pt \\"
echo "       --image_dir ./new_tiles \\"
echo "       --output_dir ./predictions"

echo -e "\n${GREEN}✅ Training script completed successfully!${NC}\n"
