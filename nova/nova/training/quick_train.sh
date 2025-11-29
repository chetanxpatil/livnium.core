#!/bin/bash
# Quick Training Script for Livnium Reply Engine

echo "============================================================"
echo "Livnium Reply Engine - Quick Training"
echo "============================================================"
echo ""

# Check if dataset exists
if [ ! -f "nova/data/daily_test.csv" ]; then
    echo "❌ Error: Dataset not found at nova/data/daily_test.csv"
    exit 1
fi

# Default values
MAX_DIALOGUES=${1:-100}
COLLAPSE_STEPS=${2:-15}
OUTPUT_DIR=${3:-"nova/model"}

echo "Training parameters:"
echo "  Max dialogues: $MAX_DIALOGUES"
echo "  Collapse steps: $COLLAPSE_STEPS"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo "Starting training..."
python3 nova/training/train_text_to_geometry.py \
    --csv nova/data/daily_test.csv \
    --max-dialogues "$MAX_DIALOGUES" \
    --collapse-steps "$COLLAPSE_STEPS" \
    --output-dir "$OUTPUT_DIR"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Training Complete!"
    echo "============================================================"
    echo ""
    echo "Output files:"
    echo "  - $OUTPUT_DIR/signature_database.json"
    echo "  - $OUTPUT_DIR/learned_patterns.json"
    echo ""
    echo "Test the reply generator:"
    echo "  python3 nova/chat/reply_generator.py --query 'Hello'"
    echo ""
    echo "Or start interactive chat:"
    echo "  python3 nova/chat/chat_demo.py"
else
    echo ""
    echo "❌ Training failed. Check the error messages above."
    exit 1
fi

