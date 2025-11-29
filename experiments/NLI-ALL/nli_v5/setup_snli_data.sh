#!/bin/bash
# Setup SNLI Data for Geometry-First Training

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

echo "═══════════════════════════════════════════════════════════════════"
echo "  SNLI DATA SETUP"
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "This will download and set up SNLI 1.0 dataset for geometry-first training."
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Check if files already exist
if [ -f "snli_1.0_train.jsonl" ] && [ -f "snli_1.0_dev.jsonl" ] && [ -f "snli_1.0_test.jsonl" ]; then
    echo "✓ SNLI data already exists in $DATA_DIR"
    echo ""
    echo "Files found:"
    ls -lh snli_1.0_*.jsonl
    echo ""
    echo "Ready to use!"
    exit 0
fi

echo "SNLI data not found. Download options:"
echo ""
echo "Option 1: Automatic download (requires wget/curl)"
echo "Option 2: Manual download"
echo ""
read -p "Choose option (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "Downloading SNLI 1.0..."
    
    # Download SNLI 1.0
    if command -v wget &> /dev/null; then
        wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    elif command -v curl &> /dev/null; then
        curl -L -o snli_1.0.zip https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    else
        echo "❌ Neither wget nor curl found. Please download manually."
        exit 1
    fi
    
    echo "Extracting..."
    unzip -q snli_1.0.zip
    
    # Move JSONL files to data directory
    if [ -d "snli_1.0" ]; then
        mv snli_1.0/*.jsonl .
        rm -rf snli_1.0
        rm snli_1.0.zip
    fi
    
    echo "✓ SNLI data downloaded and extracted!"
    
elif [ "$choice" = "2" ]; then
    echo ""
    echo "Manual download instructions:"
    echo ""
    echo "1. Download SNLI 1.0 from: https://nlp.stanford.edu/projects/snli/"
    echo "2. Extract snli_1.0.zip"
    echo "3. Copy these files to: $DATA_DIR"
    echo "   - snli_1.0_train.jsonl"
    echo "   - snli_1.0_dev.jsonl"
    echo "   - snli_1.0_test.jsonl"
    echo ""
    echo "Then run this script again to verify."
    exit 0
else
    echo "Invalid choice. Exiting."
    exit 1
fi

# Verify files
if [ -f "snli_1.0_train.jsonl" ] && [ -f "snli_1.0_dev.jsonl" ] && [ -f "snli_1.0_test.jsonl" ]; then
    echo ""
    echo "✓ Setup complete!"
    echo ""
    echo "Files:"
    ls -lh snli_1.0_*.jsonl
    echo ""
    echo "Data directory: $DATA_DIR"
    echo ""
    echo "Ready to run geometry-first training!"
    echo ""
    echo "Run:"
    echo "  python3 experiments/nli_v5/training/train_geometry_first.py \\"
    echo "    --data-dir experiments/nli_v5/data \\"
    echo "    --train 1000 \\"
    echo "    --analyze-alignment"
else
    echo "❌ Setup incomplete. Some files missing."
    exit 1
fi

