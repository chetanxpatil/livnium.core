#!/bin/bash
# Quick training script for NLI v5

cd "$(dirname "$0")/../.."

echo "Training with inverted labels..."
python3 experiments/nli_v5/training/train_v5.py \
  --clean \
  --train 1000 \
  --invert-labels \
  --learn-patterns \
  --pattern-file experiments/nli_v5/patterns/patterns_inverted.json

echo ""
echo "Testing all laws..."
python3 experiments/nli_v5/tests/test_all_laws.py

