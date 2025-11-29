#!/bin/bash
# Quick Training Script for NLI with Learned Polarity

echo "=========================================="
echo "NLI Training with Learned Polarity"
echo "=========================================="
echo ""

# Check if tqdm is installed
python3 -c "import tqdm" 2>/dev/null || {
    echo "Installing tqdm..."
    pip install tqdm -q
}

echo "Starting training..."
echo ""

# Small test run (fast, good for testing)
python3 experiments/nli/train_moksha_nli.py \
    --clean \
    --train 1000 \
    --test 200 \
    --dev 200

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "To check learned word polarities, run:"
echo "  python3 -c \"from experiments.nli.native_chain import GlobalLexicon; lexicon = GlobalLexicon(); print('not:', lexicon.get_word_polarity('not'))\""
echo ""
echo "To test a single example, run:"
echo "  python3 experiments/nli/test_golden_label_collapse.py --premise 'A dog runs' --hypothesis 'A dog is running'"
echo ""

