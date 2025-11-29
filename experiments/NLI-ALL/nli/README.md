# Natural Language Inference (NLI)

Pure geometric reasoning for Natural Language Inference tasks using Livnium's native chain architecture.

## Quick Start

```bash
# Train the system
python3 experiments/nli/train_moksha_nli.py --clean --train 1000 --test 200 --dev 200

# Test a single example
python3 experiments/nli/test_golden_label_collapse.py \
    --premise "A dog runs" \
    --hypothesis "A dog is running"
```

## Architecture

- **`native_chain.py`**: Letter-by-letter MPS architecture
- **`native_chain_encoder.py`**: Encodes text into geometric chains
- **`omcube.py`**: Quantum collapse engine for 3-way classification
- **`inference_detectors.py`**: Native logic (lexical overlap, learned polarity, etc.)
- **`train_moksha_nli.py`**: Training pipeline

## Features

- **Zero neural networks**: Pure geometric reasoning
- **Letter-by-letter encoding**: Words as chains of letter omcubes
- **Quantum collapse**: 3-way decision making (Entailment/Contradiction/Neutral)
- **Learned word polarity**: Words learn semantic polarity from training data
- **Basin reinforcement**: Physics-based learning

## Training

```bash
# Small test (fast)
python3 experiments/nli/train_moksha_nli.py --clean --train 1000 --test 200 --dev 200

# Medium run (recommended)
python3 experiments/nli/train_moksha_nli.py --clean --train 5000 --test 500 --dev 500

# Full training (best results)
python3 experiments/nli/train_moksha_nli.py --clean --train 20000 --test 2000 --dev 2000
```

## Expected Results

- **After 1000 examples**: ~45-50% accuracy
- **After 5000 examples**: ~50-55% accuracy
- **After 20000 examples**: ~55-60%+ accuracy

