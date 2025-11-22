# Natural Language Inference (NLI)

Pure geometric reasoning for Natural Language Inference tasks.

## Contents

- **`native_chain.py`**: Letter-by-letter MPS architecture
- **`native_chain_encoder.py`**: Encodes text into geometric chains
- **`omcube.py`**: Quantum collapse engine for 3-way classification
- **`inference_detectors.py`**: Native logic (lexical overlap, negation, etc.)
- **`nli_memory.py`**: Memory system for NLI
- **`train_moksha_nli.py`**: Training pipeline
- **`test_golden_label_collapse.py`**: Test suite
- **`test_omcube_capacity.py`**: Capacity tests

## Purpose

This experiment implements NLI (Entailment/Contradiction/Neutral classification) using:
- **Zero neural networks**: Pure geometric reasoning
- **Letter-by-letter encoding**: Words as chains of letter omcubes
- **Quantum collapse**: 3-way decision making
- **Basin reinforcement**: Physics-based learning

## Usage

```bash
# Train the system
python3 train_moksha_nli.py --clean --train 20000 --test 2000 --dev 2000

# Test a single example
python3 test_golden_label_collapse.py --premise "A dog runs" --hypothesis "A dog is running"
```

