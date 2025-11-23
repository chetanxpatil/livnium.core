# Livnium NLI v5: Clean & Simplified Architecture

**A streamlined, maintainable Natural Language Inference system that combines the best ideas from previous versions.**

## 🎯 Key Improvements

- ✅ **Simplified 5-layer architecture** (streamlined from v4's 7 layers)
- ✅ **Fixed decision layer** that properly predicts all 3 classes (E/C/N)
- ✅ **Clean separation of concerns** - each layer has a single responsibility
- ✅ **Better code organization** - easy to understand and modify
- ✅ **Comprehensive documentation** - clear explanations throughout

## Architecture

### 5-Layer Stack

```
Text Input
    ↓
Layer 0: Resonance (raw geometric signal)
    ↓
Layer 1: Curvature (cold density and distance)
    ↓
Layer 2: Basins (attraction wells for E and C)
    ↓
Layer 3: Valley (natural neutral from balance)
    ↓
Layer 4: Decision (final classification)
    ↓
Output: Entailment / Contradiction / Neutral
```

### Layer Details

**Layer 0: Resonance**
- Pure geometric signal from chain structure
- No logic, no thresholds - just raw similarity
- Position matters (sequential matching)

**Layer 1: Curvature**
- Cold density: positive resonance → dense, stable (E)
- Distance: negative resonance → far, edge (C)
- Curvature: second derivative approximation

**Layer 2: Basins**
- Cold basin (E): Dense, stable, pulls inward
- Far basin (C): Distance-based, edge of continent
- Shared depths across instances (learns from all examples)

**Layer 3: Valley**
- The City (N): Natural neutral from force balance
- Forms where cold and far attractions overlap
- Real gravitational mass, not just absence

**Layer 4: Decision**
- Properly handles all 3 classes
- Rules:
  1. Weak forces → Neutral
  2. Balanced forces → Neutral
  3. City dominates → Neutral
  4. One side wins → E or C

## Usage

### Training

```bash
# Small test (fast)
python3 experiments/nli_v5/train_v5.py --clean --train 1000 --test 200 --dev 200

# Medium run (recommended)
python3 experiments/nli_v5/train_v5.py --clean --train 5000 --test 500 --dev 500

# Full training (best results)
python3 experiments/nli_v5/train_v5.py --clean --train 20000 --test 2000 --dev 2000
```

### Python API

```python
from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier

# Encode
encoder = ChainEncoder()
pair = encoder.encode_pair('A cat runs', 'A cat is running')

# Classify
classifier = LivniumV5Classifier(pair)
result = classifier.classify()

print(f"Label: {result.label}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Scores: {result.scores}")

# Learn
classifier.apply_learning_feedback('entailment', learning_strength=1.0)
```

## Key Principles

1. **Gravity shapes everything** - No manual tuning needed
2. **Clean separation** - Each layer builds on the one below
3. **Proper 3-class prediction** - Fixed from v4's contradiction issue
4. **Self-correcting** - Basins learn from all examples
5. **Maintainable** - Clear code structure, easy to modify

## Comparison with Previous Versions

| Feature | v5 | v4 | v3 | v2 |
|---------|----|----|----|----|
| Layers | 5 | 7 | 3 | 2 |
| Contradiction Fix | ✅ | ❌ | ✅ | ✅ |
| Code Complexity | Low | High | Medium | Low |
| Maintainability | High | Medium | Medium | High |
| Auto-evolving Rules | ❌ | ✅ | ❌ | ❌ |

## Expected Performance

- **After 1000 examples**: ~45-50% accuracy
- **After 5000 examples**: ~50-55% accuracy
- **After 20000 examples**: ~55-60%+ accuracy

## Files

- `encoder.py` - Chain encoder with positional encoding
- `layers.py` - All 5 layers (resonance, curvature, basins, valley, decision)
- `classifier.py` - Main classifier class
- `train_v5.py` - Training script
- `README.md` - This file

## Philosophy

**Clean. Simple. Effective.**

v5 takes the best ideas from previous versions and streamlines them into a maintainable, easy-to-understand system. No unnecessary complexity, just clean geometric reasoning.

