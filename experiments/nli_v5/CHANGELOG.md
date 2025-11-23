# Changelog: nli_v5

## What's New in v5

### 🎯 Key Improvements

1. **Simplified Architecture**
   - Reduced from 7 layers (v4) to 5 layers
   - Removed: Meta Routing, Temporal Stability, Semantic Memory layers
   - Kept: Core geometric layers (Resonance, Curvature, Basins, Valley, Decision)
   - Result: Cleaner, easier to understand and maintain

2. **Fixed Decision Layer**
   - v4 had a bug where contradiction was rarely predicted
   - v5 properly handles all 3 classes (E/C/N)
   - Clear decision rules:
     - Weak forces → Neutral
     - Balanced forces → Neutral
     - City dominates → Neutral
     - One side wins → E or C

3. **Better Code Organization**
   - Single file per component (`encoder.py`, `layers.py`, `classifier.py`)
   - Clear separation of concerns
   - Comprehensive docstrings
   - Easy to modify and extend

4. **Cleaner Training Script**
   - Streamlined from v4's complex training loop
   - Clear progress reporting
   - JSON + human-readable confusion matrices

### 📁 File Structure

```
nli_v5/
├── __init__.py          # Package exports
├── encoder.py           # Chain encoder (from v3/v4)
├── layers.py            # All 5 layers in one file
├── classifier.py        # Main classifier class
├── train_v5.py          # Training script
├── test_v5.py           # Quick test script
├── README.md            # Documentation
└── CHANGELOG.md         # This file
```

### 🔄 Migration from v4

**What Changed:**
- Removed auto-evolving rules (can be added back if needed)
- Removed auto-physics engine (simplified)
- Removed autonomous meaning engine (simplified)
- Removed feature logging (can be added back if needed)

**What Stayed:**
- Chain encoder (proven to work)
- Core geometric layers (resonance, curvature, basins)
- Basin reinforcement learning
- Word polarity learning

**What Improved:**
- Decision layer now properly predicts all 3 classes
- Cleaner code structure
- Easier to understand and modify

### 🚀 Usage

```python
from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier

encoder = ChainEncoder()
pair = encoder.encode_pair('A cat runs', 'A cat is running')
classifier = LivniumV5Classifier(pair)
result = classifier.classify()

print(result.label)  # 'entailment', 'contradiction', or 'neutral'
```

### 📊 Expected Performance

- **After 1000 examples**: ~45-50% accuracy
- **After 5000 examples**: ~50-55% accuracy
- **After 20000 examples**: ~55-60%+ accuracy

### 🐛 Known Issues

- Initial predictions may favor entailment (needs training)
- Decision thresholds may need tuning for specific datasets
- Word polarity learning is simple (could be enhanced)

### 🔮 Future Enhancements

- Add back auto-evolving rules (optional)
- Add feature logging for analysis
- Add more sophisticated word polarity learning
- Add ensemble methods
- Add hyperparameter tuning

