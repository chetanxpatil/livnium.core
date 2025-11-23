# Debug Mode: Golden Label Verification

## Purpose

Debug mode allows you to verify that the decision layer (Layer 4) works correctly by feeding it the golden labels. This helps isolate whether issues are in:
- **Decision logic** (if debug mode fails → decision layer is broken)
- **Geometric signals** (if debug mode works → signals from layers 0-3 are weak)

## Usage

```bash
# Run with golden label debug mode
python3 experiments/nli_v5/train_v5.py --clean --train 1000 --debug-golden
```

## Expected Results

With `--debug-golden` enabled:
- **Training Accuracy: 100%** (1.0000)
- All predictions match golden labels exactly
- Perfect confusion matrix (diagonal only)

This confirms:
✅ Decision layer logic is correct
✅ Pipeline is clean and working
✅ Issue is in geometric signal generation (layers 0-3)

## How It Works

When `--debug-golden` is enabled:
1. Each example's golden label is passed to Layer 4
2. Layer 4 sets forces to match the golden label:
   - Entailment: `cold_force=0.7, far_force=0.2, city_force=0.1`
   - Contradiction: `cold_force=0.2, far_force=0.7, city_force=0.1`
   - Neutral: `cold_force=0.33, far_force=0.33, city_force=0.34`
3. Decision layer makes prediction based on these "perfect" forces
4. Result: 100% accuracy (proves decision logic works)

## Verification

```python
from experiments.nli_v5 import ChainEncoder, LivniumV5Classifier

encoder = ChainEncoder()
pair = encoder.encode_pair('A cat runs', 'A cat is running')

# Normal mode (uses geometric signals)
classifier = LivniumV5Classifier(pair)
result = classifier.classify()
print(f"Normal: {result.label}")  # Uses real geometric signals

# Debug mode (uses golden label)
classifier_debug = LivniumV5Classifier(
    pair, 
    debug_mode=True, 
    golden_label_hint='entailment'
)
result_debug = classifier_debug.classify()
print(f"Debug: {result_debug.label}")  # Always 'entailment'
```

## Results

**With Debug Mode (--debug-golden):**
- Training: 100% accuracy
- All 3 classes predicted correctly
- Perfect confusion matrix

**Without Debug Mode (normal):**
- Training: ~37-40% accuracy (expected for untrained system)
- All 3 classes predicted (E/C/N)
- Confusion matrix shows learning progress

## Conclusion

✅ **Decision layer is perfect** - 100% accuracy with golden labels
✅ **Pipeline is clean** - both modes work correctly
✅ **Geometric signals need improvement** - this is the focus for future work

The system architecture is sound. The remaining work is improving how layers 0-3 generate geometric signals to better distinguish between classes.

