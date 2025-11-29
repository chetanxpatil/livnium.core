# How to Run NLI v5 Commands

After the reorganization, here are the updated commands.

**Important:** Always run from the project root (`/Users/chetanpatil/Desktop/clean-nova-livnium`)

## Training

### Training with Fracture Detection (Recommended)
**Collision-based negation detection is automatically enabled.**

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/train_with_fracture.py \
  --train 1000 \
  --test 100 \
  --show-fractures
```

This trains with collision-based fracture detection integrated.
Negation is automatically detected by colliding premise and hypothesis vectors.

### Geometry-First Training
**Let geometry be the teacher, not the student.**

```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/train_geometry_first.py \
  --train 1000 \
  --analyze-alignment \
  --save-alignment experiments/nli_v5/patterns/geometry_alignment.json
```

**Note:** If SNLI data is not found, the script will:
1. Try to load from existing patterns
2. Fall back to synthetic examples for demonstration

**To use real SNLI data:**
1. Download from: https://nlp.stanford.edu/projects/snli/
2. Extract `snli_1.0_train.jsonl` to: `experiments/nli_simple/data/snli_1.0_train.jsonl`

This trains the classifier to read geometry zones instead of forcing geometry to match labels.

### Normal Training (with pattern learning)
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/train_v5.py \
  --clean \
  --train 1000 \
  --learn-patterns \
  --pattern-file experiments/nli_v5/patterns/patterns.json
```

### Inverted Labels Training (for physics discovery)
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/train_v5.py \
  --clean \
  --train 1000 \
  --invert-labels \
  --learn-patterns \
  --pattern-file experiments/nli_v5/patterns/patterns_inverted.json
```

### Training with Custom Pattern File
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/train_v5.py \
  --clean \
  --train 1000 \
  --learn-patterns \
  --pattern-file experiments/nli_v5/patterns/my_patterns.json
```

## Testing

### Test All Laws
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/tests/test_all_laws.py
```

### Test Laws Per Example
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/tests/test_laws_per_example.py
```

### Test Physics Analysis
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/tests/test_physics_analysis.py
```

### Quick Test
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/tests/test_v5.py
```

## Calibration

### Calibrate Divergence Threshold
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/calibrate_divergence.py \
  --pattern-file experiments/nli_v5/patterns/patterns_normal.json \
  --method neutral
```

## Pattern Comparison

### Compare Patterns
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/compare_patterns.py \
  --debug-file experiments/nli_v5/patterns/patterns_debug.json \
  --normal-file experiments/nli_v5/patterns/patterns_normal.json
```

### Compare Inverted Patterns
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/tests/compare_inverted_patterns.py \
  --normal-file experiments/nli_v5/patterns/patterns_normal.json \
  --inverted-file experiments/nli_v5/patterns/patterns_inverted.json
```

## Visualization

### Generate 3D Force Field Data
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/planet/compute_3d_force_field.py \
  --resolution 20 \
  --output livnium_3d_force_field.json
```

### Launch Geometry Explorer
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium/experiments/nli_v5/viewer
./launch_geometry_explorer.sh
```

## File Locations

### Pattern Files
- `experiments/nli_v5/patterns/patterns.json` - Main patterns
- `experiments/nli_v5/patterns/patterns_normal.json` - Normal mode patterns
- `experiments/nli_v5/patterns/patterns_inverted.json` - Inverted label patterns
- `experiments/nli_v5/patterns/patterns_debug.json` - Debug mode patterns

### Output Files
- `experiments/nli_v5/brain_state/brain_state.pkl` - Saved model
- `experiments/nli_v5/planet_output/livnium_3d_force_field.json` - 3D geometry data

## Quick Reference

### Setup SNLI Data First
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium/experiments/nli_v5
./setup_snli_data.sh
```

### Geometry-First Training (Recommended)
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 experiments/nli_v5/training/train_geometry_first.py \
  --data-dir experiments/nli_v5/data \
  --train 1000 \
  --analyze-alignment \
  --save-alignment experiments/nli_v5/patterns/geometry_alignment.json
```

### Easiest Way (Using Script)
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
./experiments/nli_v5/run_training.sh
```

### Manual Commands

**Most common workflow:**
```bash
# 1. Train with pattern learning
python3 experiments/nli_v5/training/train_v5.py --clean --train 1000 --learn-patterns

# 2. Test laws
python3 experiments/nli_v5/tests/test_all_laws.py

# 3. View geometry
cd experiments/nli_v5/viewer && ./launch_geometry_explorer.sh
```

### If You Get Stuck in Quote Mode

If you see `quote>` prompt:
1. Press `Ctrl+C` to cancel
2. Then run the command again (or use the script above)

