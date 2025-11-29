# Pattern Learning: Understanding Geometry from Golden Labels

## The Insight

When we feed golden labels, we're essentially getting **perfect signals**. We can use this to understand what geometric patterns lead to correct classification.

**This is reverse engineering:** If we know the answer (golden label) and we know what forces produce that answer, we can learn what patterns lead to correct classification.

## How It Works

The pattern learner records all geometric signals for each golden label:

- **Resonance**: Raw similarity signal
- **Divergence**: Field divergence (positive = C, negative = E, zero = N)
- **Convergence**: Negative divergence (positive = E, negative = C)
- **Cold density**: Convergence force (E)
- **Divergence force**: Divergence force (C)
- **Attractions**: Cold and far attraction strengths
- **Forces**: Final normalized forces (cold_force, far_force, city_force)

Then it analyzes what patterns correlate with each class.

## Usage

### Learn Patterns During Training (Normal Mode)

```bash
# Learn patterns from real geometry
python3 experiments/nli_v5/train_v5.py --clean --train 1000 --learn-patterns

# Save patterns to specific file
python3 experiments/nli_v5/train_v5.py --clean --train 1000 --learn-patterns --pattern-file my_patterns.json
```

### Learn Patterns with Debug Mode

```bash
# Learn patterns with debug mode (compare real geometry vs ideal forces)
python3 experiments/nli_v5/train_v5.py --clean --train 1000 --debug-golden --learn-patterns
```

**What's different in debug mode:**
- **Geometric signals** (resonance, divergence, convergence) are **REAL** from layers 0-3
- **Forces** (cold_force, far_force, city_force) are **ARTIFICIAL** (set to match golden labels)
- This shows: **What geometry produces vs what forces are needed**

### Why Use Debug Mode for Pattern Learning?

Debug mode lets you see:
1. **What geometry actually produces** (real signals from layers 0-3)
2. **What forces would be ideal** (artificial forces from Layer 4)
3. **The gap between them** (calibration needed?)

Example:
- Real geometry produces: `cold_force=0.4, far_force=0.3` (weak signals)
- Ideal forces needed: `cold_force=0.7, far_force=0.2` (for entailment)
- **Gap**: Geometry is too weak → need to boost signals

## What You Get

After training, you'll see:

1. **Statistics per class** (E/C/N):
   - Mean, std, min, max, median, quartiles for each signal
   - Shows what ranges correspond to each class

2. **Key Insights**:
   - Resonance ranges for E/C/N
   - Divergence ranges (should be: E negative, C positive, N near zero)
   - Convergence ranges (should be: E positive, C negative, N near zero)
   - Force distributions
   - Attraction patterns

3. **Debug Mode Insights** (if using `--debug-golden`):
   - Comparison of real geometry vs ideal forces
   - Identifies calibration gaps

### Example Output (Normal Mode)

```
GEOMETRIC PATTERN ANALYSIS (from Golden Labels) - NORMAL MODE
================================================================================

ENTAILMENT (n=3336):
--------------------------------------------------------------------------------
Signal               Mean       Std        Min        Max        Median    
--------------------------------------------------------------------------------
resonance            0.6234     0.2341     -0.1234    0.9876     0.6543
divergence           -0.4567    0.1234     -0.9876    0.1234     -0.4321
convergence          0.4567     0.1234     -0.1234    0.9876     0.4321
...

KEY INSIGHTS:
================================================================================

1. DIVERGENCE:
   E: -0.4567 ± 0.1234 (should be negative) ✓
   C: 0.3456 ± 0.2345 (should be positive) ✓
   N: 0.0123 ± 0.2345 (should be near zero) ✓
```

### Example Output (Debug Mode)

```
GEOMETRIC PATTERN ANALYSIS (from Golden Labels) - DEBUG MODE
================================================================================

⚠️  DEBUG MODE: Forces are artificially set to match golden labels.
   Geometric signals (resonance, divergence) are REAL from layers 0-3.
   This shows what geometry produces vs what forces are needed.

...

KEY INSIGHTS:
================================================================================

...

6. DEBUG MODE INSIGHT:
   Forces are set to: E(cold=0.7, far=0.2), C(cold=0.2, far=0.7), N(cold=0.33, far=0.33)
   Compare geometric signals above to see if geometry matches these ideal forces.
   If divergence is wrong sign or attractions are weak, geometry needs calibration.
```

## What This Tells You

### Normal Mode Analysis

1. **Calibration Check**
   - If divergence for E is positive (should be negative), your geometry computation is inverted
   - If divergence for C is negative (should be positive), your contradiction force isn't working

2. **Range Discovery**
   - Learn what resonance/divergence ranges actually correspond to each class
   - Use these ranges to calibrate thresholds

3. **Force Analysis**
   - See what force distributions lead to correct classification
   - E: Should have high cold_force, low far_force
   - C: Should have high far_force, low cold_force
   - N: Should have balanced forces or high city_force

### Debug Mode Analysis

1. **Gap Detection**
   - Compare real geometry signals to ideal forces
   - If real `cold_force=0.4` but ideal is `0.7`, geometry is too weak
   - If real `divergence` is wrong sign, geometry is inverted

2. **Calibration Targets**
   - See exactly what forces are needed for each class
   - Calibrate geometry to produce these forces

3. **Signal Quality**
   - Are attractions strong enough to reach ideal forces?
   - Is divergence creating real force?
   - Are forces balanced correctly?

## Using Patterns to Improve Geometry

### Step 1: Run Pattern Learning (Both Modes)

```bash
# Normal mode: See what geometry actually produces
python3 experiments/nli_v5/train_v5.py --clean --train 5000 --learn-patterns

# Debug mode: See what geometry produces vs what's needed
python3 experiments/nli_v5/train_v5.py --clean --train 5000 --debug-golden --learn-patterns
```

### Step 2: Compare Results

Look for:
- **Inverted signals**: E has positive divergence (should be negative)
- **Weak signals**: Divergence near zero for all classes
- **Force gaps**: Real forces much weaker than ideal forces (debug mode)

### Step 3: Adjust Geometry

Based on patterns, adjust:
- Divergence computation (if ranges are wrong)
- Force scaling (if forces are too weak/strong)
- Thresholds (if boundaries are off)

### Step 4: Re-test

Run again and compare patterns - they should improve.

## Example: Finding the Problem

### Normal Mode Shows:

```
DIVERGENCE:
   E: 0.1234 ± 0.2345 (should be negative) ❌
   C: -0.2345 ± 0.3456 (should be positive) ❌
```

**Problem**: Divergence computation is inverted!

**Fix**: Flip the sign in `_compute_field_divergence()`.

### Debug Mode Shows:

```
FORCES:
   E: cold=0.35, far=0.40 (real geometry)
   E: cold=0.70, far=0.20 (ideal - debug mode)
```

**Problem**: Geometry produces weak forces (0.35 vs 0.70 needed)

**Fix**: Boost cold_density computation or increase basin weights.

## Advanced: Pattern-Based Calibration

You can use patterns to auto-calibrate:

```python
from experiments.nli_v5.pattern_learner import PatternLearner

# Load normal mode patterns
learner_normal = PatternLearner()
learner_normal.load_patterns('patterns_normal.json')

# Load debug mode patterns
learner_debug = PatternLearner(debug_mode=True)
learner_debug.load_patterns('patterns_debug.json')

# Compare real vs ideal
e_real = learner_normal.stats['entailment']['signals']['cold_force']['mean']
e_ideal = learner_debug.stats['entailment']['signals']['cold_force']['mean']

# Calibration factor
calibration_factor = e_ideal / e_real  # e.g., 0.70 / 0.35 = 2.0
# Use this to boost cold_density computation
```

## Key Files

- `pattern_learner.py` - Pattern recording and analysis
- `train_v5.py` - Training script with `--learn-patterns` and `--debug-golden` flags
- `patterns.json` - Saved patterns (generated after training)

## Philosophy

**You're not debugging a bug. You're discovering physics.**

Pattern learning helps you understand what the geometry SHOULD look like. Debug mode shows you the gap between reality and ideal. Once you know the patterns, you can calibrate your geometry to match reality.

This is the scientific method:
1. Observe (record patterns in normal mode)
2. Compare (record patterns in debug mode)
3. Analyze (understand the gap)
4. Hypothesize (what should the geometry be?)
5. Test (adjust and re-run)
