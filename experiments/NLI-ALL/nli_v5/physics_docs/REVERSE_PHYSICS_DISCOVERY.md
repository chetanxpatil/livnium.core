# Reverse Physics Discovery: Probing Invariant Structure

## The Concept

Imagine you have a universe made of springs, magnets, and gravity wells... and then you suddenly tell the universe:

**"Everything you think is wrong. Everything you believed is false. Now show me what structure you keep anyway."**

This is not chaos. This is **reverse-physics discovery**.

## What This Experiment Does

**Force the system to classify everything incorrectly**:
- Give *wrong* labels (invert E/C/N)
- Intentionally mislabel examples
- Watch what the geometry does

This is like hitting the geometry with a hammer to see where it bends.

It's not training. It's not debugging. It's **probing the invariant structure** inside the omcube.

## Why This Works

Because in Livnium, **the physics is deeper than the labels**.

Labels are surface. Geometry is the core.

If you force contradictions labeled as entailment, and entailments labeled as contradiction... but the **geometry still outputs positive or negative divergence**, you discover:

### *What the geometry thinks the world looks like, not what the labels say.*

This is how real scientists test laws:
- Invert conditions
- Change sign conventions
- Scramble inputs
- Observe what stays invariant

If something **survives inversion**, it is a *true invariant* of the system.

## What Will Happen

### Before Inversion (Normal Mode)

- Entailment → negative divergence
- Contradiction → positive divergence
- Neutral → near-zero divergence

### After Forcing Wrong Labels

The system will behave like this:

### 1. Decision Layer Will Obey Forced Labels

...but the geometry underneath will **not** obey.

### 2. Pattern Learner Will Record Mismatches

You will see signals like:
- Entailment (forced) → divergence = +0.10 (geometry says contradiction!)
- Contradiction (forced) → divergence = -0.05 (geometry says entailment!)
- Neutral (forced) → divergence = random mid-range

The geometry will refuse to lie.

### 3. Hidden Pattern Will Emerge

This pattern will tell you:

**"Regardless of labels, these signals stay stable."**

That is your invariant geometry.

## What You Will Discover

You'll reveal which feature is **the true axis**:

- Does resonance refuse to move?
- Does divergence always maintain its sign?
- Does cold density cluster no matter what you force?
- Do attraction ratios resist inversion?

This is the geometric equivalent of:

**"Show me your core beliefs by flipping your world upside down."**

## What Pattern to Look For

Run forced-wrong labels and record:

```
resonance
divergence
cold_density
cold_attraction
far_attraction
curvature
opposition
```

Then check:

### 1. Which Signal Refuses to Flip?

That is your **primary semantic axis**.

(You'll probably see divergence stays stable.)

### 2. Which Signal Follows the Wrong Label?

That signal is *not fundamental*. You can remove it or demote it.

### 3. Which Combination Clusters Naturally?

That cluster is your **true neutral zone**.

### 4. Which Forces Collapse?

Those forces are artifacts, not physics.

## Why Nobody Does This

Deep learning can't do this. If you invert labels in a neural network, it collapses into noise.

Livnium is not a neural network. Livnium is a geometric universe.

It has its own physics.

If you force physics to break... it shows you what cannot be broken.

This is the deepest method in physics:
- Perturb the system
- Invert the system
- Break the system
- Watch what refuses to break

That unbreakable thing is the law.

## Critical Rule

### Do **NOT** Train With Wrong Labels

Just *evaluate* with wrong labels and record the geometry.

If you train, the system will learn garbage.

But if you **diagnose** with wrong labels, you will see its **internal invariants**.

## Expected Outcome (Prediction)

You will find:

- ✅ **Divergence stays stable** (refuses to flip)
- ✅ **Resonance barely shifts** (resistant to inversion)
- ✅ **Attractions shift slightly** (somewhat label-dependent)
- ✅ **Curvature stays stable** (geometric invariant)
- ✅ **Cold density tightly clusters** (geometric truth)
- ✅ **Neutral cluster becomes more visible** (true balance zone)
- ✅ **C/E cluster separation stays** (geometry knows the truth)
- ⚠️ **Force-based interpretations get confused** (artifacts)
- ✅ **Geometry refuses to be lied to** (physics is deeper)

This experiment will **reveal the true topology** of your meaning-space.

It will tell you:

**"Here is what Livnium actually believes meaning looks like."**

And that is priceless.

## Implementation

### ✅ Step 1: Inverted Label Mode (IMPLEMENTED)

The `--invert-labels` flag has been added to `train_v5.py`:

```bash
python3 experiments/nli_v5/train_v5.py \
  --clean \
  --train 1000 \
  --invert-labels \
  --learn-patterns \
  --pattern-file experiments/nli_v5/patterns_inverted.json
```

**What it does**:
- Inverts labels: E↔C (neutral stays as-is)
- Forces inverted label into debug mode (wrong forces)
- Records geometric signals with inverted label
- **NO TRAINING** - diagnostic only

### ✅ Step 2: Pattern Recording (IMPLEMENTED)

Pattern learner records geometric signals with inverted labels:
- Forces are set to match inverted labels (wrong)
- Geometry (divergence, resonance) is REAL from layers 0-3
- Patterns saved show what geometry produces when forced to say wrong thing

### ✅ Step 3: Analyze Invariants (IMPLEMENTED)

Use `compare_inverted_patterns.py` to find invariant signals:

```bash
python3 experiments/nli_v5/compare_inverted_patterns.py \
  --normal-file experiments/nli_v5/patterns_normal.json \
  --inverted-file experiments/nli_v5/patterns_inverted.json
```

This will show:
- Which signals refuse to flip (invariants)
- Which signals follow wrong labels (artifacts)
- Divergence sign preservation analysis

## What This Reveals

### The True Semantic Axes

Signals that refuse to flip are **fundamental**:
- Divergence sign (probably stays stable)
- Resonance magnitude (probably stays stable)
- Curvature (probably stays stable)

### The Artifacts

Signals that flip with labels are **not fundamental**:
- Force-based interpretations
- Label-dependent features
- Superficial correlations

### The True Neutral Zone

The cluster that forms naturally, regardless of labels, is the **real neutral phase**.

## Usage

### Step 1: Generate Normal Patterns

```bash
# Generate normal patterns (correct labels)
python3 experiments/nli_v5/train_v5.py \
  --clean \
  --train 1000 \
  --learn-patterns \
  --pattern-file experiments/nli_v5/patterns_normal.json
```

### Step 2: Generate Inverted Patterns

```bash
# Generate inverted patterns (wrong labels, diagnostic only)
python3 experiments/nli_v5/train_v5.py \
  --clean \
  --train 1000 \
  --invert-labels \
  --learn-patterns \
  --pattern-file experiments/nli_v5/patterns_inverted.json
```

**Note**: Accuracy will be very low (~20-30%) because labels are wrong. This is expected!

### Step 3: Compare to Find Invariants

```bash
# Compare normal vs inverted to find what refuses to flip
python3 experiments/nli_v5/compare_inverted_patterns.py \
  --normal-file experiments/nli_v5/patterns_normal.json \
  --inverted-file experiments/nli_v5/patterns_inverted.json
```

This will show:
- ✓ **INVARIANT** signals (refuse to flip - true geometric laws)
- ✗ **ARTIFACT** signals (follow wrong labels - not fundamental)

## The Deep Idea

This is **reverse-physics discovery**:

1. **Force the system to lie** (wrong labels)
2. **Watch what refuses to lie** (invariant geometry)
3. **Discover the true laws** (what cannot be broken)

The geometry will show you its **core beliefs** by refusing to change them, even when you force it to say the opposite.

## Status

✅ **IMPLEMENTED** - Ready to run!

- `--invert-labels` flag added to `train_v5.py`
- Pattern learner handles inverted mode
- Comparison script `compare_inverted_patterns.py` created
- Documentation complete

## Next Steps

1. Implement `--invert-labels` flag
2. Run diagnostic (no training, just evaluation)
3. Compare inverted vs normal patterns
4. Identify invariant signals
5. Document findings

## References

- Debug Mode: `DEBUG_MODE.md` (similar diagnostic approach)
- Pattern Learning: `PATTERN_LEARNING.md` (pattern extraction)
- Physics Laws: `core/law/` (established laws)

## Philosophy

This experiment tests the **robustness** of the physics:

- If divergence flips with labels → divergence is not fundamental
- If divergence stays stable → divergence is a true law
- If resonance resists inversion → resonance is geometric truth
- If forces collapse → forces are artifacts, not physics

The unbreakable signals are the **laws of the universe**.

