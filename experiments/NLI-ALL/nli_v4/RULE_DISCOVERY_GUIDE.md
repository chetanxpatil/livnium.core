# Rule Discovery Guide: The Priest of Rules

## Overview

This implements the **three-loop architecture**:

1. **Inner Loop - Geometry/AME**: Unsupervised physics, discovers meaning
2. **Middle Loop - Rule Maker**: Learns rules to read geometry correctly
3. **Outer Loop - Truth Game**: Optimizes for ~90% accuracy on SNLI

**Key Philosophy**: Geometry stays wild. Rules are just theories about how to read it.

## Quick Start

### Step 1: Collect Features

Run training with feature logging:

```bash
python3 experiments/nli_v4/train_v4.py \
    --train 5000 \
    --log-features experiments/nli_v4/features.csv
```

This logs geometric features + true labels to CSV.

### Step 2: Discover Rules

Train a decision tree on the features:

```bash
python3 experiments/nli_v4/rule_discovery.py \
    --features experiments/nli_v4/features.csv \
    --max-depth 4 \
    --output experiments/nli_v4/discovered_rules.json
```

This will:
- Train a shallow decision tree
- Print discovered rules
- Suggest Layer 7 parameters
- Save results to JSON

### Step 3: Apply Rules

Manually port the discovered rules into `layer7_decision.py` and rerun to see improvement.

## Architecture

### Feature Logger (`feature_logger.py`)

Logs geometric signals:
- Basin info (basin_id, basin_conf)
- Core forces (cold_attraction, far_attraction, city_pull)
- Normalized forces (cold_force, far_force, city_force)
- Geometry signals (resonance, curvature, max_force, force_ratio)
- Scores (e_score, c_score, n_score)
- Stability signals (is_stable, is_moksha, route)

### Rule Discovery (`rule_discovery.py`)

Learns rules from features:
- Trains decision tree (shallow, interpretable)
- Extracts if-then rules
- Suggests Layer 7 parameters
- Evaluates accuracy

### Integration (`train_v4.py`)

- `--log-features`: Enable feature logging
- Features logged automatically during training
- Works in both supervised and unsupervised modes

## Example Output

```
DISCOVERED RULES
======================================================================

Accuracy: 0.8523 (5000 examples)

Decision Tree Rules:
----------------------------------------------------------------------
|--- cold_force <= 0.45
|   |--- city_force > 0.55
|   |   |--- class: N
|   |--- city_force <= 0.55
|   |   |--- cold_attraction > 0.3
|   |   |   |--- class: E
|   |   |--- cold_attraction <= 0.3
|   |   |   |--- class: C
|--- cold_force > 0.45
|   |--- class: E
----------------------------------------------------------------------

Feature Importance:
----------------------------------------------------------------------
  cold_force                 0.3421
  city_force                 0.2812
  force_ratio                0.1567
  ...
----------------------------------------------------------------------
```

## Next Steps

1. **Automate Rule Search**: Create `auto_rules.py` that:
   - Runs geometry forward
   - Searches over rule parameters
   - Evaluates with physics penalties
   - Keeps best rules

2. **Physics Penalties**: Add penalties for:
   - City collapse (everything → Cold)
   - Basin monopoly (one basin dominates)
   - Loss of diversity

3. **Rule Templates**: Define rule templates that can be:
   - Mutated
   - Evaluated
   - Evolved

## Philosophy

> "Geometry stays the god of meaning. We just build a little priest of rules that watches, tests, and refines, until it's right 9 times out of 10."

The universe stays wild. The rules are just theories about how to read it.

