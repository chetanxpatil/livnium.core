# Rule Engine: Clean Symbolic Logic from Geometry

## Overview

The Rule Engine is the **"priest of rules"** - it watches geometry and translates it into clean symbolic logic that aligns with human labels.

**Philosophy:**
- Geometry produces meaning (wild, unsupervised)
- Rules interpret geometry (clean, symbolic)
- Labels are human artifacts (we translate)

## Architecture

```
Geometry (wild) → Features → Rule Engine → Labels (human)
```

The rule engine:
1. Takes geometric features (cold_attraction, far_force, basin_conf, etc.)
2. Evaluates symbolic if-then rules
3. Returns E/N/C label with confidence

## Quick Start

### 1. Generate Hand-Tuned Rules

```bash
python3 experiments/nli_v4/rule_engine.py --hand-tuned --save experiments/nli_v4/rules_hand_tuned.json
```

### 2. Load from Discovered Rules

If you have `discovered_rules.json` from rule discovery:

```bash
python3 experiments/nli_v4/rule_engine.py --discovered discovered_rules.json --save experiments/nli_v4/rules_discovered.json
```

### 3. Enable Rule Engine in Layer 7

Edit `layered_classifier.py`:

```python
# Change this line:
self.layer7 = Layer7Decision(use_rule_engine=False)

# To this:
from .rule_engine import RuleEngine
rule_engine = RuleEngine.from_hand_tuned_rules()  # or load from file
self.layer7 = Layer7Decision(use_rule_engine=True, rule_engine=rule_engine)
```

### 4. Test It

```bash
python3 experiments/nli_v4/train_v4.py --train 1000
```

## Rule Structure

Rules are evaluated in order (first match wins):

```python
{
    'condition': 'basin_conf > 0.70 and cold_attraction > 0.3',
    'label': 'E',
    'description': 'High confidence + cold attraction → Entailment'
}
```

Conditions support:
- Simple comparisons: `feature > value`, `feature < value`
- Compound: `feature1 > value1 and feature2 < value2`
- Default: `'else'` catches everything

## Hand-Tuned Rules

The hand-tuned rule set includes:

1. **Very High Confidence** (basin_conf > 0.80)
   - Trust the basin signal
   
2. **High Confidence + Clear Signal** (basin_conf > 0.65)
   - Use geometry hints (resonance, forces)
   
3. **Balanced Forces** (force_ratio < 0.15)
   - → Neutral
   
4. **Weak Forces** (max_force < 0.05)
   - → Neutral
   
5. **City Dominates** (city_force > 0.6)
   - → Neutral
   
6. **Medium Confidence** (0.40 ≤ basin_conf ≤ 0.65)
   - Use force competition with thresholds
   
7. **Low Confidence** (basin_conf < 0.40)
   - → Neutral (uncertain)
   
8. **Force Competition** (fallback)
   - Cold > Far → E
   - Else → C

## Expected Results

With hand-tuned rules, you should see:
- **Baseline (default)**: ~35% accuracy
- **With rules**: ~50-60% accuracy (immediate jump)
- **After refinement**: ~70-80% accuracy
- **After tuning**: ~80-90% accuracy

## Refining Rules

1. **Collect features** with current rules:
   ```bash
   python3 experiments/nli_v4/train_v4.py --train 5000 --log-features features.csv
   ```

2. **Discover new patterns**:
   ```bash
   python3 experiments/nli_v4/rule_discovery.py --features features.csv
   ```

3. **Update rules** based on discoveries

4. **Test and iterate**

## Why This Works

The decision tree showed:
- `basin_conf` alone can classify ~50% of SNLI
- Geometry already contains implicit E/N/C structure
- Rules just need to align force-interpretation

The rule engine:
- Extracts clean patterns from geometry
- Converts them to symbolic logic
- Maintains interpretability
- Can evolve over time

## Next Steps

1. **Test hand-tuned rules** - see immediate accuracy jump
2. **Refine thresholds** - tune based on dev set performance
3. **Add physics penalties** - prevent city collapse, basin monopoly
4. **Automate evolution** - create auto_rules.py for rule search

## Philosophy

> "Geometry stays the god of meaning. We just build a little priest of rules that watches, tests, and refines, until it's right 9 times out of 10."

The universe stays wild. The rules are just theories about how to read it.

