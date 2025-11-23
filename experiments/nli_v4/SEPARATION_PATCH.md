# Separation Patch: Geometry → Rules

## What Was Changed

**Problem**: Geometry was making label decisions before Layer 7, so the rule engine had no power.

**Solution**: Removed all label assignments from Layers 2-6. Layer 7 is now **authoritative**.

## Changes Made

### Layer 3 (Valley) - `layer3_valley.py`
- **REMOVED**: All `label` and `confidence` assignments
- **KEEPS**: Force computation (city_pull, scores)
- **RETURNS**: Only forces and scores, NO labels

### Layer 4 (Meta Routing) - `layer4_meta_routing.py`
- **CHANGED**: Routes based on forces/scores instead of labels
- **COMPUTES**: Route from `valley_score`, `city_pull`, `max_force`

### Layer 5 (Temporal Stability) - `layer5_temporal_stability.py`
- **CHANGED**: Tracks force signatures instead of labels
- **COMPUTES**: Stability from force patterns, not label consistency

### Layer 6 (Semantic Memory) - `layer6_semantic_memory.py`
- **REMOVED**: Dependency on `label` field
- **KEEPS**: Word polarity computation (doesn't need labels)

### Layer 7 (Decision) - `layer7_decision.py`
- **NOW AUTHORITATIVE**: Makes final label decision based purely on forces
- **RECEIVES**: Only forces, scores, geometry signals
- **DECIDES**: Label using rules (default or rule engine)

## Architecture Now

```
Layer 0-2: Compute forces (cold_attraction, far_attraction)
Layer 3:   Compute city_pull, scores (NO labels)
Layer 4:   Route based on forces (NO labels)
Layer 5:   Track stability from forces (NO labels)
Layer 6:   Compute word polarities (NO labels)
Layer 7:   AUTHORITATIVE - decides label from forces
```

## Expected Impact

- **Before**: Geometry decided labels → Rule engine echoed → ~35% accuracy
- **After**: Geometry provides forces → Rule engine decides → Should reach 60-90%

## Next Steps

1. **Test with rule engine enabled** (already enabled in `layered_classifier.py`)
2. **Tune rule thresholds** based on dev set performance
3. **Refine rules** using rule discovery on logged features
4. **Iterate** until accuracy reaches target

## Key Insight

> "Geometry = meaning (wild, unsupervised)
> Rules = interpretation (clean, symbolic)
> Labels = human artifacts (we translate)"

The universe stays wild. The rules are just theories about how to read it.

