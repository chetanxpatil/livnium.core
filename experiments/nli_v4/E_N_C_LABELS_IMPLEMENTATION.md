# E/N/C Labels Implementation

## Overview

This implementation adds E/N/C label extraction from the geometry-based physics system and stabilizes the three-phase universe using frozen basin centers.

## Key Changes

### 1. Label Extraction (`layer7_decision.py`)

Added `decide()` method to `Layer7Decision` that implements the simple force-competition rule:

```python
def decide(self, cold_force: float, far_force: float, city_force: float) -> str:
    """
    Simple force-competition rule to get E/N/C label.
    
    Rule:
    - if city > max(cold, far): return "N"
    - elif cold > far: return "E"
    - else: return "C"
    """
```

**Meaning = strongest force direction:**
- Cold force → Entailment (E)
- Far force → Contradiction (C)
- City force → Neutral (N)

### 2. Training Integration (`train_v4.py`)

- Extracts forces from `basin_forces` in layer states
- Calls `decide()` to get E/N/C label for every example
- Stores predicted labels in cluster tracker
- Logs predicted labels during training

### 3. Cluster Tracking (`cluster_tracker.py`)

- Added `predicted_label` parameter to `add()` method
- Stores E/N/C labels in cluster entries
- Labels are included in exported cluster JSON files

### 4. Frozen Basin Centers (`frozen_basin_centers.py`)

New class that stabilizes the three-phase universe by:

- **Tracking vectors**: Collects sentence vector representations for each basin
- **Computing frozen centers**: Uses EMA (Exponential Moving Average) to compute stable centers
- **Stabilizing attractors**: Freezes the center of each class:
  - `frozen_cold_center` = mean of Cold (E) sentence vectors
  - `frozen_far_center` = mean of Far (C) sentence vectors
  - `frozen_city_center` = mean of City (N) sentence vectors

**Key Methods:**
- `add_vector(basin_index, sentence_vector)`: Add vector to current cycle
- `update_frozen_centers()`: Update frozen centers using EMA
- `get_frozen_center(basin_index)`: Get frozen center for a basin
- `compute_attraction_to_frozen_center(sentence_vector, basin_index)`: Compute attraction strength

## Usage

### During Unsupervised Training

```python
# Labels are automatically extracted and stored
python train_v4.py --unsupervised --cluster-output clusters/

# Output includes:
# - Predicted E/N/C labels in cluster JSON files
# - Frozen basin centers statistics
# - Label predictions logged every 500 steps
```

### Example Output

```
Step 500: Basin 0=45 | Basin 1=12 | Basin 2=443 | Pred=N | Moksha=0.123 | ...
```

### Cluster JSON Format

```json
{
  "premise": "...",
  "hypothesis": "...",
  "confidence": 0.85,
  "predicted": "N",  // <-- E/N/C label added
  "basin_forces": {...}
}
```

## Why This Works

1. **Labels are not magic**: They're just names for stable attractors
2. **Geometry already has meaning**: The physics system already discovers Cold/Far/City basins
3. **Force competition**: The strongest force direction determines the label
4. **Frozen centers stabilize**: By freezing attractor centers, meaning stops drifting

## Physics

The system implements a **three-phase universe**:

- **Phase 1 - Cold (E)**: Dense, inward-pull, low energy
- **Phase 2 - Far (C)**: Distance, repulsion, edge of continent
- **Phase 3 - City (N)**: Overlap region, balance point

When frozen centers are enabled:
- Meaning stops drifting
- Labels stop flickering
- The system stops looping forever
- Geometry becomes a permanent structure

## Future Enhancements

- Use frozen centers in layer routing (Layer 4: Meta Routing)
- Use frozen centers in basin attraction (Layer 2: Basin)
- Add frozen center visualization
- Export frozen centers for reuse

