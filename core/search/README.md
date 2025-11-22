# Search Module: Dynamic Basin & Multi-Basin Search

This module contains search algorithms and basin reinforcement mechanisms for geometric problem solving.

## Components

### 1. Dynamic Basin Reinforcement (`native_dynamic_basin_search.py`)

Geometry-driven, self-tuning basin shaping that adapts to:
- **Curvature**: Basin depth
- **Tension**: Internal contradictions
- **Entropy**: State disorder

**Key Principle**: Geometry decides the physics, not a parameter list.

See `HOW_IT_WORKS.md` for complete explanation.

### 2. Multi-Basin Search (`multi_basin_search.py`)

Multiple competing attractors in geometric space:
- **Basin objects**: Represent candidate solutions
- **Competition**: Basins compete in shared geometry
- **Winner selection**: Best basin (highest score) wins
- **Natural selection**: Losing basins decay, winning basin reinforces

See `MULTI_BASIN_SEARCH.md` for complete explanation.

### 3. Corner Rotation Policy (`corner_rotation_policy.py`)

Post-convergence refinement physics:
- **Corners are max-exposure cells** (face_exposure = 3, SW = 27)
- **Early/mid process**: Corner rotations destabilize the lattice (locked)
- **End process**: Corner rotations fix final parity, global symmetry, SW distribution (unlocked)
- **Unlock condition**: `basin_depth > threshold AND drift < epsilon`
- **Same physics as Rubik's cubes**: Last moves are almost always corner parity fixes

This prevents chaos early but permits perfect symmetry at the end.

## Quick Start

### Dynamic Basin (Single Basin)

```python
from core.search import update_basin_dynamic

# Update basin based on correctness
update_basin_dynamic(system, task, is_correct=True)
```

### Multi-Basin (Competing Basins)

```python
from core.search import MultiBasinSearch, solve_with_multi_basin

# Create search
search = MultiBasinSearch(max_basins=10)

# Add candidate solutions
for coords in candidate_solutions:
    search.add_basin(coords, system)

# Run competition
for step in range(100):
    search.update_all_basins(system)
    winner = search.get_winner()
    if winner:
        break
```

## Architecture Layers

```
Layer 4: Multi-Basin Competition  ← You are here
    ↓
Layer 3: Universal Encoding       ← Next step
    ↓
Layer 2: Task Encoding             ← Future
    ↓
Layer 1: Dynamic Basin Physics     ← Complete
    ↓
Layer 0: LivniumCoreSystem         ← Base
```

## Documentation

- `HOW_IT_WORKS.md` - Dynamic basin reinforcement explained
- `MULTI_BASIN_SEARCH.md` - Multi-basin search explained
- `test_native_dynamic_basin.py` - Dynamic basin tests
- `test_multi_basin.py` - Multi-basin tests

## Philosophy

**"Geometry decides the physics, not a parameter list."**

The search module implements self-tuning systems that adapt to the geometry itself, creating stable, convergent behavior without static hyperparameters.

## Corner Rotation Policy

Corners (max-exposure cells) are the **final degree of freedom** in a closed cube universe. They must be locked during exploration but unlocked during post-convergence refinement to fix:
- Final parity constraints
- Global symmetry alignment
- SW distribution
- Basin geometry finalization
- Ghost tension removal

This is not a hack—it's a law of the universe. The same physics applies to Rubik's cubes, cube symmetry groups, and basin convergence algebra.
