# Search Module: Dynamic Basin & Multi-Basin Search for Livnium-T

This module contains search algorithms and basin reinforcement mechanisms for geometric problem solving in tetrahedral/simplex geometry.

## Components

### 1. Dynamic Basin Reinforcement (`native_dynamic_basin_search.py`)

Geometry-driven, self-tuning basin shaping that adapts to:
- **Curvature**: Basin depth
- **Tension**: Internal contradictions
- **Entropy**: State disorder

**Key Principle**: Geometry decides the physics, not a parameter list.

Adapted for Livnium-T's 5-node topology (node IDs instead of coordinates).

### 2. Multi-Basin Search (`multi_basin_search.py`)

Multiple competing attractors in simplex space:
- **Basin objects**: Represent candidate solutions (using node IDs)
- **Competition**: Basins compete in shared geometry (SW fields)
- **Winner selection**: Best basin (highest score) wins
- **Natural selection**: Losing basins decay, winning basin reinforces

### 3. Vertex Rotation Policy (`vertex_rotation_policy.py`)

Post-convergence refinement physics:
- **Vertices are max-exposure nodes** (exposure f = 3, SW = 27)
- **Early/mid process**: Vertex rotations destabilize the simplex (locked)
- **End process**: Vertex rotations fix final parity, global symmetry, SW distribution (unlocked)
- **Unlock condition**: `basin_depth > threshold AND drift < epsilon`
- **Similar to Rubik's cubes**: Last moves are almost always corner parity fixes

This prevents chaos early but permits perfect symmetry at the end.

## Key Differences from Core Search

| Feature | Core Search | Livnium-T Search |
|---------|------------|------------------|
| **Geometry** | Cubic (3×3×3 lattice) | Tetrahedral (5-node topology) |
| **Coordinates** | 3D tuples (x,y,z) | Node IDs (0-4) |
| **Max Exposure** | Corners (f=3) | Vertices (f=3) |
| **Rotation Group** | Cubic (24 elements) | Tetrahedral (12 elements) |
| **Policy** | Corner rotation policy | Vertex rotation policy |

## Quick Start

### Dynamic Basin (Single Basin)

```python
from core_t.search import update_basin_dynamic

# Update basin based on correctness
# Task must have input_node_ids and output_node_id attributes
update_basin_dynamic(system, task, is_correct=True)
```

### Multi-Basin (Competing Basins)

```python
from core_t.search import MultiBasinSearch, solve_with_multi_basin

# Create search
search = MultiBasinSearch(max_basins=10)

# Add candidate solutions (using node IDs)
candidate_1 = [1, 2, 3]  # Nodes 1, 2, 3
candidate_2 = [2, 3, 4]  # Nodes 2, 3, 4
search.add_basin(candidate_1, system)
search.add_basin(candidate_2, system)

# Run competition
for step in range(100):
    stats = search.update_all_basins(system)
    winner = search.get_winner()
    if winner:
        break
```

### Vertex Rotation Policy

```python
from core_t.search import should_allow_vertex_rotations

# Check if vertex rotations are allowed
allow = should_allow_vertex_rotations(
    system,
    active_node_ids=[1, 2, 3],
    convergence_stats={'num_alive': 1}
)
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
Layer 0: LivniumTSystem            ← Base
```

## Philosophy

**"Geometry decides the physics, not a parameter list."**

The search module implements self-tuning systems that adapt to the geometry itself, creating stable, convergent behavior without static hyperparameters.

## Vertex Rotation Policy

Vertices (max-exposure nodes) are the **final degree of freedom** in a closed simplex universe. They must be locked during exploration but unlocked during post-convergence refinement to fix:
- Final parity constraints
- Global symmetry alignment
- SW distribution
- Basin geometry finalization
- Ghost tension removal

This is not a hack—it's a law of the universe. The same physics applies to Rubik's cubes, symmetry groups, and basin convergence algebra.

## Node ID Reference

- **0**: Core (Om) - f=0, SW=0, immovable
- **1-4**: Vertices (LOs) - f=3, SW=27 each, movable

## Example Usage

```python
from core_t.classical import LivniumTSystem
from core_t.search import MultiBasinSearch

# Create system
system = LivniumTSystem()

# Create search
search = MultiBasinSearch(max_basins=5)

# Add candidate solutions
search.add_basin([1, 2], system)  # Candidate 1: vertices 1, 2
search.add_basin([2, 3], system)  # Candidate 2: vertices 2, 3
search.add_basin([3, 4], system)  # Candidate 3: vertices 3, 4

# Run search
for step in range(50):
    stats = search.update_all_basins(system)
    print(f"Step {step}: {stats['num_alive']} alive, winner: {stats['winner']}")
    
    if stats['num_alive'] == 1:
        break

# Get winner
winner = search.get_winner()
if winner:
    print(f"Winner: nodes {winner.active_node_ids}, score: {winner.score}")
```

