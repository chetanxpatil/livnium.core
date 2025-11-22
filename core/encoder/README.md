# Universal Encoder: Layer 3

## Overview

The Universal Encoder converts **any problem** into geometric patterns:
- **Constraints** → Tension fields (energy landscape)
- **Solutions** → Basins (candidate attractors)

This is Layer 3 of the universal problem solver architecture:

```
Layer 4: Multi-Basin Competition  ← Complete
    ↓
Layer 3: Universal Encoding       ← You are here
    ↓
Layer 2: Task Encoding             ← Future
    ↓
Layer 1: Dynamic Basin Physics     ← Complete
    ↓
Layer 0: LivniumCoreSystem         ← Base
```

## Core Principle (Corrected)

**Constraints define the *tension landscape*.  
Basins represent *solutions that minimize that tension*.**

### In Symbols

```
constraints → tension patches  
solutions → attractors (basins)  
search → basin competition + tension minimization
```

### Mapping

```
Problem Domain          →  Geometric Pattern
─────────────────────────────────────────────
Variables             →  Cell groups
Constraints           →  Tension fields (NOT basins!)
Solutions             →  Basins (attractors)
Dependencies         →  Coupling patterns
Relations            →  SW connections
```

## Key Correction

### ❌ Wrong (Previous Understanding)
- Constraints → Basin shapes
- This mixes up constraints with solutions

### ✅ Correct (Current Understanding)
- **Constraints** → **Tension fields** (energy landscape)
- **Solutions** → **Basins** (candidate attractors)
- Search finds basins that minimize constraint tension

## What the Universal Encoder Does

### 1. Identify Variables
Map each variable → one or more coordinates:
```python
variable_1 → [(x₁, y₁, z₁), ...]
variable_2 → [(x₂, y₂, z₂), ...]
```

### 2. Identify Constraints
For each constraint:
- Create a local patch of coordinates
- Define how to compute **tension** if violated
- Define **curvature** if satisfied

```python
constraint: "x₁ + x₂ = 5"
    → Tension field: high tension if x₁ + x₂ ≠ 5
    → Curvature: low tension (satisfied) if x₁ + x₂ = 5
```

### 3. Generate Candidate Basins
Depending on the problem:
- Candidate colorings (graph coloring)
- Partial assignments (SAT)
- Full assignments (Ramsey)
- Path proposals (pathfinding)

### 4. Hand Basins + Tension Field to Multi-Basin Layer
- Basins compete in the tension landscape
- Best basin (lowest tension) wins
- Multi-basin search finds the solution

## Architecture

### Components

1. **`problem_encoder.py`** ✅
   - Main universal encoder interface
   - Converts problem → tension fields + candidate basins

2. **`constraint_encoder.py`** ✅
   - Encodes constraints as **tension fields**
   - Computes tension for constraint violations
   - NOT basin shapes - tension patches!

3. **`graph_encoder.py`** ⏳ (Planned)
   - Encodes graphs as geometric structures
   - K₄ patches → tension fields
   - Candidate colorings → basins

4. **`logic_encoder.py`** ⏳ (Planned)
   - Encodes logical formulas
   - Clauses → tension fields
   - Assignments → basins

5. **`language_encoder.py`** ⏳ (Planned)
   - Encodes natural language
   - Semantic constraints → tension
   - Interpretations → basins

## What Problems Can Be Encoded?

### ✅ Logical Problems
- **SAT** (Boolean satisfiability)
  - Clauses → tension fields
  - Assignments → basins

- **Propositional logic**
  - Formulas → tension
  - Models → basins

### ✅ Graph Problems
- **Graph coloring**
  - Monochromatic triangles → tension
  - Colorings → basins

- **Ramsey problems**
  - K₄ constraints → tension fields
  - 2-colorings → basins

### ✅ Optimization Problems
- **Feature selection**
  - Constraints → tension
  - Feature sets → basins

### ✅ Language Problems
- **SNLI** (Natural language inference)
  - Semantic constraints → tension
  - Interpretations → basins

## Usage

```python
from core.encoder import UniversalProblemEncoder
from core.search import solve_with_multi_basin

# Create encoder
encoder = UniversalProblemEncoder(system)

# Encode a problem
problem = {
    'type': 'graph_coloring',
    'vertices': [0, 1, 2, 3, 4],
    'edges': [(0,1), (1,2), ...],
    'n_candidates': 10
}

# Convert to geometry
encoded = encoder.encode(problem)

# Returns:
# - encoded.tension_fields (from constraints)
# - encoded.candidate_basins (from solution space)

# Solve with multi-basin search
winner, steps, stats = solve_with_multi_basin(
    system,
    encoded.candidate_basins,
    max_steps=1000
)
```

## Status

**Status**: ✅ **Core Architecture Complete**

- ✅ Constraint encoder (tension fields)
- ✅ Problem encoder (main interface)
- ✅ Graph coloring encoding
- ⏳ SAT encoding (TODO)
- ⏳ General CSP encoding (TODO)
- ⏳ Language encoder (TODO)

## Test Results

```
✓ Constraints encoded as tension fields (NOT basins)
✓ Tension increases when constraint violated
✓ Tension decreases when constraint satisfied
✓ Multiple constraints create tension landscape
```

## Next Steps

1. Complete SAT encoding
2. Complete general CSP encoding
3. Add language encoder (SNLI, entailment)
4. Integrate with multi-basin search
5. Test with real problems

---

**Key Insight**: Constraints create the energy landscape (tension). Solutions are attractors (basins) that minimize that tension. The search finds the best basin.
