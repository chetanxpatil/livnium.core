# Law Extraction Module

Auto-discovery of physical laws from Livnium Core.

## Purpose

This module enables Livnium to **discover its own physical laws** instead of having them hardcoded. It observes system behavior and extracts:

- **Invariants** (conserved quantities that remain constant)
- **Functional relationships** (e.g., `divergence = 0.38 - alignment`)

## How It Works

1. **Record States**: Each timestep, the system exports its physics state
2. **Detect Invariants**: Quantities that remain constant are identified as conservation laws
3. **Detect Relationships**: Linear relationships between variables are discovered
4. **Extract Laws**: The system outputs discovered laws in human-readable format

## Quick Start

### Run the Example Script

```bash
python3 core/law/example_law_extraction.py
```

This will:
1. Create a Livnium system
2. Run it for 50 timesteps
3. Extract and display discovered laws

### Use in Your Own Code

```python
from core.runtime.orchestrator import Orchestrator
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig

# Create system
config = LivniumCoreConfig(lattice_size=3)
system = LivniumCoreSystem(config)
orchestrator = Orchestrator(system)

# Run system for N steps
for _ in range(100):
    orchestrator.step()

# Extract discovered laws
laws = orchestrator.extract_laws()
print(orchestrator.get_law_summary())
```

## Example Output

### Static System (No Evolution)
If you run a static system (no rotations, no basin updates), you'll see:
```
Invariants (Conserved Quantities):
  - SW_sum: 486.000000 (constant)
  - alignment: 1.000000 (constant)
  - All quantities constant

Functional Relationships:
  (Many spurious relationships from fitting lines to constant data)
```

**This is correct!** A frozen system has perfect invariants but no evolving laws.

### Evolving System (With Forces)
When the system actually evolves (rotations, basin updates, dynamic forces):
```
Invariants (Conserved Quantities):
  - SW_sum: 486.000000 (constant)  ← Fundamental conservation law!

Functional Relationships:
  - energy = 1.000000 * SW_sum      ← Real law: energy equals SW
  - tension = -0.003252 * SW_sum + 3.080375  ← Relationship between tension and SW
```

**These are real physical laws** discovered from system behavior!

## Integration

The law extractor is automatically integrated into the `Orchestrator`:
- Records physics state after each timestep
- Can extract laws at any time
- Provides human-readable summaries

## Important: System Must Evolve

**The law extractor works correctly, but it needs an evolving system to discover laws.**

### Static System = Only Invariants
If your system doesn't change:
- All quantities remain constant
- Only invariants are detected
- Relationships are spurious (fitting lines to constant data)

### Evolving System = Real Laws
To discover real laws, your system must:
- Apply rotations (change geometry)
- Update basins (change SW, curvature, tension)
- Apply dynamic forces (change energy)
- Run recursion (change structure)
- Process information (change state)

The example script (`example_law_extraction.py`) shows how to evolve the system.

## Advanced Features (v2-v6)

All advanced features are now implemented in `AdvancedLawExtractor`:

### v2: Nonlinear Function Discovery
- Polynomial relationships (y = a*x² + b*x + c)
- Power laws (y = a * x^b)
- Exponential relationships (y = a * exp(b*x))
- Logarithmic relationships (y = a * log(x) + b)

### v3: Symbolic Regression
- Basic symbolic expression discovery
- Common mathematical forms (linear, quadratic, inverse, sqrt)
- Automatic formula generation

### v4: Law Stability + Confidence Scoring
- Confidence scores based on fitting error
- Stability tracking over time
- Law persistence tracking
- Only high-confidence laws are reported

### v5: Multi-Layer Law Fusion
- Combine laws from different recursion depths
- Cross-layer law validation
- Fused laws have higher confidence

### v6: Basin-Based Law Extraction
- Extract laws specific to individual basins
- Basin-specific relationships
- Competition-aware law discovery

## Usage: Advanced Extractor

```python
from core.law.advanced_law_extractor import AdvancedLawExtractor
from core.runtime.orchestrator import Orchestrator

# Create orchestrator
orchestrator = Orchestrator(system)

# Replace with advanced extractor
orchestrator.law_extractor = AdvancedLawExtractor(
    min_confidence=0.6,
    stability_window=20
)

# Run system...
for _ in range(100):
    orchestrator.step()

# Extract all laws
all_laws = orchestrator.law_extractor.extract_all()

# Access different types:
# - all_laws["invariants"]
# - all_laws["linear_relationships"]
# - all_laws["nonlinear_relationships"]
# - all_laws["symbolic_expressions"]
# - all_laws["discovered_laws"]  # With confidence/stability
# - all_laws["fused_laws"]  # Multi-layer
# - all_laws["basin_laws"]  # Basin-specific
```

## Example Scripts

- **Basic**: `example_law_extraction.py` - Simple v1 features
- **Advanced**: `example_advanced_law_extraction.py` - All v2-v6 features
