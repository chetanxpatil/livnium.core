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

## Future Enhancements

- **v2**: Nonlinear function discovery
- **v3**: Symbolic regression
- **v4**: Law stability and confidence scoring
- **v5**: Multi-layer law fusion across recursion depths
- **v6**: Basin-based law extraction
