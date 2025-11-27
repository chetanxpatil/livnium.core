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

```
=== Discovered Laws ===

Invariants (Conserved Quantities):
  - SW_sum: 486.000000 (constant)

Functional Relationships:
  - divergence = -1.000000 * alignment + 0.380000
```

## Integration

The law extractor is automatically integrated into the `Orchestrator`:
- Records physics state after each timestep
- Can extract laws at any time
- Provides human-readable summaries

## Future Enhancements

- Nonlinear function discovery
- Symbolic regression
- Law stability and confidence scoring
- Multi-layer law fusion across recursion depths
- Basin-based law extraction
