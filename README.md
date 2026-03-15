
THIS IS OLD VERSION, THIS IS MY PURE THINKING MESS!

TO GET SAFE CLEAN UNDERSTANDING WHAT I ACHIVED IS

YOU WILL FIND IT HERE:

https://github.com/chetanxpatil/livnium












# LIVNIUM

**Law-Governed Geometric Computing Platform**

LIVNIUM is a system that replaces "searching for answers" with "removing impossible futures until only one path can fall."

Like recent energy-based / test-time optimization work, Livnium treats inference as iterative dynamics but it enforces a strict immutable ‘kernel of laws’ with compliance gates, so domains can’t smuggle heuristics into physics.

> **⚠️ Stability Notice**: This version (v1.0.0) is **not stable** and is intended for research purposes only. The API may change, and there may be bugs or incomplete features. Use at your own risk. Production use is not recommended at this time.

## What is LIVNIUM?

LIVNIUM is a geometric computing platform built on immutable laws. Instead of training neural networks to approximate functions, LIVNIUM uses geometric constraints and physics-based collapse to eliminate impossible states until only valid solutions remain.

### Core Philosophy

- **Kernel (LUGK)**: Immutable laws and invariants — pure mathematics, no dependencies
- **Engine (LUGE)**: Runtime dynamics that enforce the laws
- **Domains**: Plugins that encode domain-specific problems into geometric space

The kernel is locked — it represents the constitutional laws of the system. The engine implements those laws. Domains use the laws but cannot modify them.

## Quick Start

```python
import torch
from livnium.engine.collapse.engine import CollapseEngine
from livnium.domains.toy.encoder import ToyEncoder
from livnium.domains.toy.head import ToyHead

# Create components
encoder = ToyEncoder(dim=64)
collapse_engine = CollapseEngine(dim=64, num_layers=3)
head = ToyHead(dim=64, num_classes=3)

# Encode input
x_a = torch.randn(2)
x_b = torch.randn(2)
h0, v_a, v_b = encoder.build_initial_state(x_a, x_b)

# Collapse using kernel physics
h_final, trace = collapse_engine.collapse(h0)

# Get output
logits = head(h_final, v_a, v_b)
```

## Installation

```bash
pip install -r requirements.txt
```

See `livnium/QUICKSTART.md` for detailed setup and examples.

## Architecture

```
livnium/
  kernel/          # Immutable laws (pure math, no torch/numpy)
  engine/          # Runtime dynamics (collapse, basins)
  domains/         # Problem encoders (SNLI, market, ramsey, toy, mindmap)
  training/        # Training infrastructure
  datasets/        # Data loaders
  instrumentation/ # Logging, metrics, profiling
  examples/        # Training scripts
  integration/     # Constraint query API + document pipeline
scripts/           # Shell scripts for running examples and tests
```

## Documentation

- **[Architecture Guide](livnium/ARCHITECTURE.md)** - System design and boundaries
- **[Quick Start](livnium/QUICKSTART.md)** - Getting started with examples
- **[Full Documentation](livnium/README.md)** - Complete API reference

## Available Domains

- **Toy** - Minimal test domain for kernel+engine integration
- **SNLI** - Stanford Natural Language Inference
- **Market** - Financial time series regime classification
- **Ramsey** - Graph coloring constraint satisfaction
- **Mindmap** - Text document analysis and thought graph visualization

## Key Features

- **Law-preserving**: Kernel laws are immutable and verified
- **Compositional**: Domains plug in without modifying core
- **Geometric**: Problems encoded as geometric constraints
- **Physics-based**: Uses alignment, divergence, and tension laws

## License

LIVNIUM is licensed under **LIVNIUM License v1.0** — a law-preserving research license. See [LICENSE](LICENSE) for details.

**Key points:**
- ✅ Free for research and education
- ✅ Open research derivatives allowed
- ❌ Commercial use requires separate license
- 🔒 Kernel integrity must be preserved

## Research

LIVNIUM has been applied to:
- Natural language inference (SNLI)
- Financial market regime detection
- Constraint satisfaction (Ramsey numbers)
- Rule 30 chaos analysis (see `archives-public/rule30/`)

## Contributing

LIVNIUM follows strict architectural boundaries:
- Kernel is locked — changes require justification
- Engine implements kernel laws
- Domains use but don't modify laws

See `livnium/ARCHITECTURE.md` for the complete design philosophy.

## Citation

If you use LIVNIUM in research:

```
LIVNIUM: Law-Governed Geometric Computing Platform
Chetan Patil, 2025
https://github.com/chetanxpatil/livnium.core
```

---

**LIVNIUM** — Removing impossible futures until only one path can fall.
