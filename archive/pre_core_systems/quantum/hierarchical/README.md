# Hierarchical Geometry Machine

**This is a quantum-inspired classical system. It is NOT a physical quantum computer.**

## Overview

The Hierarchical Geometry Machine uses a 3-level "geometry-in-geometry" architecture to achieve 5000+ qubit-analogue capacity with linear memory scaling.

## Architecture

The system supports **dynamic hierarchy** with N levels:

- **Level 0**: Base geometry (fundamental structure)
- **Level 1**: Geometry in geometry (meta-operations)
- **Level 2**: Geometry in geometry in geometry (meta-meta-operations)
- **Level 3+**: Additional meta-levels (as needed)

**Default**: 3 levels (Level 0 + 2 meta levels)

**Dynamic**: Can be configured with 1, 2, 3, 4, 5, 6... levels as needed

See `docs/hierarchical/DYNAMIC_HIERARCHY.md` for details on using dynamic levels.

## Key Features

- 3-level hierarchical geometry structure
- Linear memory scaling (~400 bytes per qubit analogue)
- 5000+ qubit-analogue capacity
- Geometric state representation
- **Pure geometry-in-geometry** (not tensor networks)

## Important Distinction

**This system uses geometry-in-geometry architecture, NOT tensor networks.**

- ❌ **NOT** MPS (Matrix Product States) - that's in `livnium_core/`
- ❌ **NOT** tensor networks - that's in `livnium_core/`
- ✅ **IS** geometric coordinate transformations
- ✅ **IS** hierarchical state management
- ✅ **IS** quantum-inspired classical computation

## Structure

- `geometry/` - 3-level geometry hierarchy (level0, level1, level2)
- `core/` - Core processor
- `simulators/` - Various simulators
- `algorithms/` - Quantum-inspired algorithms
- `examples/` - Example usage
- `tests/` - Test files
- `docs/` - Documentation

## Usage

### Standard 3-Level System
```python
from quantum.hierarchical.core.quantum_processor import QuantumProcessor

processor = QuantumProcessor(base_dimension=3)
qubit_id = processor.create_qubit((0.1, 0.2, 0.3))
```

### Dynamic N-Level System
```python
from quantum.hierarchical.geometry.dynamic_hierarchical_geometry import DynamicHierarchicalGeometrySystem

# Create with any number of levels (1, 2, 3, 4, 5, 6...)
system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=5)
state = system.add_base_state((0.1, 0.2, 0.3))
system.add_operation(level=1, operation_type='rotation', angle=0.5)
```

### Hierarchy v2 (Advanced Features)
```python
from quantum.hierarchical.geometry.hierarchy_v2 import HierarchyV2System, OperationType

# Advanced system with level graph, operation registry, and propagation engine
system = HierarchyV2System(base_dimension=3, num_levels=20)

# Register operations (documented, auditable)
system.register_operation(
    OperationType.ROTATION, level=7,
    parameters={'angle': 0.5},
    description='Rotation at level 7',
    propagates_down=True
)

# Visualize level graph
print(system.get_level_graph(format='tree'))

# Get operation registry
registry = system.get_operation_registry()
```

See `docs/hierarchical/HIERARCHY_V2.md` for full documentation.

## Important Note

This system uses quantum-inspired language and geometric structures, but operates entirely on classical hardware. It does not perform actual quantum computation.

