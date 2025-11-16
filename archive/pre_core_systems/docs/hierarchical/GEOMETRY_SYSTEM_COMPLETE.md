# Hierarchical Geometry System: Complete Code Documentation

## Overview

The **Hierarchical Geometry System** implements a "geometry-in-geometry" architecture where geometric operations operate on geometric structures at multiple levels. This creates a scalable, hierarchical system for quantum-inspired computation.

**Core Principle:** `Geometry > Geometry > Geometry > ...`

---

## Architecture Overview

The system is organized into **levels**:

- **Level 0**: Base Geometry (fundamental states)
- **Level 1**: Geometry in Geometry (operations on base)
- **Level 2**: Geometry in Geometry in Geometry (operations on operations)
- **Level N**: Dynamic hierarchy (unlimited levels)

Additionally, there are **specialized systems**:
- **Dual Cube System**: Positive/negative semantic spaces
- **Trapped Phi System**: Frozen contradictions as structural memory
- **Hierarchy V2**: Advanced features (registry, propagation, level graph)

---

## Level 0: Base Geometry

### `base_geometry.py`

**Purpose:** Foundation layer - stores quantum states in geometric space.

#### `BaseGeometricState`
```python
@dataclass
class BaseGeometricState:
    coordinates: Tuple[float, ...]  # Geometric position
    amplitude: complex              # Quantum amplitude
    phase: float                    # Phase angle
```

**Key Methods:**
- `get_geometric_distance(other)`: Compute distance between states
- `rotate(angle, axis)`: Rotate state in geometric space

**What it does:**
- Represents a single quantum state at a geometric position
- Stores amplitude and phase information
- Provides basic geometric operations

#### `BaseGeometry`
```python
class BaseGeometry:
    def __init__(self, dimension: int = 3)
    def add_state(coordinates, amplitude, phase) -> BaseGeometricState
    def get_geometry_structure() -> Dict
```

**What it does:**
- Container for multiple `BaseGeometricState` objects
- Manages states in a geometric space of specified dimension
- Provides structure information for introspection

**Usage:**
```python
base = BaseGeometry(dimension=3)
state = base.add_state((0.5, 0.3, 0.7), amplitude=1.0+0j, phase=0.0)
```

---

### `sparse_base_geometry.py`

**Purpose:** Optimized version that only stores non-zero states.

#### `SparseBaseGeometricState`
- Same as `BaseGeometricState` but optimized for sparse storage

#### `SparseBaseGeometry`
```python
class SparseBaseGeometry:
    def __init__(self, dimension: int = 3, threshold: float = 1e-15)
    def add_state(coordinates, amplitude, phase) -> Optional[SparseBaseGeometricState]
    def get_state(coordinates) -> Optional[SparseBaseGeometricState]
    def get_amplitude(coordinates) -> complex
    def set_amplitude(coordinates, amplitude)
```

**Key Features:**
- **Sparse storage**: Only stores states with amplitude > threshold
- **Dictionary-based**: `states: Dict[coordinates -> state]`
- **Active tracking**: `active_coordinates: Set[coordinates]` for fast iteration
- **Automatic cleanup**: Removes states below threshold

**What it does:**
- Efficiently stores only significant quantum states
- Automatically manages sparse representation
- Provides fast lookup and iteration

**Usage:**
```python
sparse = SparseBaseGeometry(dimension=3, threshold=1e-10)
state = sparse.add_state((0.5, 0.3, 0.7), amplitude=0.5+0j)
# If amplitude < threshold, state is not stored (returns None)
```

---

## Level 1: Geometry in Geometry

### `geometry_in_geometry.py`

**Purpose:** Operations that operate ON the base geometry.

#### `MetaGeometricOperation`
```python
@dataclass
class MetaGeometricOperation:
    operation_type: str      # 'rotation', 'scaling', 'translation'
    parameters: Dict         # Operation parameters
    target_geometry: BaseGeometry
```

**Key Methods:**
- `apply() -> BaseGeometry`: Apply operation to base geometry
- `_transform_coordinates(coords)`: Transform coordinates based on operation type

**Supported Operations:**
- `rotation`: Rotate coordinates around an axis
- `scaling`: Scale coordinates by a factor
- `translation`: Translate coordinates by an offset

**What it does:**
- Defines a geometric operation that transforms the base geometry
- Applies transformations to all states in the base geometry
- Returns a new transformed geometry

#### `GeometryInGeometry`
```python
class GeometryInGeometry:
    def __init__(self, base_geometry: BaseGeometry)
    def add_meta_operation(operation_type, **parameters) -> MetaGeometricOperation
    def apply_all_operations() -> BaseGeometry
    def get_meta_structure() -> Dict
```

**What it does:**
- Wraps a base geometry and applies meta-operations to it
- Maintains a list of operations to apply
- Can apply all operations sequentially

**Usage:**
```python
base = BaseGeometry(dimension=3)
base.add_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)

geo_in_geo = GeometryInGeometry(base)
geo_in_geo.add_meta_operation('rotation', angle=0.5, axis=0)
geo_in_geo.add_meta_operation('scaling', scale=1.2)

transformed = geo_in_geo.apply_all_operations()
```

---

### `sparse_geometry_in_geometry.py`

**Purpose:** Efficient version that operates on sparse base geometry.

#### `EfficientMetaGeometricOperation`
- Similar to `MetaGeometricOperation` but optimized for sparse storage
- Only processes active (non-zero) states
- Automatically handles sparse threshold

#### `SparseGeometryInGeometry`
- Wraps `SparseBaseGeometry` instead of `BaseGeometry`
- Operations only process active coordinates
- More efficient for large systems with many zero states

---

## Level 2: Geometry in Geometry in Geometry

### `geometry_in_geometry_in_geometry.py`

**Purpose:** Operations that operate ON geometry-in-geometry.

#### `MetaMetaGeometricOperation`
```python
@dataclass
class MetaMetaGeometricOperation:
    operation_type: str
    parameters: Dict
    target_geometry_in_geometry: GeometryInGeometry
```

**What it does:**
- Operates on the meta-operations themselves
- Can transform operation parameters
- Can compose multiple operations

#### `GeometryInGeometryInGeometry`
```python
class GeometryInGeometryInGeometry:
    def __init__(self, geometry_in_geometry: GeometryInGeometry)
    def add_meta_meta_operation(operation_type, **parameters)
    def apply_all_operations() -> GeometryInGeometry
```

**What it does:**
- Wraps geometry-in-geometry and applies meta-meta operations
- Operates at the highest level of the 3-level hierarchy

#### `HierarchicalGeometrySystem`
```python
class HierarchicalGeometrySystem:
    def __init__(self, base_dimension: int = 3)
    def add_base_state(coordinates, amplitude, phase)
    def add_meta_operation(operation_type, **parameters)
    def add_meta_meta_operation(operation_type, **parameters)
    def get_full_structure() -> Dict
```

**What it does:**
- Complete 3-level hierarchical system
- Manages all three levels together
- Provides unified interface

**Usage:**
```python
system = HierarchicalGeometrySystem(base_dimension=3)
system.add_base_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)
system.add_meta_operation('rotation', angle=0.5, axis=0)
system.add_meta_meta_operation('scale_operations', scale=1.1)
```

---

### `sparse_hierarchical_geometry.py`

**Purpose:** Complete hierarchical system with sparse optimization at all levels.

- Uses `SparseBaseGeometry` at Level 0
- Uses `SparseGeometryInGeometry` at Level 1
- Optimized operations at Level 2
- Most efficient for large-scale systems

---

### `projection_hierarchical_geometry.py`

**Purpose:** Uses hierarchy to PROJECT high-entanglement states into manageable representations.

#### `ProjectionOperation`
```python
@dataclass
class ProjectionOperation:
    projection_type: str  # 'entanglement_compression', 'local_projection', etc.
    parameters: Dict
    target_geometry_in_geometry: SparseGeometryInGeometry
```

**Projection Types:**
- `entanglement_compression`: Compress entanglement by projecting onto hierarchy
- `local_projection`: Project local correlations to Level 0
- `hierarchical_decomposition`: Decompose into hierarchical components

**What it does:**
- Instead of storing full entanglement, projects it into hierarchy levels
- Level 0: Local correlations (sparse)
- Level 1: Medium-range correlations
- Level 2: Long-range correlations (compressed)

---

## Dynamic Hierarchical Geometry

### `dynamic_hierarchical_geometry.py`

**Purpose:** Allows N-level geometry stacking (not just 3 levels).

#### `HierarchicalOperation`
```python
@dataclass
class HierarchicalOperation:
    level: int
    operation_type: str  # 'rotation', 'scale', 'entangle', 'transform'
    parameters: Dict[str, Any]
```

**Operation Types:**
- `rotation`: Rotate geometry
- `scale`: Scale geometry
- `entangle`: Apply entanglement (stub - not fully implemented)
- `transform`: General transformation

#### `HierarchicalLevel`
```python
class HierarchicalLevel:
    def __init__(self, level: int, base_geometry: Any)
    def add_operation(operation_type, **parameters) -> HierarchicalOperation
    def apply_all_operations() -> Any
```

**What it does:**
- Represents a single level in the hierarchy
- Wraps the level below it
- Can have multiple operations

#### `DynamicHierarchicalGeometrySystem`
```python
class DynamicHierarchicalGeometrySystem:
    def __init__(self, base_dimension: int = 3, num_levels: int = 3)
    def add_base_state(coordinates, amplitude, phase)
    def add_operation(level: int, operation_type, **parameters)
    def add_meta_operation(operation_type, **parameters)  # Level 1
    def add_meta_meta_operation(operation_type, **parameters)  # Level 2
    def get_full_structure() -> Dict
```

**Key Features:**
- **N-level support**: Can have 1, 2, 3, 4, 5, 6... levels
- **Dynamic building**: Levels are built recursively
- **Unified interface**: Same API regardless of number of levels

**Usage:**
```python
# 3 levels (like fixed system)
system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=3)

# 5 levels
system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=5)

# Add operations at any level
system.add_operation(level=1, operation_type='rotation', angle=0.5, axis=0)
system.add_operation(level=3, operation_type='scale', scale=1.2)
```

**What it does:**
- Builds hierarchy dynamically: Level 1 wraps Level 0, Level 2 wraps Level 1, etc.
- Each level can have multiple operations
- Operations are applied from bottom to top

---

## Hierarchy V2: Advanced Features

### `hierarchy_v2.py`

**Purpose:** Advanced hierarchical system with registry, propagation, and visualization.

#### `OperationType` (Enum)
```python
class OperationType(Enum):
    ROTATION = "rotation"
    SCALE = "scale"
    TRANSLATION = "translation"
    ENTANGLE = "entangle"
    TRANSFORM = "transform"
    COMPOSE = "compose"
    PROJECT = "project"
    FILTER = "filter"
    AGGREGATE = "aggregate"
```

#### `RegisteredOperation`
```python
@dataclass
class RegisteredOperation:
    operation_id: str
    operation_type: OperationType
    level: int
    parameters: Dict[str, Any]
    description: str
    propagates_down: bool = True
    timestamp: float
```

**What it does:**
- Fully documented operation with metadata
- Trackable and auditable
- Can propagate down through levels

#### `OperationRegistry`
```python
class OperationRegistry:
    def register(operation_type, level, parameters, description, propagates_down) -> RegisteredOperation
    def get_operation(operation_id) -> Optional[RegisteredOperation]
    def get_operations_at_level(level) -> List[RegisteredOperation]
    def get_operations_by_type(operation_type) -> List[RegisteredOperation]
    def get_registry_summary() -> Dict
```

**What it does:**
- Maintains registry of all operations
- Tracks operations by level and type
- Provides audit trail

#### `PropagationEngine`
```python
class PropagationEngine:
    def propagate_operation(operation: RegisteredOperation) -> Dict
    def get_propagation_graph() -> Dict
```

**What it does:**
- Propagates operations down through hierarchy levels
- Level N → Level N-1 → ... → Level 1 → Level 0
- Tracks propagation history

**Propagation Flow:**
```
Operation at Level 5
  ↓ propagates to
Level 4
  ↓ propagates to
Level 3
  ↓ propagates to
Level 2
  ↓ propagates to
Level 1
  ↓ propagates to
Level 0 (base)
```

#### `LevelGraph`
```python
class LevelGraph:
    def build_graph() -> LevelNode
    def visualize(format: str = 'tree') -> str
```

**What it does:**
- Builds visual map of hierarchy structure
- Shows levels, operations, and states
- Supports text, tree, and JSON formats

**Visualization Example:**
```
Level 0: base (0 ops, 100 states)
  └── Level 1: meta_1 (5 ops, 0 states)
      └── Level 2: meta_2 (3 ops, 0 states)
```

#### `HierarchyV2System`
```python
class HierarchyV2System:
    def __init__(self, base_dimension: int = 3, num_levels: int = 3)
    def add_base_state(coordinates, amplitude, phase)
    def register_operation(operation_type, level, parameters, description, propagates_down) -> RegisteredOperation
    def get_level_graph(format: str = 'tree') -> str
    def get_operation_registry() -> Dict
    def get_propagation_history() -> Dict
    def get_full_system_info() -> Dict
```

**Key Features:**
- **Operation Registry**: All operations are registered and documented
- **Propagation Engine**: Operations propagate down through levels
- **Level Graph**: Visual representation of hierarchy
- **Full Audit Trail**: Complete history of all operations

**Usage:**
```python
system = HierarchyV2System(base_dimension=3, num_levels=5)

# Add base state
system.add_base_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)

# Register operation (automatically propagates)
op = system.register_operation(
    operation_type=OperationType.ROTATION,
    level=3,
    parameters={'angle': 0.5, 'axis': 0},
    description='Rotation at level 3',
    propagates_down=True
)

# Visualize hierarchy
print(system.get_level_graph(format='tree'))

# Get registry summary
registry = system.get_operation_registry()
print(f"Total operations: {registry['total_operations']}")
```

---

## Dual Cube System

### `dual_cube_system.py`

**Purpose:** Implements positive and negative semantic spaces.

#### `DualState`
```python
@dataclass
class DualState:
    positive_state: Optional[BaseGeometricState]
    negative_state: Optional[BaseGeometricState]
    energy: float
    contradiction_score: float
```

**What it does:**
- Represents a state that exists in both positive and negative cubes
- Tracks energy and contradiction score

#### `DualCubeSystem`
```python
class DualCubeSystem:
    def __init__(self, base_dimension: int = 3, num_levels: int = 3)
    def add_positive_state(coordinates, amplitude, phase) -> BaseGeometricState
    def add_negative_state(coordinates, amplitude, phase) -> BaseGeometricState
    def add_dual_state(coordinates, positive_amplitude, negative_amplitude) -> DualState
    def detect_contradiction(state, context) -> float
    def move_to_negative_cube(state, contradiction_score) -> Optional[BaseGeometricState]
    def apply_decoherence_drift(rate: float = None)
    def apply_cancellation()
    def get_decoherence_measure() -> Dict
    def diagnose_confusion(coordinates) -> Dict
```

**Three-State Semantic Universe:**
- **Cube⁺ (+1)**: Positive cube - stable meanings, attractors, verified patterns
- **Cube⁻ (-1)**: Negative cube - contradictions, conflicts, decohered states
- **Dual States**: States that exist in both cubes

**Key Concepts:**
- **Contradiction Detection**: Detects when states conflict
- **Decoherence Drift**: Amplitude migrates from +cube → −cube over time
- **Cancellation**: If pattern appears in both cubes with opposite sign, they cancel
- **Confusion Diagnosis**: Checks if coordinates live mostly in −cube

**What it does:**
- Maintains two separate geometric spaces (positive and negative)
- Tracks contradictions and decoherence
- Provides semantic analysis of states

**Usage:**
```python
dual = DualCubeSystem(base_dimension=3, num_levels=3)

# Add positive state (stable meaning)
dual.add_positive_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)

# Add negative state (contradiction)
dual.add_negative_state((0.5, 0.3, 0.7), 0.5+0j, np.pi)

# Add dual state (exists in both)
dual.add_dual_state((0.2, 0.4, 0.6), positive_amplitude=0.8+0j, negative_amplitude=0.3+0j)

# Measure decoherence
decoherence = dual.get_decoherence_measure()
print(f"Decoherence fraction: {decoherence['decoherence_fraction']}")

# Diagnose confusion
diagnosis = dual.diagnose_confusion((0.5, 0.3, 0.7))
print(f"Confusion score: {diagnosis['confusion_score']}")
```

---

## Trapped Phi System

### `trapped_phi.py`

**Purpose:** Extends dual cube with trapped/frozen contradictions as structural memory.

#### `ExtendedGeometricState`
```python
@dataclass
class ExtendedGeometricState(BaseGeometricState):
    state_type: int = +1  # +1 (Cube⁺), -1 (Cube⁻), or 0 (Cube⁰)
    phi_energy: float = 0.0
    age: int = 0
    contradiction_score: float = 0.0
    decoherence_score: float = 0.0
    active_age: int = 0
```

**State Types:**
- **+1 (Cube⁺)**: Active meaning (fluid)
- **-1 (Cube⁻)**: Active contradiction (fluid)
- **0 (Cube⁰)**: Trapped/frozen φ (solid structure)

#### `TrappedPhiSystem`
```python
class TrappedPhiSystem:
    def __init__(self, contra_trap_threshold=0.7, deco_trap_threshold=0.6, 
                 min_active_age=10, trap_half_life=100, leak_prob=0.01,
                 max_trapped_fraction=0.25)
    def should_trap(state: ExtendedGeometricState) -> bool
    def trap_state(state: ExtendedGeometricState) -> ExtendedGeometricState
    def apply_decay(states: List[ExtendedGeometricState]) -> int
    def get_trapped_fraction(states) -> float
    def enforce_capacity_limit(states) -> Dict
    def update_active_age(states)
    def analyze_scars(states, cluster_threshold=0.1) -> Dict
```

**Trapping Rules:**
- State must have high contradiction score (> threshold)
- State must have high decoherence score (> threshold)
- State must be active for minimum age (10 ticks)
- If all conditions met → trap to Cube⁰

**Decay Mechanism:**
- Trapped states have a half-life (100 ticks)
- After half-life, random decay probability (1% per tick)
- Decayed states leak back to Cube⁻

**Capacity Limits:**
- Maximum 25% of states can be trapped
- If exceeded: raise thresholds dynamically, force decay of oldest

**Scar Analysis:**
- Analyzes clusters of trapped states
- Identifies "scars" (persistent structural obstacles)

**What it does:**
- Manages three-state universe: Cube⁺, Cube⁻, Cube⁰
- Traps highly contradictory states as structural memory
- Prevents universe from "turning to stone" with capacity limits
- Analyzes scar patterns (clusters of trapped states)

---

## Dual Cube with Trapped System

### `dual_cube_with_trapped.py`

**Purpose:** Combines dual cube system with trapped phi support.

#### `DualCubeWithTrappedSystem`
```python
class DualCubeWithTrappedSystem(DualCubeSystem):
    def __init__(self, base_dimension: int = 3, num_levels: int = 3, **trapped_params)
    def add_positive_state(coordinates, amplitude, phase) -> ExtendedGeometricState
    def add_negative_state(coordinates, amplitude, phase) -> ExtendedGeometricState
    def update_scores()
    def apply_trapping() -> int
    def apply_decay() -> int
    def enforce_capacity_limits() -> Dict
    def step()
    def get_trapped_statistics() -> Dict
```

**Evolution Step:**
1. Update contradiction/decoherence scores
2. Apply trapping (check and trap states)
3. Apply decay (leak trapped states back)
4. Enforce capacity limits

**What it does:**
- Extends `DualCubeSystem` with trapped phi functionality
- Maintains three-state universe: Cube⁺, Cube⁻, Cube⁰
- Provides unified interface for all three states
- Tracks statistics about trapped states

**Usage:**
```python
system = DualCubeWithTrappedSystem(base_dimension=3, num_levels=3)

# Add states
system.add_positive_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)
system.add_negative_state((0.5, 0.3, 0.7), 0.5+0j, np.pi)

# Evolve system
result = system.step()
print(f"Trapped: {result['trapped']}, Decayed: {result['decayed']}")

# Get statistics
stats = system.get_trapped_statistics()
print(f"Trapped fraction: {stats['trapped_fraction']}")
print(f"Number of scars: {stats['scar_analysis']['num_scars']}")
```

---

## System Relationships

```
BaseGeometry (Level 0)
    ↓
GeometryInGeometry (Level 1)
    ↓
GeometryInGeometryInGeometry (Level 2)
    ↓
DynamicHierarchicalGeometrySystem (N levels)
    ↓
HierarchyV2System (N levels + registry + propagation)
```

**Specialized Systems:**
```
DualCubeSystem
    ↓
DualCubeWithTrappedSystem (extends DualCubeSystem)
    ↓
    uses TrappedPhiSystem
```

---

## Key Design Patterns

### 1. **Hierarchical Wrapping**
Each level wraps the level below it:
- Level 1 wraps Level 0
- Level 2 wraps Level 1
- Level N wraps Level N-1

### 2. **Sparse Optimization**
Sparse versions only store non-zero states:
- `SparseBaseGeometry` uses dictionary storage
- Only processes active coordinates
- Automatic threshold management

### 3. **Operation Propagation**
Operations can propagate down through levels:
- Operation at Level N affects all levels below
- Propagation is tracked and auditable

### 4. **Semantic Spaces**
Dual cube system maintains separate semantic spaces:
- Positive cube: stable meanings
- Negative cube: contradictions
- Trapped cube: frozen structure

---

## Memory and Performance

### Base Geometry
- **Memory per state**: ~100-200 bytes
- **Scaling**: O(n) where n = number of states
- **Operations**: O(n) for transformations

### Sparse Base Geometry
- **Memory per state**: ~100-200 bytes (only non-zero)
- **Scaling**: O(k) where k = number of non-zero states
- **Operations**: O(k) for transformations (only active states)

### Hierarchical System
- **Memory**: Same as base (levels don't add memory)
- **Operations**: O(n × L) where L = number of levels
- **Key Insight**: Hierarchy depth does NOT affect capacity

---

## Usage Examples

### Basic 3-Level Hierarchy
```python
from quantum.hierarchical.geometry.level2.geometry_in_geometry_in_geometry import HierarchicalGeometrySystem

system = HierarchicalGeometrySystem(base_dimension=3)
system.add_base_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)
system.add_meta_operation('rotation', angle=0.5, axis=0)
system.add_meta_meta_operation('scale_operations', scale=1.1)
```

### Dynamic N-Level Hierarchy
```python
from quantum.hierarchical.geometry.dynamic_hierarchical_geometry import DynamicHierarchicalGeometrySystem

system = DynamicHierarchicalGeometrySystem(base_dimension=3, num_levels=5)
system.add_base_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)
system.add_operation(level=1, operation_type='rotation', angle=0.5, axis=0)
system.add_operation(level=3, operation_type='scale', scale=1.2)
```

### Hierarchy V2 with Registry
```python
from quantum.hierarchical.geometry.hierarchy_v2 import HierarchyV2System, OperationType

system = HierarchyV2System(base_dimension=3, num_levels=5)
system.add_base_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)

op = system.register_operation(
    operation_type=OperationType.ROTATION,
    level=3,
    parameters={'angle': 0.5, 'axis': 0},
    description='Rotation at level 3',
    propagates_down=True
)

print(system.get_level_graph(format='tree'))
```

### Dual Cube System
```python
from quantum.hierarchical.geometry.dual_cube_system import DualCubeSystem

dual = DualCubeSystem(base_dimension=3, num_levels=3)
dual.add_positive_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)
dual.add_negative_state((0.5, 0.3, 0.7), 0.5+0j, np.pi)

decoherence = dual.get_decoherence_measure()
diagnosis = dual.diagnose_confusion((0.5, 0.3, 0.7))
```

### Dual Cube with Trapped Phi
```python
from quantum.hierarchical.geometry.dual_cube_with_trapped import DualCubeWithTrappedSystem

system = DualCubeWithTrappedSystem(base_dimension=3, num_levels=3)
system.add_positive_state((0.5, 0.3, 0.7), 1.0+0j, 0.0)

result = system.step()
stats = system.get_trapped_statistics()
```

---

## Summary

The Hierarchical Geometry System provides:

1. **Base Geometry (Level 0)**: Fundamental state storage
2. **Geometry in Geometry (Level 1)**: Operations on base
3. **Geometry in Geometry in Geometry (Level 2)**: Operations on operations
4. **Dynamic Hierarchy**: N-level support
5. **Hierarchy V2**: Advanced features (registry, propagation, visualization)
6. **Dual Cube System**: Positive/negative semantic spaces
7. **Trapped Phi System**: Frozen contradictions as structural memory

**Key Insight**: The hierarchy depth does NOT affect qubit capacity - capacity is memory-limited, not hierarchy-limited. This allows unlimited reasoning depth without explosion.

---

*Complete documentation of the Hierarchical Geometry System codebase.*

