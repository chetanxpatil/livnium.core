# Core Structure: Complete Layer-by-Layer Guide

## Overview

The Livnium Core System is organized into **8 layers** (0-7), each building on the previous. Each layer is a separate folder in `core/`.

---

## Layer 0: Recursive Geometry Engine
**Location**: `core/recursive/`

**Purpose**: The structural foundation - makes everything scalable through recursive geometry.

**Components**:
- `recursive_geometry_engine.py` - Main recursive engine
- `geometry_subdivision.py` - Subdivision rules (N×N×N → M×M×M)
- `recursive_projection.py` - State projection across scales
- `recursive_conservation.py` - Invariant preservation
- `moksha_engine.py` - Fixed-point convergence (the exit mechanism)

**Key Features**:
- Subdivides geometry into smaller geometry
- Projects high-dimensional states downward
- Conservation recursion (ΣSW preserved per scale)
- Recursive entanglement (compressed into lower scale)
- Recursive observer (macro → micro)
- Recursive motion (rotation at macro → rotation in micro)
- Recursive problem solving (search across layers)
- **Moksha**: Fixed-point convergence and release from recursion

**Capacity**: Exponential with linear memory
- 5×5×5 base with 2 levels = **94,625 cells**

---

## Layer 1: Classical Layer
**Location**: `core/classical/`

**Purpose**: The base geometric lattice system with invariants.

**Components**:
- `livnium_core_system.py` - Main system (LivniumCoreSystem, LatticeCell, Observer, RotationGroup)

**Key Features**:
- N×N×N lattice (any odd N ≥ 3)
- Symbolic Weight (SW = 9·f)
- Face exposure classification
- 90° rotation group (24 elements)
- Observer system (Global/Local)
- Semantic polarity (cos(θ))
- Invariants conservation (ΣSW, class counts)

**Exports**:
- `LivniumCoreSystem` - Main system class
- `LatticeCell` - Cell representation
- `Observer` - Global/Local observers
- `RotationAxis`, `CellClass`, `RotationGroup`

---

## Layer 2: Quantum Layer
**Location**: `core/quantum/`

**Purpose**: Quantum states, gates, entanglement, and measurement.

**Components**:
- `quantum_cell.py` - Quantum state per cell
- `quantum_gates.py` - Unitary gate library (H, X, Y, Z, rotations, CNOT)
- `quantum_lattice.py` - Quantum-geometry integration
- `entanglement_manager.py` - Multi-cell entanglement
- `measurement_engine.py` - Born rule + collapse
- `geometry_quantum_coupling.py` - Geometry ↔ Quantum mapping

**Key Features**:
- Superposition (complex amplitudes)
- Quantum gates (full unitary library)
- Entanglement (Bell states, geometric)
- Measurement (Born rule + collapse)
- Geometry-Quantum coupling (face exposure → entanglement)

**Exports**:
- `QuantumCell`, `QuantumGates`, `GateType`
- `QuantumLattice`, `EntanglementManager`, `EntangledPair`
- `MeasurementEngine`, `MeasurementResult`
- `GeometryQuantumCoupling`

---

## Layer 3: Memory Layer
**Location**: `core/memory/`

**Purpose**: Working memory and long-term memory.

**Components**:
- `memory_cell.py` - Per-cell memory capsules (MemoryCell, MemoryState)
- `memory_lattice.py` - Global memory lattice
- `memory_coupling.py` - Memory coupling mechanisms

**Key Features**:
- Per-cell memory capsules
- Working memory (short-term)
- Long-term memory (persistent)
- Memory coupling (cross-cell memory links)
- Cross-step recursive updates

**Exports**:
- `MemoryCell`, `MemoryState`
- `MemoryLattice`, `MemoryCoupling`

---

## Layer 4: Reasoning Layer
**Location**: `core/reasoning/`

**Purpose**: Search, tree expansion, rules, and problem solving.

**Components**:
- `search_engine.py` - Search algorithms (BFS, DFS, A*, Beam, Greedy)
- `rule_engine.py` - Rule-based reasoning (Rule, RuleSet)
- `reasoning_engine.py` - High-level reasoning orchestration
- `problem_solver.py` - Problem-solving interface (ProblemSolver)

**Key Features**:
- Search strategies (BFS, DFS, A*, Beam, Greedy)
- Tree expansion
- Rule-based reasoning
- Symbolic reasoning
- Problem-solving loop

**Exports**:
- `SearchEngine`, `SearchNode`, `SearchStrategy`
- `RuleEngine`, `Rule`, `RuleSet`
- `ReasoningEngine`, `ProblemSolver`

---

## Layer 5: Semantic Layer
**Location**: `core/semantic/`

**Purpose**: Meaning, language, and inference.

**Components**:
- `semantic_processor.py` - Main semantic processor
- `feature_extractor.py` - Feature extraction
- `meaning_graph.py` - Symbol-to-meaning graph (MeaningGraph, SemanticNode)
- `inference_engine.py` - Inference engine

**Key Features**:
- Feature extraction
- Semantic embeddings
- Symbol-to-meaning graph
- Negation detection
- Context propagation
- Entailment/contradiction mechanics
- Causal link detection

**Exports**:
- `SemanticProcessor`, `FeatureExtractor`
- `MeaningGraph`, `SemanticNode`
- `InferenceEngine`

---

## Layer 6: Meta Layer
**Location**: `core/meta/`

**Purpose**: Self-reflection, calibration, and introspection.

**Components**:
- `meta_observer.py` - MetaObserver (self-reflection)
- `anomaly_detector.py` - Anomaly detection
- `calibration_engine.py` - Adaptive calibration
- `introspection.py` - Introspection engine

**Key Features**:
- Reflection
- Introspection
- Reasoning about own states
- Anomaly detection
- Self-alignment
- Invariance drift detection
- Adaptive calibration

**Exports**:
- `MetaObserver`, `AnomalyDetector`
- `CalibrationEngine`, `IntrospectionEngine`

---

## Layer 7: Runtime Layer
**Location**: `core/runtime/`

**Purpose**: Orchestration, episodes, and temporal management.

**Components**:
- `temporal_engine.py` - Temporal engine (Timestep management)
- `orchestrator.py` - Orchestrator (cross-layer coordination)
- `episode_manager.py` - Episode management

**Key Features**:
- Timestep engine
- Scheduling
- Macro/micro update rhythm
- Propagation order
- Stabilization rules
- Cross-layer arbitration
- Episode management

**Exports**:
- `TemporalEngine`, `Timestep`
- `Orchestrator`, `EpisodeManager`

---

## Search Module
**Location**: `core/search/`

**Purpose**: Dynamic basin reinforcement and multi-basin search.

**Components**:
- `native_dynamic_basin_search.py` - Dynamic basin reinforcement
- `multi_basin_search.py` - Multi-basin competition
- `test_native_dynamic_basin.py` - Tests
- `test_multi_basin.py` - Tests
- `HOW_IT_WORKS.md` - Dynamic basin documentation
- `MULTI_BASIN_SEARCH.md` - Multi-basin documentation

**Key Features**:
- Geometry-driven parameters (curvature, tension, entropy)
- Self-tuning basin shaping
- Multiple competing attractors
- Natural selection through geometry

## Universal Encoder Module (New)
**Location**: `core/Universal Encoder/`

**Purpose**: Convert any problem into geometric patterns (SW structures).

**Status**: 🚧 In Development

**Planned Components**:
- `problem_encoder.py` - Main universal encoder interface
- `constraint_encoder.py` - Constraint encoding
- `graph_encoder.py` - Graph encoding
- `logic_encoder.py` - Logic encoding
- `language_encoder.py` - Natural language encoding

**Key Features**:
- Universal encoding for any problem type
- Standardized interface
- Feature → Coordinate mapping
- Constraint → Basin shape mapping
- Dependency → Coupling pattern mapping

---

## Configuration
**Location**: `core/config.py`

**Purpose**: Central configuration with feature switches.

**Key Features**:
- `LivniumCoreConfig` - Configuration class
- Feature switches for all layers
- Lattice size configuration
- Quantum feature toggles

---

## Tests
**Location**: `core/tests/`

**Purpose**: Test suite for all layers.

**Components**:
- `test_livnium_core.py` - Classical system tests
- `test_generalized_n.py` - Generalized N×N×N tests
- `test_quantum.py` - Quantum layer tests
- `test_entanglement_capacity.py` - Entanglement capacity tests
- `test_qubit_capacity.py` - Qubit capacity tests

---

## Documentation Files

**Core Documentation**:
- `README.md` - Main overview
- `ARCHITECTURE.md` - 8-layer architecture details
- `CORE_STRUCTURE.md` - Layer-by-layer structure guide (this file)
- `QUANTUM_LAYER.md` - Quantum layer details
- `LAYER_0.md` - Recursive geometry engine details
- `MOKSHA.md` - Moksha engine details

**Search & Basin Documentation**:
- `DYNAMIC_BASIN_REINFORCEMENT.md` - Dynamic basin approach
- `BASIN_REINFORCED_FORMULA.md` - Basin-reinforced formula
- `PHI_CYCLE_SEARCH.md` - φ-cycle search
- `ITERATION_ANALYSIS.md` - Iteration analysis

**Ramsey & Constraint Documentation**:
- `CONSTRAINT_GRAPH_PHYSICS.md` - Constraint-graph physics
- `RAMSEY_DYNAMIC_BASIN.md` - Ramsey with dynamic basin
- `RAMSEY_READY_PATCHES.md` - Ramsey-ready patches

**Physics Documentation**:
- `GRADIENT_DRIVEN_PROPAGATION.md` - Gradient-driven propagation
- `MULTI_NODE_TENSION_PROPAGATION.md` - Multi-node tension
- `PHI_ADJUSTMENT_HOOKS.md` - φ adjustment hooks

---

## Layer Dependencies

```
Layer 7 (Runtime)
    ↓
Layer 6 (Meta)
    ↓
Layer 5 (Semantic)
    ↓
Layer 4 (Reasoning)
    ↓
Layer 3 (Memory)
    ↓
Layer 2 (Quantum)
    ↓
Layer 1 (Classical)
    ↓
Layer 0 (Recursive Geometry)
```

**Each layer builds on the previous**, with Layer 0 (Recursive Geometry) as the structural foundation that makes everything scalable.

---

## Usage Pattern

```python
# Layer 1: Classical (base)
from core.classical import LivniumCoreSystem, LivniumCoreConfig

# Layer 2: Quantum (optional)
from core.quantum import QuantumLattice, GateType

# Layer 0: Recursive (for scalability)
from core.recursive import RecursiveGeometryEngine, MokshaEngine

# Layer 4: Reasoning (for problem solving)
from core.reasoning import ProblemSolver

# Search: Dynamic basin (for task solving)
from core.search import update_basin_dynamic
```

---

## Summary

- **Layer 0**: Recursive Geometry (structural foundation, scalability)
- **Layer 1**: Classical (base geometric system)
- **Layer 2**: Quantum (superposition, gates, entanglement)
- **Layer 3**: Memory (working & long-term memory)
- **Layer 4**: Reasoning (search, rules, problem solving)
- **Layer 5**: Semantic (meaning, inference, language)
- **Layer 6**: Meta (self-reflection, calibration)
- **Layer 7**: Runtime (orchestration, episodes)
- **Search**: Dynamic basin reinforcement (geometry-driven search)

Each layer is self-contained but builds on previous layers, creating a complete, scalable system.

