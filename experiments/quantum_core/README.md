# Quantum-Inspired Livnium Core

This directory contains quantum-related experiments and tests using the Livnium Core System's quantum layer and recursive geometry engine.

## Files

### Tests
- **`test_recursive_qubit_capacity.py`**: Tests how many omcubes can be represented using recursive geometry (currently ~2.5M omcubes)
- **`test_entanglement_capacity.py`**: Tests entanglement capacity - how many omcubes can be entangled simultaneously (pairwise, chains, clusters)

### Experiments
- **`aes128_quantum_topology_mapper.py`**: Quantum-enhanced topology mapper for AES-128 that uses superposition and recursive geometry to explore the key space landscape
- **`aes128_round_sweep_experiment_livnium.py`**: AES-128 round sweep experiment using full Livnium architecture (recursive geometry + quantum layer + tension fields)

### Documentation
- **`OMCUBE_ANALYSIS.md`**: Analysis of omcube types (logical vs physical) in the Livnium implementation

## Key Features

### Recursive Geometry Omcube Capacity
- **2.5 Million Omcubes**: The recursive geometry engine can represent millions of omcubes
- **Perfect Simulation**: Error-free, infinite coherence (simulated logical omcubes)
- **Exponential Capacity**: Linear memory, exponential omcube count

### Entanglement Capacity
- **Pairwise Entanglement**: Bell pairs between omcubes (2.5M omcubes across all levels)
- **Chain Entanglement**: Linear chains of entangled omcubes (2.5M omcubes)
- **Cluster Entanglement**: Fully connected clusters (94K+ omcubes)
- **Recursive Entanglement**: Cross-level entanglement across all recursive levels
- **Total Capacity**: **2.5 Million omcubes can be entangled simultaneously** (not just 125!)

### Quantum-Enhanced Cryptanalysis
- **Superposition Search**: Uses quantum superposition to explore key space
- **Tension Fields**: Encodes constraints as geometric tension
- **Recursive Search**: Subdivides key space across multiple scales

## Usage

### Test Omcube Capacity
```bash
python3 experiments/quantum_core/test_recursive_qubit_capacity.py
```

### Test Entanglement Capacity
```bash
python3 experiments/quantum_core/test_entanglement_capacity.py
```

### Run Quantum Topology Mapper
```bash
python3 experiments/quantum_core/aes128_quantum_topology_mapper.py
```

### Run AES Round Sweep (Quantum)
```bash
python3 experiments/quantum_core/aes128_round_sweep_experiment_livnium.py
```

## Architecture

These experiments use:
- **Layer 0**: Recursive Geometry Engine (subdivides space)
- **Layer 2**: Quantum Layer (superposition, entanglement, measurement)
- **Tension Fields**: Constraint encoding as geometric tension

## Results

### Omcube Capacity
- **Level 0**: 125 omcubes (5×5×5 base)
- **Level 1**: 3,375 omcubes
- **Level 2**: 91,125 omcubes
- **Level 3**: 2,460,375 omcubes
- **Total**: 2,555,000 omcubes

### Entanglement Capacity (Full Recursive)
- **Pairwise**: 2,555,000 entangled omcubes across all recursive levels
  - Level 0: 125 omcubes
  - Level 1: 3,375 omcubes (125 geometries × 27 cells)
  - Level 2: 91,125 omcubes (3,375 geometries × 27 cells)
  - Level 3: 2,460,375 omcubes (91,125 geometries × 27 cells)
- **Chains**: 2,555,000 entangled omcubes (chains across all levels)
- **Clusters**: 94,625+ entangled omcubes (clusters up to depth 2)
- **Cross-level**: 3,375+ potential cross-level entanglements

## Notes

- All omcubes are **simulated logical omcubes** (perfect, error-free)
- Recursive geometry compresses entanglement into lower-scale geometry
- Quantum features require proper configuration in `LivniumCoreConfig`

## ⚠️ Important: What These Numbers Mean

### ✅ What IS Real:
- **Cell counts are accurate**: 2.5M cells = 2.5M omcube capacity
- **Math is verified**: All recursive geometry calculations are correct
- **Tests are reproducible**: Run the tests yourself to verify
- **Simulation is real**: These are actual classical simulations of quantum states

### ⚠️ What These Are NOT:
- **NOT physical qubits**: No actual quantum hardware
- **NOT quantum speedup**: Still classical computation (no exponential speedup)
- **NOT error-corrected**: Perfect by design, not by error correction
- **NOT real entanglement**: Simulated correlations, not physical entanglement

### 🎯 Honest Terminology:
- **"Omcubes"** = 3×3×3 geometric structures that can hold quantum-like states
- **"2.5M omcubes"** = 2.5M cells, each an omcube capable of holding a quantum state
- **"Entangled"** = Simulated quantum correlations between omcubes
- **"Perfect"** = No errors by design (idealized simulation)

**See `VERIFICATION.md` for complete mathematical verification of all claims.**

