# Quantum-Inspired Livnium Core

This directory contains quantum-related experiments and tests using the Livnium Core System's quantum layer and recursive geometry engine.

## Files

### Tests
- **`test_recursive_qubit_capacity.py`**: Tests how many logical qubits can be represented using recursive geometry (currently ~2.5M qubits)
- **`test_entanglement_capacity.py`**: Tests entanglement capacity - how many qubits can be entangled simultaneously (pairwise, chains, clusters)

### Experiments
- **`aes128_quantum_topology_mapper.py`**: Quantum-enhanced topology mapper for AES-128 that uses superposition and recursive geometry to explore the key space landscape
- **`aes128_round_sweep_experiment_livnium.py`**: AES-128 round sweep experiment using full Livnium architecture (recursive geometry + quantum layer + tension fields)

### Documentation
- **`QUBIT_ANALYSIS.md`**: Analysis of qubit types (logical vs physical) in the Livnium implementation

## Key Features

### Recursive Geometry Qubit Capacity
- **2.5 Million Logical Qubits**: The recursive geometry engine can represent millions of qubits
- **Perfect Simulation**: Error-free, infinite coherence (simulated logical qubits)
- **Exponential Capacity**: Linear memory, exponential qubit count

### Entanglement Capacity
- **Pairwise Entanglement**: Bell pairs between qubits
- **Chain Entanglement**: Linear chains of entangled qubits
- **Cluster Entanglement**: Fully connected clusters
- **Recursive Entanglement**: Cross-level entanglement (potential)

### Quantum-Enhanced Cryptanalysis
- **Superposition Search**: Uses quantum superposition to explore key space
- **Tension Fields**: Encodes constraints as geometric tension
- **Recursive Search**: Subdivides key space across multiple scales

## Usage

### Test Qubit Capacity
```bash
python3 experiments/quantum-inspired-livnium-core/test_recursive_qubit_capacity.py
```

### Test Entanglement Capacity
```bash
python3 experiments/quantum-inspired-livnium-core/test_entanglement_capacity.py
```

### Run Quantum Topology Mapper
```bash
python3 experiments/quantum-inspired-livnium-core/aes128_quantum_topology_mapper.py
```

### Run AES Round Sweep (Quantum)
```bash
python3 experiments/quantum-inspired-livnium-core/aes128_round_sweep_experiment_livnium.py
```

## Architecture

These experiments use:
- **Layer 0**: Recursive Geometry Engine (subdivides space)
- **Layer 2**: Quantum Layer (superposition, entanglement, measurement)
- **Tension Fields**: Constraint encoding as geometric tension

## Results

### Qubit Capacity
- **Level 0**: 125 qubits (5×5×5 base)
- **Level 1**: 3,375 qubits
- **Level 2**: 91,125 qubits
- **Level 3**: 2,460,375 qubits
- **Total**: 2,555,000 logical qubits

### Entanglement Capacity
- **Pairwise**: 150 Bell pairs, 125 entangled qubits
- **Chains**: 1 chain of length 125
- **Clusters**: 13 clusters, 550 pairs
- **Cross-level**: 3,375 potential entanglements

## Notes

- All qubits are **simulated logical qubits** (perfect, error-free)
- Recursive geometry compresses entanglement into lower-scale geometry
- Quantum features require proper configuration in `LivniumCoreConfig`

## ⚠️ Important: What These Numbers Mean

### ✅ What IS Real:
- **Cell counts are accurate**: 2.5M cells = 2.5M quantum state capacity
- **Math is verified**: All recursive geometry calculations are correct
- **Tests are reproducible**: Run the tests yourself to verify
- **Simulation is real**: These are actual classical simulations of quantum states

### ⚠️ What These Are NOT:
- **NOT physical qubits**: No actual quantum hardware
- **NOT quantum speedup**: Still classical computation (no exponential speedup)
- **NOT error-corrected**: Perfect by design, not by error correction
- **NOT real entanglement**: Simulated correlations, not physical entanglement

### 🎯 Honest Terminology:
- **"Simulated logical qubits"** = Classical simulation of perfect quantum states
- **"2.5M qubits"** = 2.5M cells, each capable of holding a quantum state
- **"Entangled"** = Simulated quantum correlations between cells
- **"Perfect"** = No errors by design (idealized simulation)

**See `VERIFICATION.md` for complete mathematical verification of all claims.**

