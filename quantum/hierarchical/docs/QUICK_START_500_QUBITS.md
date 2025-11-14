# Quick Start: Running 500-Qubit Simulator

## Simple Usage

### Basic Example

```python
from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator

# Create 500-qubit simulator
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)

# Apply gates
sim.hadamard(0)
sim.cnot(0, 1)

# Measure
results = sim.run(num_shots=1000)
print(results)
```

### Run from Command Line

```bash
# Test the simulator
cd /Users/chetanpatil/Desktop/clean-nova-livnium
python3 quantum_computer/simulators/mps_hierarchical_simulator.py
```

### Run Your Own Circuit

Create a file `my_circuit.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator

# Create simulator
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)

# Build your circuit
for i in range(500):
    sim.hadamard(i)

# Create entanglement
for i in range(0, 500, 2):
    sim.cnot(i, i+1)

# Run simulation
results = sim.run(num_shots=1000)

print(f"Results: {results}")
```

Run it:
```bash
python3 my_circuit.py
```

## Available Simulators

### 1. MPS Hierarchical (500+ qubits) - RECOMMENDED
```python
from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)
```
- ✅ Handles 500+ qubits
- ✅ Memory efficient (MPS)
- ✅ Uses geometry > geometry > geometry

### 2. Hierarchical Geometry (20-25 qubits)
```python
from quantum_computer.simulators.hierarchical_geometry_simulator import HierarchicalGeometrySimulator
sim = HierarchicalGeometrySimulator(20)
```
- ✅ Sparse optimization
- ✅ Uses geometry hierarchy

### 3. Optimized Geometry (20-25 qubits)
```python
from quantum_computer.simulators.geometry_quantum_simulator_optimized import OptimizedGeometryQuantumSimulator
sim = OptimizedGeometryQuantumSimulator(20, sparse_mode=True)
```
- ✅ Sparse storage
- ✅ Efficient for medium systems

## Examples

### Example 1: Simple 500-Qubit Circuit

```python
from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator

sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)

# Initialize first 10 qubits in superposition
for i in range(10):
    sim.hadamard(i)

# Create Bell pairs
for i in range(0, 10, 2):
    sim.cnot(i, i+1)

# Measure
results = sim.run(num_shots=100)
print(results)
```

### Example 2: Large Scale Circuit

```python
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)

# Initialize all qubits
for i in range(500):
    sim.hadamard(i)

# Create 1D chain entanglement
for i in range(499):
    sim.cnot(i, i+1)

# Measure
results = sim.run(num_shots=1000)
```

### Example 3: Check Capacity

```python
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)
info = sim.get_capacity_info()

print(f"Qubits: {info['num_qubits']}")
print(f"Memory: {info['memory_mb']:.2f} MB")
print(f"Scaling: {info['scaling']}")
```

## Command Line Examples

### Test Capacity
```bash
python3 quantum_computer/simulators/mps_hierarchical_simulator.py
```

### Run Custom Script
```bash
# Create your script
cat > test_500.py << 'EOF'
from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator
sim = MPSHierarchicalGeometrySimulator(500)
sim.hadamard(0)
sim.cnot(0, 1)
results = sim.run(100)
print(results)
EOF

# Run it
python3 test_500.py
```

## Parameters

### Bond Dimension (χ)
- **Low (4-8)**: Less memory, faster, but may lose accuracy
- **Medium (16-32)**: Balanced
- **High (64+)**: More accurate, but slower and more memory

```python
# Low memory, faster
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=4)

# Balanced
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)

# High accuracy
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=32)
```

## Troubleshooting

### Import Error
```bash
# Make sure you're in the project root
cd /Users/chetanpatil/Desktop/clean-nova-livnium

# Or set PYTHONPATH
export PYTHONPATH=/Users/chetanpatil/Desktop/clean-nova-livnium:$PYTHONPATH
```

### Memory Issues
- Reduce bond dimension: `bond_dimension=4` instead of `8`
- Use smaller number of qubits for testing
- Check available system memory

### Slow Performance
- Reduce bond dimension
- Use fewer gates
- For very large systems, consider distributed processing

## Quick Reference

```python
# Import
from quantum_computer.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator

# Create (500 qubits)
sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)

# Gates
sim.hadamard(qubit)
sim.pauli_x(qubit)
sim.pauli_z(qubit)
sim.cnot(control, target)

# Measurement
result = sim.measure(qubit)
results = sim.measure_all()
stats = sim.run(num_shots=1000)

# Info
info = sim.get_capacity_info()
```

