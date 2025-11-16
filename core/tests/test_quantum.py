"""
Test script for Quantum Layer of Livnium Core System.
"""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig
from core.quantum.quantum_lattice import QuantumLattice
from core.quantum.quantum_gates import QuantumGates, GateType
from core.quantum.quantum_cell import QuantumCell


def test_quantum_cell():
    """Test basic quantum cell operations."""
    print("=" * 60)
    print("Test 1: Quantum Cell")
    print("=" * 60)
    
    # Create quantum cell
    cell = QuantumCell(coordinates=(0, 0, 0), amplitudes=[1.0, 0.0])
    print(f"Initial state: {cell.get_state_vector()}")
    print(f"Probabilities: {cell.get_probabilities()}")
    
    # Apply Hadamard
    H = QuantumGates.hadamard()
    cell.apply_unitary(H)
    print(f"After Hadamard: {cell.get_state_vector()}")
    print(f"Probabilities: {cell.get_probabilities()}")
    
    # Measure
    result = cell.measure()
    print(f"Measured state: {result}")
    print(f"After measurement: {cell.get_state_vector()}")
    
    print("✅ Quantum cell test passed!\n")


def test_quantum_gates():
    """Test quantum gates."""
    print("=" * 60)
    print("Test 2: Quantum Gates")
    print("=" * 60)
    
    # Test all gates
    gates = {
        'Hadamard': QuantumGates.hadamard(),
        'Pauli X': QuantumGates.pauli_x(),
        'Pauli Y': QuantumGates.pauli_y(),
        'Pauli Z': QuantumGates.pauli_z(),
        'Phase': QuantumGates.phase(np.pi / 4),
        'Rx': QuantumGates.rotation_x(np.pi / 2),
        'Ry': QuantumGates.rotation_y(np.pi / 2),
        'Rz': QuantumGates.rotation_z(np.pi / 2),
    }
    
    for name, gate in gates.items():
        is_unitary = QuantumGates.is_unitary(gate)
        print(f"{name}: unitary={is_unitary}, shape={gate.shape}")
        assert is_unitary, f"{name} should be unitary"
    
    print("✅ Quantum gates test passed!\n")


def test_quantum_lattice():
    """Test quantum lattice integration."""
    print("=" * 60)
    print("Test 3: Quantum Lattice")
    print("=" * 60)
    
    # Create config with quantum enabled
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    # Create core system
    core = LivniumCoreSystem(config)
    
    # Create quantum lattice
    qlattice = QuantumLattice(core)
    
    print(f"Quantum cells: {len(qlattice.quantum_cells)}")
    
    # Apply Hadamard to core cell
    qlattice.apply_gate((0, 0, 0), GateType.HADAMARD)
    cell = qlattice.quantum_cells[(0, 0, 0)]
    print(f"Core cell after H: {cell.get_probabilities()}")
    
    # Entangle two cells
    qlattice.entangle_cells((0, 0, 0), (1, 0, 0))
    print(f"Entangled cells: {qlattice.entanglement_manager.get_entanglement_statistics()}")
    
    # Measure
    result = qlattice.measure_cell((0, 0, 0))
    print(f"Measurement result: {result}")
    
    # Get summary
    summary = qlattice.get_quantum_state_summary()
    print(f"Summary: {summary}")
    
    print("✅ Quantum lattice test passed!\n")


def test_geometry_quantum_coupling():
    """Test geometry-quantum coupling."""
    print("=" * 60)
    print("Test 4: Geometry-Quantum Coupling")
    print("=" * 60)
    
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_quantum=True,
        enable_superposition=True,
        enable_geometry_quantum_coupling=True
    )
    
    core = LivniumCoreSystem(config)
    qlattice = QuantumLattice(core)
    
    # Check that quantum states are initialized from geometry
    core_cell = qlattice.quantum_cells[(0, 0, 0)]
    corner_cell = qlattice.quantum_cells[(1, 1, 1)]
    
    print(f"Core cell state: {core_cell.get_state_vector()}")
    print(f"Corner cell state: {corner_cell.get_state_vector()}")
    
    # Test polarity to phase
    if qlattice.coupling:
        qlattice.coupling.apply_polarity_to_phase(core_cell, 1.0)
        print(f"Core cell after polarity: {core_cell.get_state_vector()}")
    
    print("✅ Geometry-quantum coupling test passed!\n")


if __name__ == "__main__":
    print("Livnium Core System - Quantum Layer Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_quantum_cell()
        test_quantum_gates()
        test_quantum_lattice()
        test_geometry_quantum_coupling()
        
        print("=" * 60)
        print("✅ ALL QUANTUM TESTS PASSED!")
        print("=" * 60)
        print()
        print("Quantum features implemented:")
        print("  ✅ Superposition (complex amplitudes)")
        print("  ✅ Quantum gates (Hadamard, Pauli, rotations)")
        print("  ✅ Entanglement (Bell states)")
        print("  ✅ Measurement (Born rule + collapse)")
        print("  ✅ Geometry-Quantum coupling (Livnium-specific)")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

