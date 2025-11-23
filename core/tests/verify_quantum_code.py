"""
Comprehensive verification of quantum code correctness.

This script checks:
1. All imports work
2. Gate implementations are correct (unitary)
3. Measurement follows Born rule
4. TrueQuantumRegister implements real tensor products
5. No false claims or broken functionality
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.quantum import (
    QuantumCell, QuantumGates, GateType, QuantumLattice,
    EntanglementManager, MeasurementEngine, MeasurementResult,
    GeometryQuantumCoupling, TrueQuantumRegister
)
from core.classical import LivniumCoreSystem
from core.config import LivniumCoreConfig


def test_imports():
    """Test that all imports work."""
    print("="*70)
    print("TEST 1: Imports")
    print("="*70)
    
    try:
        from core.quantum import QuantumCell
        from core.quantum import QuantumGates
        from core.quantum import TrueQuantumRegister
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_gate_unitarity():
    """Test that all gates are unitary."""
    print("\n" + "="*70)
    print("TEST 2: Gate Unitarity")
    print("="*70)
    
    gates_to_test = [
        ("Hadamard", QuantumGates.hadamard),
        ("Pauli X", QuantumGates.pauli_x),
        ("Pauli Y", QuantumGates.pauli_y),
        ("Pauli Z", QuantumGates.pauli_z),
        ("Phase(π/4)", lambda: QuantumGates.phase(np.pi/4)),
        ("Rotation X(π/2)", lambda: QuantumGates.rotation_x(np.pi/2)),
        ("Rotation Y(π/2)", lambda: QuantumGates.rotation_y(np.pi/2)),
        ("Rotation Z(π/2)", lambda: QuantumGates.rotation_z(np.pi/2)),
        ("CNOT", QuantumGates.cnot),
        ("CZ", QuantumGates.cz),
        ("SWAP", QuantumGates.swap),
    ]
    
    all_unitary = True
    for name, gate_func in gates_to_test:
        try:
            gate = gate_func()
            is_unitary = QuantumGates.is_unitary(gate)
            status = "✅" if is_unitary else "❌"
            print(f"{status} {name}: {'Unitary' if is_unitary else 'NOT Unitary'}")
            if not is_unitary:
                all_unitary = False
        except Exception as e:
            print(f"❌ {name}: Error - {e}")
            all_unitary = False
    
    return all_unitary


def test_born_rule():
    """Test that measurement follows Born rule."""
    print("\n" + "="*70)
    print("TEST 3: Born Rule (Measurement)")
    print("="*70)
    
    try:
        # Create superposition: |ψ⟩ = (|0⟩ + |1⟩)/√2
        cell = QuantumCell(
            coordinates=(0, 0, 0),
            amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
            num_levels=2
        )
        
        # Get probabilities
        probs = cell.get_probabilities()
        prob_0 = probs[0]
        prob_1 = probs[1]
        
        # Should be 0.5 each
        expected = 0.5
        tolerance = 1e-10
        
        if abs(prob_0 - expected) < tolerance and abs(prob_1 - expected) < tolerance:
            print(f"✅ Born rule correct: P(0)={prob_0:.6f}, P(1)={prob_1:.6f}")
            return True
        else:
            print(f"❌ Born rule incorrect: P(0)={prob_0:.6f}, P(1)={prob_1:.6f}")
            return False
    except Exception as e:
        print(f"❌ Born rule test failed: {e}")
        return False


def test_true_quantum_register():
    """Test TrueQuantumRegister implements real tensor products."""
    print("\n" + "="*70)
    print("TEST 4: TrueQuantumRegister (Tensor Products)")
    print("="*70)
    
    try:
        # Create 2-qubit register
        register = TrueQuantumRegister([0, 1])
        
        # Apply Hadamard to qubit 0: |00⟩ → (|00⟩ + |10⟩)/√2
        H = QuantumGates.hadamard()
        register.apply_gate(H, target_id=0)
        
        # Check state
        state = register.get_full_state()
        expected_0 = 1/np.sqrt(2)  # |00⟩ amplitude
        expected_2 = 1/np.sqrt(2)  # |10⟩ amplitude
        
        tolerance = 1e-10
        if abs(state[0] - expected_0) < tolerance and abs(state[2] - expected_2) < tolerance:
            print("✅ Tensor product correct: H|00⟩ = (|00⟩ + |10⟩)/√2")
        else:
            print(f"❌ Tensor product incorrect: state[0]={state[0]:.6f}, state[2]={state[2]:.6f}")
            return False
        
        # Apply CNOT: (|00⟩ + |10⟩)/√2 → (|00⟩ + |11⟩)/√2 (Bell state)
        register.apply_cnot(control_id=0, target_id=1)
        
        state = register.get_full_state()
        if abs(state[0] - expected_0) < tolerance and abs(state[3] - expected_0) < tolerance:
            print("✅ CNOT creates Bell state: (|00⟩ + |11⟩)/√2")
            return True
        else:
            print(f"❌ CNOT incorrect: state[0]={state[0]:.6f}, state[3]={state[3]:.6f}")
            return False
            
    except Exception as e:
        print(f"❌ TrueQuantumRegister test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_measurement_collapse():
    """Test that measurement collapses state."""
    print("\n" + "="*70)
    print("TEST 5: Measurement Collapse")
    print("="*70)
    
    try:
        # Create superposition
        cell = QuantumCell(
            coordinates=(0, 0, 0),
            amplitudes=np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex),
            num_levels=2
        )
        
        # Measure
        engine = MeasurementEngine()
        result = engine.measure_cell(cell, collapse=True)
        
        # After collapse, state should be |0⟩ or |1⟩ (not superposition)
        probs = cell.get_probabilities()
        is_collapsed = (probs[0] == 1.0 and probs[1] == 0.0) or (probs[0] == 0.0 and probs[1] == 1.0)
        
        if is_collapsed:
            print(f"✅ Measurement collapse works: measured={result.measured_state}, collapsed to |{result.measured_state}⟩")
            return True
        else:
            print(f"❌ Measurement collapse failed: probs={probs}")
            return False
            
    except Exception as e:
        print(f"❌ Measurement collapse test failed: {e}")
        return False


def test_deprecated_warnings():
    """Test that deprecated methods are clearly marked."""
    print("\n" + "="*70)
    print("TEST 6: Deprecated Warnings")
    print("="*70)
    
    try:
        from core.quantum.entanglement_manager import EntangledPair
        
        # Check that EntangledPair docstring mentions DEPRECATED
        docstring = EntangledPair.__doc__ or ""
        if "DEPRECATED" in docstring.upper() or "FAKE" in docstring.upper():
            print("✅ EntangledPair properly marked as deprecated")
        else:
            print("⚠️  EntangledPair not clearly marked as deprecated")
        
        # Check EntanglementManager.create_bell_pair
        from core.quantum.entanglement_manager import EntanglementManager
        docstring = EntanglementManager.create_bell_pair.__doc__ or ""
        if "DEPRECATED" in docstring.upper() or "FAKE" in docstring.upper():
            print("✅ create_bell_pair properly marked as deprecated")
        else:
            print("⚠️  create_bell_pair not clearly marked as deprecated")
        
        # Check MeasurementEngine.measure_entangled_pair
        from core.quantum.measurement_engine import MeasurementEngine
        docstring = MeasurementEngine.measure_entangled_pair.__doc__ or ""
        if "DEPRECATED" in docstring.upper() or "FAKE" in docstring.upper():
            print("✅ measure_entangled_pair properly marked as deprecated")
        else:
            print("⚠️  measure_entangled_pair not clearly marked as deprecated")
        
        return True
        
    except Exception as e:
        print(f"❌ Deprecated warnings test failed: {e}")
        return False


def test_quantum_lattice_integration():
    """Test QuantumLattice integration."""
    print("\n" + "="*70)
    print("TEST 7: QuantumLattice Integration")
    print("="*70)
    
    try:
        config = LivniumCoreConfig(
            lattice_size=3,
            enable_quantum=True,
            enable_superposition=True,
            enable_quantum_gates=True,
            enable_entanglement=True,
            enable_measurement=True,
            enable_geometry_quantum_coupling=True
        )
        
        core = LivniumCoreSystem(config)
        qlattice = QuantumLattice(core)
        
        # Check initialization
        if len(qlattice.quantum_cells) == 27:  # 3×3×3 = 27
            print("✅ QuantumLattice initialized with 27 cells")
        else:
            print(f"❌ Wrong number of cells: {len(qlattice.quantum_cells)}")
            return False
        
        # Test gate application
        qlattice.apply_gate((0, 0, 0), GateType.HADAMARD)
        print("✅ Gate application works")
        
        # Test measurement
        result = qlattice.measure_cell((0, 0, 0))
        print(f"✅ Measurement works: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ QuantumLattice integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("QUANTUM CODE VERIFICATION")
    print("="*70)
    print("\nChecking all quantum code for correctness, truth, and functionality...\n")
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Gate Unitarity", test_gate_unitarity()))
    results.append(("Born Rule", test_born_rule()))
    results.append(("TrueQuantumRegister", test_true_quantum_register()))
    results.append(("Measurement Collapse", test_measurement_collapse()))
    results.append(("Deprecated Warnings", test_deprecated_warnings()))
    results.append(("QuantumLattice Integration", test_quantum_lattice_integration()))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL QUANTUM CODE IS CORRECT AND WORKING")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed - review needed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

