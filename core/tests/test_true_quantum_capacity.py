"""
Test TrueQuantumRegister capacity - the real tensor product quantum mechanics.
"""

import sys
import os
import numpy as np
import psutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.quantum.true_quantum_layer import TrueQuantumRegister
from core.quantum.quantum_gates import QuantumGates


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_true_quantum_register(n_qubits: int):
    """Test TrueQuantumRegister with n qubits."""
    print(f"\n{'='*70}")
    print(f"Testing TrueQuantumRegister with {n_qubits} qubits")
    print(f"{'='*70}")
    
    start_memory = get_memory_usage()
    
    try:
        # Create register
        qubit_ids = list(range(n_qubits))
        register = TrueQuantumRegister(qubit_ids)
        
        after_init_memory = get_memory_usage()
        memory_used = after_init_memory - start_memory
        
        # Calculate expected memory
        dim = 2 ** n_qubits
        # Each complex number is 16 bytes (8 bytes real + 8 bytes imag)
        expected_memory_mb = (dim * 16) / (1024 * 1024)
        
        print(f"  State vector dimension: 2^{n_qubits} = {dim:,}")
        print(f"  Expected memory: {expected_memory_mb:.2f} MB")
        print(f"  Actual memory used: {memory_used:.2f} MB")
        
        # Test operations
        print(f"\n  Testing operations...")
        
        # Apply Hadamard to first qubit
        H = QuantumGates.hadamard()
        register.apply_gate(H, target_id=0)
        print(f"    ✅ Applied Hadamard to qubit 0")
        
        # Apply CNOT if we have at least 2 qubits
        if n_qubits >= 2:
            register.apply_cnot(control_id=0, target_id=1)
            print(f"    ✅ Applied CNOT(0, 1)")
        
        # Measure first qubit
        result = register.measure_qubit(0)
        print(f"    ✅ Measured qubit 0: {result}")
        
        return {
            'n_qubits': n_qubits,
            'dim': dim,
            'expected_memory_mb': expected_memory_mb,
            'actual_memory_mb': memory_used,
            'success': True
        }
        
    except MemoryError as e:
        print(f"  ❌ MemoryError: {e}")
        return {
            'n_qubits': n_qubits,
            'success': False,
            'error': 'MemoryError',
            'error_msg': str(e)
        }
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'n_qubits': n_qubits,
            'success': False,
            'error': type(e).__name__,
            'error_msg': str(e)
        }


def run_capacity_test():
    """Test increasing qubit counts until failure."""
    print("="*70)
    print("TRUE QUANTUM REGISTER CAPACITY TEST")
    print("="*70)
    print("Testing tensor product quantum mechanics (2^N state vector)")
    print()
    
    # Test different qubit counts (conservative - stop before memory limit)
    # 24 qubits = 2^24 = 16M complex numbers = ~256 MB (safe)
    # 25 qubits = 2^25 = 32M complex numbers = ~512 MB
    # 26 qubits = 2^26 = 64M complex numbers = ~1 GB
    test_counts = [1, 2, 3, 5, 10, 15, 18, 20, 22, 24]
    
    results = []
    
    for n in test_counts:
        result = test_true_quantum_register(n)
        results.append(result)
        
        if not result.get('success', False):
            print(f"\n⚠️  Failed at {n} qubits")
            break
        
        # Check if we're hitting memory limits
        if result.get('expected_memory_mb', 0) > 10000:  # 10GB
            print(f"\n⚠️  Approaching memory limit at {n} qubits")
            print(f"    Expected memory: {result.get('expected_memory_mb', 0):.2f} MB")
            break
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        print(f"\n✅ Successfully tested up to {max(r['n_qubits'] for r in successful)} qubits")
        print("\nDetailed results:")
        for r in successful:
            print(f"  {r['n_qubits']:2d} qubits: dim={r['dim']:>12,}, "
                  f"expected={r.get('expected_memory_mb', 0):>8.2f} MB, "
                  f"actual={r.get('actual_memory_mb', 0):>6.2f} MB")
    else:
        print("❌ No successful tests")
    
    return results


if __name__ == "__main__":
    run_capacity_test()

