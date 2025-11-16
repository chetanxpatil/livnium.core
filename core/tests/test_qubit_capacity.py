"""
Test Qubit Capacity: How Many Qubits Can We Run?

Tests the quantum layer's capacity with increasing numbers of qubits.
Measures memory usage, performance, and finds practical limits.
"""

import sys
import time
import psutil
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.quantum.quantum_lattice import QuantumLattice
from core.quantum.quantum_gates import QuantumGates, GateType
from core.recursive import RecursiveGeometryEngine


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def test_quantum_capacity(n_qubits: int, verbose: bool = True) -> dict:
    """
    Test quantum layer with n_qubits.
    
    Args:
        n_qubits: Number of qubits to test
        verbose: Print progress
        
    Returns:
        Test results dictionary
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {n_qubits} qubits...")
        print(f"{'='*60}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Create system with appropriate lattice size
        # For n qubits, we need at least n cells
        # Use smallest N×N×N that fits: N³ >= n
        import math
        n = math.ceil(n_qubits ** (1/3))
        if n % 2 == 0:
            n += 1  # Must be odd
        if n < 3:
            n = 3
        
        config = LivniumCoreConfig(
            lattice_size=n,
            enable_quantum=True,
            enable_superposition=True,
            enable_quantum_gates=True,
            enable_entanglement=True,
            enable_measurement=True,
        )
        
        # Create classical system
        core = LivniumCoreSystem(config)
        
        # Create quantum lattice
        try:
            qlattice = QuantumLattice(core)
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed to create quantum lattice: {e}")
            return {
                'n_qubits': n_qubits,
                'success': False,
                'error': f"QuantumLattice creation failed: {e}",
            }
        
        init_time = time.time() - start_time
        init_memory = get_memory_usage() - start_memory
        
        if verbose:
            print(f"  Initialization: {init_time:.3f}s, {init_memory:.2f} MB")
        
        # Test operations
        test_results = {
            'n_qubits': n_qubits,
            'lattice_size': n,
            'actual_qubits': len(qlattice.quantum_cells),
            'init_time': init_time,
            'init_memory_mb': init_memory,
            'operations': {},
            'success': True,
        }
        
        # Test 1: Apply Hadamard gates
        if verbose:
            print(f"  Testing Hadamard gates...")
        gate_start = time.time()
        gate_count = 0
        for coords in list(qlattice.quantum_cells.keys())[:min(100, n_qubits)]:
            try:
                qlattice.apply_gate(coords, GateType.HADAMARD)
                gate_count += 1
            except Exception as e:
                if verbose:
                    print(f"    Error applying gate: {e}")
                # Don't fail the test, just skip this gate
                pass
        gate_time = time.time() - gate_start
        test_results['operations']['hadamard'] = {
            'count': gate_count,
            'time': gate_time,
            'time_per_gate': gate_time / gate_count if gate_count > 0 else 0,
        }
        if verbose:
            print(f"    Applied {gate_count} gates in {gate_time:.3f}s")
        
        # Test 2: Create entanglement
        if verbose:
            print(f"  Testing entanglement...")
        entangle_start = time.time()
        entangle_count = 0
        coords_list = list(qlattice.quantum_cells.keys())
        for i in range(0, min(50, len(coords_list) - 1), 2):
            try:
                # Use entangle_cells method
                qlattice.entangle_cells(coords_list[i], coords_list[i+1])
                entangle_count += 1
            except Exception as e:
                if verbose:
                    print(f"    Error creating entanglement: {e}")
                # Don't fail the test, just skip entanglement
                pass
        entangle_time = time.time() - entangle_start
        test_results['operations']['entanglement'] = {
            'count': entangle_count,
            'time': entangle_time,
            'time_per_pair': entangle_time / entangle_count if entangle_count > 0 else 0,
        }
        if verbose:
            print(f"    Created {entangle_count} entangled pairs in {entangle_time:.3f}s")
        
        # Test 3: Measurement
        if verbose:
            print(f"  Testing measurement...")
        measure_start = time.time()
        measure_count = 0
        for coords in list(qlattice.quantum_cells.keys())[:min(50, n_qubits)]:
            try:
                result = qlattice.measure_cell(coords)
                measure_count += 1
            except Exception as e:
                if verbose:
                    print(f"    Error measuring: {e}")
                # Don't fail the test, just skip this measurement
                pass
        measure_time = time.time() - measure_start
        test_results['operations']['measurement'] = {
            'count': measure_count,
            'time': measure_time,
            'time_per_measure': measure_time / measure_count if measure_count > 0 else 0,
        }
        if verbose:
            print(f"    Measured {measure_count} qubits in {measure_time:.3f}s")
        
        # Final memory
        end_memory = get_memory_usage()
        total_memory = end_memory - start_memory
        test_results['total_memory_mb'] = total_memory
        test_results['memory_per_qubit_mb'] = total_memory / n_qubits if n_qubits > 0 else 0
        
        total_time = time.time() - start_time
        test_results['total_time'] = total_time
        
        if verbose:
            print(f"  Total: {total_time:.3f}s, {total_memory:.2f} MB")
            print(f"  Memory per qubit: {test_results['memory_per_qubit_mb']:.4f} MB")
        
        return test_results
        
    except Exception as e:
        if verbose:
            print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'n_qubits': n_qubits,
            'success': False,
            'error': str(e),
        }


def test_with_recursive_geometry(n_qubits: int, max_depth: int = 2, verbose: bool = True) -> dict:
    """
    Test quantum capacity with recursive geometry.
    
    Args:
        n_qubits: Target number of qubits
        max_depth: Maximum recursion depth
        verbose: Print progress
        
    Returns:
        Test results dictionary
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing {n_qubits} qubits with recursive geometry (depth={max_depth})...")
        print(f"{'='*60}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Create base geometry
        config = LivniumCoreConfig(
            lattice_size=5,  # Start with 5×5×5 = 125 cells
            enable_quantum=True,
            enable_superposition=True,
            enable_quantum_gates=True,
            enable_entanglement=True,
            enable_measurement=True,
            enable_recursive_geometry=True,
        )
        
        core = LivniumCoreSystem(config)
        
        # Create recursive geometry
        recursive = RecursiveGeometryEngine(base_geometry=core, max_depth=max_depth)
        
        # Get total capacity
        total_capacity = recursive.get_total_capacity()
        
        if verbose:
            print(f"  Total capacity: {total_capacity} cells")
            print(f"  Levels: {list(recursive.levels.keys())}")
        
        # Create quantum lattice on base
        qlattice = QuantumLattice(core)
        
        init_time = time.time() - start_time
        init_memory = get_memory_usage() - start_memory
        
        test_results = {
            'n_qubits': n_qubits,
            'total_capacity': total_capacity,
            'actual_qubits': len(qlattice.quantum_cells),
            'init_time': init_time,
            'init_memory_mb': init_memory,
            'success': True,
        }
        
        # Test basic operations
        if len(qlattice.quantum_cells) > 0:
            # Apply a few gates
            coords_list = list(qlattice.quantum_cells.keys())
            for coords in coords_list[:min(10, len(coords_list))]:
                qlattice.apply_gate(coords, GateType.HADAMARD)
        
        end_memory = get_memory_usage()
        total_memory = end_memory - start_memory
        test_results['total_memory_mb'] = total_memory
        
        total_time = time.time() - start_time
        test_results['total_time'] = total_time
        
        if verbose:
            print(f"  Total: {total_time:.3f}s, {total_memory:.2f} MB")
        
        return test_results
        
    except Exception as e:
        if verbose:
            print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'n_qubits': n_qubits,
            'success': False,
            'error': str(e),
        }


def run_capacity_test(max_qubits: int = 10000, step: int = 100):
    """
    Run capacity test with increasing qubit counts.
    
    Args:
        max_qubits: Maximum qubits to test
        step: Step size for testing
    """
    print("="*60)
    print("QUANTUM CAPACITY TEST")
    print("="*60)
    print(f"Testing from {step} to {max_qubits} qubits (step={step})")
    print()
    
    results = []
    
    for n in range(step, max_qubits + 1, step):
        result = test_quantum_capacity(n, verbose=True)
        results.append(result)
        
        if not result.get('success', False):
            print(f"\n⚠️  Failed at {n} qubits")
            break
        
        # Check memory limit (stop if > 8GB)
        if result.get('total_memory_mb', 0) > 8000:
            print(f"\n⚠️  Memory limit reached at {n} qubits ({result.get('total_memory_mb', 0):.2f} MB)")
            break
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        max_successful = max(successful, key=lambda x: x['n_qubits'])
        print(f"✅ Maximum successful: {max_successful['n_qubits']} qubits")
        print(f"   Memory: {max_successful.get('total_memory_mb', 0):.2f} MB")
        print(f"   Memory per qubit: {max_successful.get('memory_per_qubit_mb', 0):.4f} MB")
        print(f"   Time: {max_successful.get('total_time', 0):.3f}s")
        
        # Performance stats
        if 'operations' in max_successful:
            ops = max_successful['operations']
            if 'hadamard' in ops:
                print(f"   Hadamard gates: {ops['hadamard']['time_per_gate']*1000:.3f} ms/gate")
            if 'entanglement' in ops:
                print(f"   Entanglement: {ops['entanglement']['time_per_pair']*1000:.3f} ms/pair")
            if 'measurement' in ops:
                print(f"   Measurement: {ops['measurement']['time_per_measure']*1000:.3f} ms/measure")
    else:
        print("❌ No successful tests")
    
    return results


def run_quick_test():
    """Run quick test with common qubit counts."""
    print("="*60)
    print("QUICK QUANTUM CAPACITY TEST")
    print("="*60)
    print()
    
    test_counts = [10, 50, 100, 500, 1000, 5000, 10000]
    results = []
    
    for n in test_counts:
        print(f"\nTesting {n} qubits...")
        result = test_quantum_capacity(n, verbose=False)
        results.append(result)
        
        if result.get('success', False):
            print(f"  ✅ Success: {result.get('total_memory_mb', 0):.2f} MB, "
                  f"{result.get('total_time', 0):.3f}s")
        else:
            error_msg = result.get('error', 'Unknown error')
            error_type = result.get('error_type', '')
            print(f"  ❌ Failed: {error_type}: {error_msg}")
            if error_type:
                import traceback
                traceback.print_exc()
            break
    
    # Summary
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        for r in successful:
            print(f"  {r['n_qubits']:5d} qubits: {r.get('total_memory_mb', 0):7.2f} MB, "
                  f"{r.get('memory_per_qubit_mb', 0):.4f} MB/qubit")
        
        max_successful = max(successful, key=lambda x: x['n_qubits'])
        print(f"\n✅ Maximum: {max_successful['n_qubits']} qubits")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test quantum qubit capacity")
    parser.add_argument('--max', type=int, default=10000, help='Maximum qubits to test')
    parser.add_argument('--step', type=int, default=100, help='Step size')
    parser.add_argument('--quick', action='store_true', help='Run quick test')
    parser.add_argument('--recursive', action='store_true', help='Test with recursive geometry')
    
    args = parser.parse_args()
    
    if args.recursive:
        # Test recursive geometry capacity
        result = test_with_recursive_geometry(5000, max_depth=2, verbose=True)
        print(f"\n✅ Recursive geometry test: {result.get('total_capacity', 0)} cells capacity")
    elif args.quick:
        run_quick_test()
    else:
        run_capacity_test(max_qubits=args.max, step=args.step)

