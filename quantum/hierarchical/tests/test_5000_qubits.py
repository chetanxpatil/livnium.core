#!/usr/bin/env python3
"""
5000-Qubit-Analogue Capacity Test

⚠️ IMPORTANT: This is a quantum-inspired classical hierarchical simulator, NOT a physical quantum computer.

Tests the hierarchical geometry system's ability to handle 5000 qubit-analogues (quantum-inspired classical units).
This test verifies:
1. Qubit-analogue creation (5000 units)
2. Operations on 5000 qubit-analogues
3. Memory efficiency
4. Performance metrics

The system uses:
- Geometric compression (hierarchical state grouping)
- Tensor networks (MPS - Matrix Product States)
- Lightweight qubit-analogue objects (classical data structures)
- Hierarchical state management (geometry > geometry > geometry)
"""

import numpy as np
import time
import tracemalloc
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.core.quantum_processor import QuantumProcessor
from quantum.hierarchical.simulators.hierarchical_simulator import HierarchicalQuantumSimulator
from quantum.hierarchical.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator


def test_5000_qubit_creation():
    """Test creating 5000 qubits."""
    print("=" * 70)
    print("TEST 1: 5000-Qubit Creation")
    print("=" * 70)
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        processor = QuantumProcessor(base_dimension=3)
        
        print(f"\nCreating 5000 qubits...")
        qubit_ids = []
        
        for i in range(5000):
            # Distribute qubits in geometric space
            x = (i % 10) * 0.1
            y = ((i // 10) % 10) * 0.1
            z = (i // 100) * 0.1
            qid = processor.create_qubit((x, y, z))
            qubit_ids.append(qid)
            
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                print(f"  Progress: {i+1}/5000 qubits | Time: {elapsed:.2f}s | Memory: {peak/1024/1024:.2f} MB")
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - start_time
        
        info = processor.get_system_info()
        
        print(f"\n✅ SUCCESS!")
        print(f"  Qubits created: {len(qubit_ids):,}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Memory (current): {current/1024/1024:.2f} MB")
        print(f"  Memory (peak): {peak/1024/1024:.2f} MB")
        print(f"  Memory per qubit: {(current/1024/1024)/5000:.6f} MB")
        print(f"  Qubits/second: {5000/elapsed:.0f}")
        
        return {
            'success': True,
            'num_qubits': 5000,
            'time': elapsed,
            'memory_mb': peak / 1024 / 1024,
            'memory_per_qubit_mb': (current / 1024 / 1024) / 5000,
            'qubits_per_second': 5000 / elapsed
        }
        
    except Exception as e:
        tracemalloc.stop()
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'time': elapsed
        }


def test_5000_qubit_operations():
    """Test operations on 5000 qubits."""
    print("\n" + "=" * 70)
    print("TEST 2: 5000-Qubit Operations")
    print("=" * 70)
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        processor = QuantumProcessor(base_dimension=3)
        
        print(f"\nCreating 5000 qubits...")
        qubit_ids = []
        for i in range(5000):
            x = (i % 10) * 0.1
            y = ((i // 10) % 10) * 0.1
            z = (i // 100) * 0.1
            qid = processor.create_qubit((x, y, z))
            qubit_ids.append(qid)
        
        print(f"✅ Created {len(qubit_ids)} qubits")
        
        # Apply operations
        print(f"\nApplying operations...")
        operations_applied = 0
        
        # Apply Hadamard to first 100 qubits
        print(f"  - Applying Hadamard to qubits 0-99...")
        for i in range(100):
            processor.apply_hadamard(qubit_ids[i])
            operations_applied += 1
        
        # Apply CNOT gates
        print(f"  - Applying CNOT gates (pairs 0-1, 2-3, ..., 98-99)...")
        for i in range(0, 100, 2):
            processor.apply_cnot(qubit_ids[i], qubit_ids[i+1])
            operations_applied += 1
        
        # Apply operations to scattered qubits
        print(f"  - Applying operations to scattered qubits (100, 500, 1000, 2000, 3000, 4000)...")
        scattered_indices = [100, 500, 1000, 2000, 3000, 4000]
        for idx in scattered_indices:
            if idx < len(qubit_ids):
                processor.apply_hadamard(qubit_ids[idx])
                operations_applied += 1
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS!")
        print(f"  Total operations: {operations_applied}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Memory (peak): {peak/1024/1024:.2f} MB")
        print(f"  Operations/second: {operations_applied/elapsed:.0f}")
        
        return {
            'success': True,
            'num_qubits': 5000,
            'operations': operations_applied,
            'time': elapsed,
            'memory_mb': peak / 1024 / 1024,
            'ops_per_second': operations_applied / elapsed
        }
        
    except Exception as e:
        tracemalloc.stop()
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'time': elapsed
        }


def test_5000_qubit_mps_simulator():
    """Test MPS simulator with 5000 qubits."""
    print("\n" + "=" * 70)
    print("TEST 3: 5000-Qubit MPS Simulator")
    print("=" * 70)
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        print(f"\nCreating MPS simulator with 5000 qubits...")
        sim = MPSHierarchicalGeometrySimulator(5000, bond_dimension=8)
        
        # Get capacity info
        info = sim.get_capacity_info()
        print(f"\n  Capacity Info:")
        print(f"  - Qubits: {info['num_qubits']:,}")
        print(f"  - Memory: {info['memory_mb']:.2f} MB")
        print(f"  - Scaling: {info['scaling']}")
        
        # Apply some gates
        print(f"\nApplying gates...")
        print(f"  - Hadamard on qubits 0-9...")
        for i in range(10):
            sim.hadamard(i)
        
        print(f"  - CNOT on pairs (0-1, 2-3, 4-5, 6-7, 8-9)...")
        for i in range(0, 10, 2):
            sim.cnot(i, i+1)
        
        print(f"  - Hadamard on scattered qubits (100, 1000, 2000, 3000, 4000)...")
        for idx in [100, 1000, 2000, 3000, 4000]:
            sim.hadamard(idx)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS!")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Memory (peak): {peak/1024/1024:.2f} MB")
        print(f"  Theoretical memory (full state): 2^5000 states = impossible!")
        print(f"  Actual memory: {peak/1024/1024:.2f} MB (MPS compression)")
        
        return {
            'success': True,
            'num_qubits': 5000,
            'time': elapsed,
            'memory_mb': peak / 1024 / 1024,
            'theoretical_memory': '2^5000 (impossible)',
            'mps_memory_mb': info['memory_mb']
        }
        
    except Exception as e:
        tracemalloc.stop()
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'time': elapsed
        }


def test_5000_qubit_hierarchical_simulator():
    """Test hierarchical simulator with 5000 qubits."""
    print("\n" + "=" * 70)
    print("TEST 4: 5000-Qubit Hierarchical Simulator")
    print("=" * 70)
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        print(f"\nCreating hierarchical simulator...")
        simulator = HierarchicalQuantumSimulator(base_dimension=3)
        
        print(f"Creating 5000 qubits...")
        qubit_ids = []
        for i in range(5000):
            x = (i % 10) * 0.1
            y = ((i // 10) % 10) * 0.1
            z = (i // 100) * 0.1
            qid = simulator.add_qubit((x, y, z))
            qubit_ids.append(qid)
            
            if (i + 1) % 1000 == 0:
                print(f"  Progress: {i+1}/5000 qubits")
        
        print(f"✅ Created {len(qubit_ids)} qubits")
        
        # Build circuit
        print(f"\nBuilding circuit...")
        print(f"  - Hadamard on qubits 0-9...")
        for i in range(10):
            simulator.hadamard(qubit_ids[i])
        
        print(f"  - CNOT on pairs...")
        for i in range(0, 10, 2):
            simulator.cnot(qubit_ids[i], qubit_ids[i+1])
        
        # Run simulation
        print(f"\nRunning simulation (100 shots)...")
        sim_results = simulator.run(num_shots=100)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.time() - start_time
        
        print(f"\n✅ SUCCESS!")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Memory (peak): {peak/1024/1024:.2f} MB")
        print(f"  Shots: {sim_results['shots']}")
        print(f"  Unique outcomes: {len(sim_results['results'])}")
        
        return {
            'success': True,
            'num_qubits': 5000,
            'time': elapsed,
            'memory_mb': peak / 1024 / 1024,
            'shots': sim_results['shots'],
            'unique_outcomes': len(sim_results['results'])
        }
        
    except Exception as e:
        tracemalloc.stop()
        elapsed = time.time() - start_time
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'time': elapsed
        }


def print_summary(results):
    """Print summary of all tests."""
    print("\n" + "=" * 70)
    print("5000-QUBIT TEST SUMMARY")
    print("=" * 70)
    
    for i, result in enumerate(results, 1):
        test_name = [
            "Qubit Creation",
            "Operations",
            "MPS Simulator",
            "Hierarchical Simulator"
        ][i-1]
        
        print(f"\nTest {i}: {test_name}")
        if result.get('success'):
            print(f"  ✅ PASSED")
            if 'num_qubits' in result:
                print(f"  Qubits: {result['num_qubits']:,}")
            if 'time' in result:
                print(f"  Time: {result['time']:.3f}s")
            if 'memory_mb' in result:
                print(f"  Memory: {result['memory_mb']:.2f} MB")
            if 'memory_per_qubit_mb' in result:
                print(f"  Memory/qubit: {result['memory_per_qubit_mb']:.6f} MB")
            if 'operations' in result:
                print(f"  Operations: {result['operations']}")
            if 'ops_per_second' in result:
                print(f"  Ops/sec: {result['ops_per_second']:.0f}")
        else:
            print(f"  ❌ FAILED")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    
    # Overall status
    all_passed = all(r.get('success', False) for r in results)
    print(f"\n{'=' * 70}")
    if all_passed:
        print("✅ ALL TESTS PASSED - 5000 QUBITS SUPPORTED!")
    else:
        print("⚠️  SOME TESTS FAILED - CHECK RESULTS ABOVE")
    print("=" * 70)


def main():
    """Run all 5000-qubit tests."""
    print("\n" + "=" * 70)
    print("5000-QUBIT CAPACITY TEST")
    print("=" * 70)
    print("\nTesting hierarchical geometry system's ability to handle 5000 qubits...")
    print("This test verifies creation, operations, and simulation capabilities.\n")
    
    results = []
    
    # Test 1: Qubit creation
    result1 = test_5000_qubit_creation()
    results.append(result1)
    
    # Test 2: Operations
    result2 = test_5000_qubit_operations()
    results.append(result2)
    
    # Test 3: MPS Simulator
    result3 = test_5000_qubit_mps_simulator()
    results.append(result3)
    
    # Test 4: Hierarchical Simulator
    result4 = test_5000_qubit_hierarchical_simulator()
    results.append(result4)
    
    # Print summary
    print_summary(results)
    
    return results


if __name__ == '__main__':
    main()

