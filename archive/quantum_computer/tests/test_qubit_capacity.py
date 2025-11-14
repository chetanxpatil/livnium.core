"""
Test Qubit Capacity: How many qubits can the hierarchical geometry system handle?

Tests the system's capacity by creating increasing numbers of qubits
and measuring performance.
"""

import numpy as np
import time
import tracemalloc
from collections import Counter

from quantum_computer.core.quantum_processor import QuantumProcessor
from quantum_computer.simulators.hierarchical_simulator import HierarchicalQuantumSimulator


def test_qubit_creation(max_qubits: int = 10000, step: int = 100):
    """
    Test how many qubits can be created.
    
    Args:
        max_qubits: Maximum qubits to test
        step: Step size for testing
    """
    print("=" * 70)
    print("Qubit Capacity Test: Hierarchical Geometry Quantum Computer")
    print("=" * 70)
    
    results = []
    
    for num_qubits in range(step, max_qubits + 1, step):
        print(f"\nTesting {num_qubits} qubits...")
        
        # Start memory tracking
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Create processor
            processor = QuantumProcessor(base_dimension=3)
            
            # Create qubits
            qubit_ids = []
            for i in range(num_qubits):
                # Distribute qubits in geometric space
                x = (i % 10) * 0.1
                y = ((i // 10) % 10) * 0.1
                z = (i // 100) * 0.1
                qid = processor.create_qubit((x, y, z))
                qubit_ids.append(qid)
            
            # Get memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            elapsed = time.time() - start_time
            
            # Get system info
            info = processor.get_system_info()
            
            results.append({
                'num_qubits': num_qubits,
                'success': True,
                'time': elapsed,
                'memory_current_mb': current / 1024 / 1024,
                'memory_peak_mb': peak / 1024 / 1024,
                'qubits_per_second': num_qubits / elapsed if elapsed > 0 else 0,
                'memory_per_qubit_mb': (current / 1024 / 1024) / num_qubits if num_qubits > 0 else 0
            })
            
            print(f"  ✅ Success!")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Memory: {current/1024/1024:.2f} MB (peak: {peak/1024/1024:.2f} MB)")
            print(f"  Qubits/sec: {num_qubits/elapsed:.0f}")
            print(f"  Memory/qubit: {(current/1024/1024)/num_qubits:.4f} MB")
            
        except Exception as e:
            tracemalloc.stop()
            elapsed = time.time() - start_time
            
            results.append({
                'num_qubits': num_qubits,
                'success': False,
                'error': str(e),
                'time': elapsed
            })
            
            print(f"  ❌ Failed: {e}")
            print(f"  Maximum achieved: {num_qubits - step} qubits")
            break
    
    return results


def test_operations_capacity(max_qubits: int = 1000):
    """
    Test capacity with operations (gates).
    
    Args:
        max_qubits: Maximum qubits to test
    """
    print("\n" + "=" * 70)
    print("Operations Capacity Test")
    print("=" * 70)
    
    results = []
    
    for num_qubits in [10, 50, 100, 500, 1000, 2000, 5000]:
        if num_qubits > max_qubits:
            break
        
        print(f"\nTesting {num_qubits} qubits with operations...")
        
        tracemalloc.start()
        start_time = time.time()
        
        try:
            processor = QuantumProcessor(base_dimension=3)
            
            # Create qubits
            qubit_ids = []
            for i in range(num_qubits):
                x = (i % 10) * 0.1
                y = ((i // 10) % 10) * 0.1
                z = (i // 100) * 0.1
                qid = processor.create_qubit((x, y, z))
                qubit_ids.append(qid)
            
            # Apply operations
            operations_applied = 0
            for i in range(min(100, num_qubits)):
                processor.apply_hadamard(qubit_ids[i])
                operations_applied += 1
            
            # Apply CNOT gates
            for i in range(min(50, num_qubits - 1)):
                processor.apply_cnot(qubit_ids[i], qubit_ids[i+1])
                operations_applied += 1
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed = time.time() - start_time
            
            results.append({
                'num_qubits': num_qubits,
                'operations': operations_applied,
                'success': True,
                'time': elapsed,
                'memory_mb': peak / 1024 / 1024,
                'ops_per_second': operations_applied / elapsed if elapsed > 0 else 0
            })
            
            print(f"  ✅ Success!")
            print(f"  Operations: {operations_applied}")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Memory: {peak/1024/1024:.2f} MB")
            print(f"  Ops/sec: {operations_applied/elapsed:.0f}")
            
        except Exception as e:
            tracemalloc.stop()
            print(f"  ❌ Failed: {e}")
            break
    
    return results


def test_simulator_capacity(max_qubits: int = 100):
    """
    Test simulator capacity.
    
    Args:
        max_qubits: Maximum qubits to test
    """
    print("\n" + "=" * 70)
    print("Simulator Capacity Test")
    print("=" * 70)
    
    results = []
    
    for num_qubits in [5, 10, 20, 50, 100, 200]:
        if num_qubits > max_qubits:
            break
        
        print(f"\nTesting simulator with {num_qubits} qubits...")
        
        tracemalloc.start()
        start_time = time.time()
        
        try:
            simulator = HierarchicalQuantumSimulator(base_dimension=3)
            
            # Create qubits
            qubit_ids = []
            for i in range(num_qubits):
                x = (i % 10) * 0.1
                y = ((i // 10) % 10) * 0.1
                z = (i // 100) * 0.1
                qid = simulator.add_qubit((x, y, z))
                qubit_ids.append(qid)
            
            # Build circuit
            for i in range(min(10, num_qubits)):
                simulator.hadamard(qubit_ids[i])
            
            for i in range(min(5, num_qubits - 1)):
                simulator.cnot(qubit_ids[i], qubit_ids[i+1])
            
            # Run simulation
            sim_results = simulator.run(num_shots=100)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed = time.time() - start_time
            
            results.append({
                'num_qubits': num_qubits,
                'success': True,
                'time': elapsed,
                'memory_mb': peak / 1024 / 1024,
                'shots': sim_results['shots'],
                'unique_outcomes': len(sim_results['results'])
            })
            
            print(f"  ✅ Success!")
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Memory: {peak/1024/1024:.2f} MB")
            print(f"  Shots: {sim_results['shots']}")
            print(f"  Unique outcomes: {len(sim_results['results'])}")
            
        except Exception as e:
            tracemalloc.stop()
            print(f"  ❌ Failed: {e}")
            break
    
    return results


def print_summary(creation_results, operation_results, simulator_results):
    """Print summary of all tests."""
    print("\n" + "=" * 70)
    print("CAPACITY TEST SUMMARY")
    print("=" * 70)
    
    # Creation test summary
    if creation_results:
        successful = [r for r in creation_results if r.get('success', False)]
        if successful:
            max_created = max(r['num_qubits'] for r in successful)
            print(f"\n✅ Maximum Qubits Created: {max_created:,}")
            
            last_success = successful[-1]
            print(f"   Time: {last_success['time']:.3f}s")
            print(f"   Memory: {last_success['memory_current_mb']:.2f} MB")
            print(f"   Memory per qubit: {last_success['memory_per_qubit_mb']:.4f} MB")
            print(f"   Qubits/sec: {last_success['qubits_per_second']:.0f}")
    
    # Operations test summary
    if operation_results:
        successful = [r for r in operation_results if r.get('success', False)]
        if successful:
            max_ops = max(r['num_qubits'] for r in successful)
            print(f"\n✅ Maximum Qubits with Operations: {max_ops:,}")
            last_success = successful[-1]
            print(f"   Operations: {last_success['operations']}")
            print(f"   Ops/sec: {last_success['ops_per_second']:.0f}")
    
    # Simulator test summary
    if simulator_results:
        successful = [r for r in simulator_results if r.get('success', False)]
        if successful:
            max_sim = max(r['num_qubits'] for r in successful)
            print(f"\n✅ Maximum Qubits in Simulator: {max_sim:,}")
            last_success = successful[-1]
            print(f"   Shots: {last_success['shots']}")
            print(f"   Unique outcomes: {last_success['unique_outcomes']}")
    
    print("\n" + "=" * 70)


def main():
    """Run all capacity tests."""
    print("Starting capacity tests...\n")
    
    # Test 1: Qubit creation
    creation_results = test_qubit_creation(max_qubits=5000, step=100)
    
    # Test 2: Operations capacity
    operation_results = test_operations_capacity(max_qubits=2000)
    
    # Test 3: Simulator capacity
    simulator_results = test_simulator_capacity(max_qubits=200)
    
    # Print summary
    print_summary(creation_results, operation_results, simulator_results)
    
    print("\n✅ Capacity tests complete!")


if __name__ == '__main__':
    main()

