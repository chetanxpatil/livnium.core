"""
Test capacity of geometry-based quantum simulator.
"""


from quantum.hierarchical.simulators.geometry_quantum_simulator import GeometryQuantumSimulator
import time
import tracemalloc


def test_capacity(max_qubits: int = 20):
    """Test how many qubits the geometry simulator can handle."""
    print("=" * 70)
    print("Geometry Simulator Capacity Test")
    print("=" * 70)
    
    results = []
    
    for num_qubits in range(5, max_qubits + 1, 2):
        state_size = 2 ** num_qubits
        
        print(f"\nTesting {num_qubits} qubits ({state_size:,} states)...")
        print(f"  Memory estimate: ~{state_size * 16 / (1024**2):.2f} MB")
        
        tracemalloc.start()
        start_time = time.time()
        
        try:
            # Create simulator
            sim = GeometryQuantumSimulator(num_qubits)
            creation_time = time.time() - start_time
            
            # Apply a few gates
            gate_start = time.time()
            sim.hadamard(0)
            gate_time = time.time() - gate_start
            
            sim.cnot(0, 1)
            
            # Get probabilities
            prob_start = time.time()
            probs = sim.get_probabilities()
            prob_time = time.time() - prob_start
            
            # Check how many non-zero states
            non_zero = sum(1 for p in probs if p > 1e-10)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            total_time = time.time() - start_time
            
            results.append({
                'num_qubits': num_qubits,
                'state_size': state_size,
                'success': True,
                'creation_time': creation_time,
                'gate_time': gate_time,
                'prob_time': prob_time,
                'total_time': total_time,
                'memory_mb': peak / 1024 / 1024,
                'non_zero_states': non_zero
            })
            
            print(f"  ✅ Success!")
            print(f"  Creation: {creation_time:.3f}s")
            print(f"  Gate: {gate_time:.3f}s")
            print(f"  Prob: {prob_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
            print(f"  Memory: {peak/1024/1024:.2f} MB")
            print(f"  Non-zero states: {non_zero}")
            
        except MemoryError as e:
            tracemalloc.stop()
            print(f"  ❌ Memory Error: {e}")
            results.append({
                'num_qubits': num_qubits,
                'success': False,
                'error': 'MemoryError'
            })
            break
        except Exception as e:
            tracemalloc.stop()
            print(f"  ❌ Error: {e}")
            results.append({
                'num_qubits': num_qubits,
                'success': False,
                'error': str(e)
            })
            break
    
    # Summary
    print("\n" + "=" * 70)
    print("CAPACITY SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        max_qubits = max(r['num_qubits'] for r in successful)
        last = successful[-1]
        
        print(f"\n✅ Maximum qubits: {max_qubits}")
        print(f"   States: {last['state_size']:,}")
        print(f"   Memory: {last['memory_mb']:.2f} MB")
        print(f"   Gate time: {last['gate_time']:.3f}s")
        print(f"   Total time: {last['total_time']:.3f}s")
    
    return results


if __name__ == "__main__":
    test_capacity(max_qubits=20)

