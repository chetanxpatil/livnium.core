#!/usr/bin/env python3
"""
THE REAL TEST: Maximum Entanglement

This is the test that will reveal the truth:
- Hadamard on ALL 500 qubits = uniform superposition (2^500 states)
- CNOT on ALL 499 adjacent pairs = maximum entanglement chain
- Bond dimension will explode from χ=8 to potentially χ=2^250
- This will either: crash, run forever, or give wrong answers
"""


from quantum.hierarchical.simulators.mps_hierarchical_simulator import MPSHierarchicalGeometrySimulator
import time
import tracemalloc

print("=" * 70)
print("THE REAL TEST: Maximum Entanglement on 500 Qubits")
print("=" * 70)
print("\nThis test will:")
print("  1. Create uniform superposition on ALL 500 qubits")
print("  2. Create maximum entanglement chain (CNOT on all pairs)")
print("  3. This is the WORST-CASE scenario for MPS")
print("\nExpected outcome:")
print("  - Bond dimension will explode")
print("  - Memory will explode")
print("  - Will either: CRASH, RUN FOREVER, or GIVE WRONG ANSWER")
print("\n" + "=" * 70)

tracemalloc.start()
start_time = time.time()

try:
    sim = MPSHierarchicalGeometrySimulator(500, bond_dimension=8)
    
    print("\nStep 1: Hadamard on ALL 500 qubits...")
    print("  (This creates uniform superposition = 2^500 states)")
    
    for i in range(500):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            print(f"  Progress: {i}/500 qubits | Time: {elapsed:.2f}s | Memory: {peak/1024/1024:.2f} MB")
        sim.hadamard(i)
    
    print("\nStep 2: CNOT on ALL 499 adjacent pairs...")
    print("  (This creates maximum entanglement chain)")
    print("  (Bond dimension will EXPLODE)")
    
    for i in range(499):
        if i % 50 == 0:
            elapsed = time.time() - start_time
            current, peak = tracemalloc.get_traced_memory()
            print(f"  Progress: {i}/499 pairs | Time: {elapsed:.2f}s | Memory: {peak/1024/1024:.2f} MB")
        sim.cnot(i, i+1)
    
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    
    print("\n" + "=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Peak memory: {peak/1024/1024:.2f} MB")
    
    info = sim.get_capacity_info()
    print(f"  Final memory: {info['memory_mb']:.2f} MB")
    print(f"  Bond dimension: {info['bond_dimension']}")
    
    print("\n⚠️  If this completed quickly with low memory,")
    print("   it means the MPS truncated (cut off) entanglement.")
    print("   The answer is WRONG - it's an approximation.")
    
except MemoryError as e:
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("\n" + "=" * 70)
    print("❌ CRASHED: Out of Memory")
    print("=" * 70)
    print(f"  Time before crash: {elapsed:.2f} seconds")
    print(f"  Peak memory: {peak/1024/1024:.2f} MB")
    print(f"  Error: {e}")
    print("\n  This is what happens when bond dimension explodes.")
    print("  MPS cannot handle maximum entanglement.")
    
except KeyboardInterrupt:
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("\n" + "=" * 70)
    print("⏸️  INTERRUPTED: Taking too long")
    print("=" * 70)
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Peak memory: {peak/1024/1024:.2f} MB")
    print("\n  This is exponential time - MPS cannot handle this.")
    
except Exception as e:
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print("\n" + "=" * 70)
    print("❌ ERROR")
    print("=" * 70)
    print(f"  Time: {elapsed:.2f} seconds")
    print(f"  Peak memory: {peak/1024/1024:.2f} MB")
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    tracemalloc.stop()

