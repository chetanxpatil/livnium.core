"""
Test Entanglement Capacity: How Many Qubits Can We Entangle?

Tests the entanglement system's capacity with increasing numbers of entangled pairs.
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
from core.quantum.quantum_gates import GateType
import math


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def test_entanglement_capacity(n_qubits: int, max_pairs: int = None, verbose: bool = True) -> dict:
    """
    Test entanglement capacity with n_qubits.
    
    Args:
        n_qubits: Number of qubits
        max_pairs: Maximum pairs to create (default: n_qubits // 2)
        verbose: Print progress
        
    Returns:
        Test results dictionary
    """
    if max_pairs is None:
        max_pairs = n_qubits // 2
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing entanglement: {n_qubits} qubits, {max_pairs} pairs")
        print(f"{'='*60}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Create system
        n = math.ceil(n_qubits ** (1/3))
        if n % 2 == 0:
            n += 1
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
        
        core = LivniumCoreSystem(config)
        qlattice = QuantumLattice(core)
        
        init_time = time.time() - start_time
        init_memory = get_memory_usage() - start_memory
        
        if verbose:
            print(f"  Initialization: {init_time:.3f}s, {init_memory:.2f} MB")
            print(f"  Actual qubits: {len(qlattice.quantum_cells)}")
        
        # Get coordinates list
        coords_list = list(qlattice.quantum_cells.keys())
        available_pairs = min(max_pairs, len(coords_list) // 2)
        
        # Test entanglement creation
        if verbose:
            print(f"  Creating {available_pairs} entangled pairs...")
        
        entangle_start = time.time()
        successful_pairs = 0
        failed_pairs = 0
        
        for i in range(0, available_pairs * 2, 2):
            if i + 1 >= len(coords_list):
                break
            try:
                qlattice.entangle_cells(coords_list[i], coords_list[i+1])
                successful_pairs += 1
            except Exception as e:
                failed_pairs += 1
                if verbose and failed_pairs <= 3:
                    print(f"    Error creating pair {i//2}: {e}")
        
        entangle_time = time.time() - entangle_start
        
        # Get entanglement statistics
        if qlattice.entanglement_manager:
            ent_stats = qlattice.entanglement_manager.get_entanglement_statistics()
        else:
            ent_stats = {}
        
        # Test measurement on entangled qubits
        if verbose:
            print(f"  Testing measurement on entangled qubits...")
        measure_start = time.time()
        measure_count = 0
        
        for i in range(0, min(10, successful_pairs * 2), 2):
            if i + 1 < len(coords_list):
                try:
                    result1 = qlattice.measure_cell(coords_list[i])
                    result2 = qlattice.measure_cell(coords_list[i+1])
                    measure_count += 1
                except Exception as e:
                    if verbose:
                        print(f"    Error measuring pair {i//2}: {e}")
        
        measure_time = time.time() - measure_start
        
        # Final memory
        end_memory = get_memory_usage()
        total_memory = end_memory - start_memory
        
        total_time = time.time() - start_time
        
        results = {
            'n_qubits': n_qubits,
            'actual_qubits': len(qlattice.quantum_cells),
            'successful_pairs': successful_pairs,
            'failed_pairs': failed_pairs,
            'entangle_time': entangle_time,
            'time_per_pair': entangle_time / successful_pairs if successful_pairs > 0 else 0,
            'measure_count': measure_count,
            'measure_time': measure_time,
            'entanglement_stats': ent_stats,
            'total_memory_mb': total_memory,
            'memory_per_pair_mb': total_memory / successful_pairs if successful_pairs > 0 else 0,
            'total_time': total_time,
            'success': successful_pairs > 0,
        }
        
        if verbose:
            print(f"  ✅ Created {successful_pairs} entangled pairs in {entangle_time:.3f}s")
            print(f"  Time per pair: {results['time_per_pair']*1000:.3f} ms")
            print(f"  Total memory: {total_memory:.2f} MB")
            print(f"  Memory per pair: {results['memory_per_pair_mb']:.4f} MB")
            if ent_stats:
                print(f"  Entanglement info: {ent_stats}")
        
        return results
        
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


def test_max_entangled_pairs(n_qubits: int, verbose: bool = True) -> dict:
    """
    Test maximum number of entangled pairs we can create.
    
    Args:
        n_qubits: Number of qubits
        verbose: Print progress
        
    Returns:
        Test results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing MAX entanglement: {n_qubits} qubits")
        print(f"{'='*60}")
    
    # Test with maximum pairs (all qubits paired)
    max_pairs = n_qubits // 2
    return test_entanglement_capacity(n_qubits, max_pairs=max_pairs, verbose=verbose)


def test_entanglement_chain(n_qubits: int, verbose: bool = True) -> dict:
    """
    Test creating a chain of entangled qubits.
    
    Args:
        n_qubits: Number of qubits
        verbose: Print progress
        
    Returns:
        Test results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing entanglement CHAIN: {n_qubits} qubits")
        print(f"{'='*60}")
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        # Create system
        n = math.ceil(n_qubits ** (1/3))
        if n % 2 == 0:
            n += 1
        if n < 3:
            n = 3
        
        config = LivniumCoreConfig(
            lattice_size=n,
            enable_quantum=True,
            enable_superposition=True,
            enable_quantum_gates=True,
            enable_entanglement=True,
        )
        
        core = LivniumCoreSystem(config)
        qlattice = QuantumLattice(core)
        
        coords_list = list(qlattice.quantum_cells.keys())
        
        # Create chain: qubit 0 <-> qubit 1 <-> qubit 2 <-> ... <-> qubit N
        if verbose:
            print(f"  Creating chain of {len(coords_list)} qubits...")
        
        chain_start = time.time()
        chain_length = 0
        
        for i in range(len(coords_list) - 1):
            try:
                qlattice.entangle_cells(coords_list[i], coords_list[i+1])
                chain_length += 1
            except Exception as e:
                if verbose:
                    print(f"    Error at link {i}: {e}")
                break
        
        chain_time = time.time() - chain_start
        
        # Get statistics
        if qlattice.entanglement_manager:
            ent_stats = qlattice.entanglement_manager.get_entanglement_statistics()
        else:
            ent_stats = {}
        
        end_memory = get_memory_usage()
        total_memory = end_memory - start_memory
        
        results = {
            'n_qubits': n_qubits,
            'actual_qubits': len(qlattice.quantum_cells),
            'chain_length': chain_length,
            'chain_time': chain_time,
            'time_per_link': chain_time / chain_length if chain_length > 0 else 0,
            'entanglement_stats': ent_stats,
            'total_memory_mb': total_memory,
            'total_time': time.time() - start_time,
            'success': chain_length > 0,
        }
        
        if verbose:
            print(f"  ✅ Created chain of {chain_length} links in {chain_time:.3f}s")
            print(f"  Time per link: {results['time_per_link']*1000:.3f} ms")
            print(f"  Total memory: {total_memory:.2f} MB")
        
        return results
        
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


def run_entanglement_tests():
    """Run comprehensive entanglement tests."""
    print("="*60)
    print("ENTANGLEMENT CAPACITY TEST")
    print("="*60)
    print()
    
    test_counts = [100, 500, 1000, 5000, 10000]
    results = []
    
    print("Test 1: Maximum Entangled Pairs")
    print("-"*60)
    for n in test_counts:
        result = test_max_entangled_pairs(n, verbose=False)
        results.append(result)
        
        if result.get('success', False):
            pairs = result.get('successful_pairs', 0)
            time_per_pair = result.get('time_per_pair', 0) * 1000
            memory = result.get('total_memory_mb', 0)
            print(f"  {n:5d} qubits: {pairs:5d} pairs, {time_per_pair:6.3f} ms/pair, {memory:7.2f} MB")
        else:
            print(f"  {n:5d} qubits: ❌ Failed - {result.get('error', 'Unknown')}")
            break
    
    print()
    print("Test 2: Entanglement Chain")
    print("-"*60)
    chain_results = []
    for n in test_counts:
        result = test_entanglement_chain(n, verbose=False)
        chain_results.append(result)
        
        if result.get('success', False):
            chain_len = result.get('chain_length', 0)
            time_per_link = result.get('time_per_link', 0) * 1000
            memory = result.get('total_memory_mb', 0)
            print(f"  {n:5d} qubits: {chain_len:5d} links, {time_per_link:6.3f} ms/link, {memory:7.2f} MB")
        else:
            print(f"  {n:5d} qubits: ❌ Failed - {result.get('error', 'Unknown')}")
            break
    
    # Summary
    print()
    print("="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        max_result = max(successful, key=lambda x: x['n_qubits'])
        print(f"✅ Maximum entangled pairs: {max_result['successful_pairs']} pairs")
        print(f"   At {max_result['n_qubits']} qubits")
        print(f"   Time per pair: {max_result['time_per_pair']*1000:.3f} ms")
        print(f"   Memory: {max_result['total_memory_mb']:.2f} MB")
    
    successful_chains = [r for r in chain_results if r.get('success', False)]
    if successful_chains:
        max_chain = max(successful_chains, key=lambda x: x['n_qubits'])
        print(f"✅ Maximum chain length: {max_chain['chain_length']} links")
        print(f"   At {max_chain['n_qubits']} qubits")
        print(f"   Time per link: {max_chain['time_per_link']*1000:.3f} ms")
    
    return results, chain_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test entanglement capacity")
    parser.add_argument('--qubits', type=int, default=1000, help='Number of qubits')
    parser.add_argument('--pairs', type=int, default=None, help='Number of pairs to create')
    parser.add_argument('--chain', action='store_true', help='Test entanglement chain')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    
    args = parser.parse_args()
    
    if args.full:
        run_entanglement_tests()
    elif args.chain:
        test_entanglement_chain(args.qubits, verbose=True)
    else:
        test_entanglement_capacity(args.qubits, max_pairs=args.pairs, verbose=True)

