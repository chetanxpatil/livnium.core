"""
Demo: Entangled Basins - Idea A

Demonstrates deterministic correlation between two "machines" (simulated)
using shared seed and identical inputs.

This shows how Idea A works: same seed + same input → same basin signature.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.idea_a.entangled_basins import (
    initialize_shared_system,
    process_to_basin,
    verify_correlation,
    EntangledBasinsProcessor,
    CorrelationVerifier,
    BasinSignatureGenerator
)


def demo_basic_correlation():
    """Basic demo: two machines with same seed and input."""
    print("=" * 70)
    print("DEMO: Entangled Basins - Basic Correlation")
    print("=" * 70)
    print()
    
    # Shared seed
    seed = 42
    
    # Machine A
    print("Machine A:")
    print(f"  Initializing with seed={seed}...")
    system_a = initialize_shared_system(seed)
    input_text = "hello world"
    print(f"  Processing input: '{input_text}'")
    basin_a = process_to_basin(system_a, input_text, max_steps=50)
    hash_a = BasinSignatureGenerator.compute_basin_hash(basin_a)
    print(f"  Basin signature: {len(basin_a)} cells")
    print(f"  Basin hash: {hash_a}")
    print()
    
    # Machine B (same seed, same input)
    print("Machine B:")
    print(f"  Initializing with seed={seed}...")
    system_b = initialize_shared_system(seed)
    print(f"  Processing input: '{input_text}'")
    basin_b = process_to_basin(system_b, input_text, max_steps=50)
    hash_b = BasinSignatureGenerator.compute_basin_hash(basin_b)
    print(f"  Basin signature: {len(basin_b)} cells")
    print(f"  Basin hash: {hash_b}")
    print()
    
    # Verify correlation
    print("Correlation Check:")
    result = CorrelationVerifier.verify_correlation(basin_a, basin_b)
    print(f"  Correlated: {result.correlated}")
    print(f"  Match type: {result.match_details['match_type']}")
    print(f"  Length match: {result.match_details['length_match']}")
    print()
    
    if result.correlated:
        print("✅ SUCCESS: Both machines reached the same basin!")
        print("   This demonstrates deterministic correlation from shared seed.")
    else:
        print("⚠️  WARNING: Basins differ (may need more evolution steps)")
    
    print()


def demo_multiple_inputs():
    """Demo with multiple inputs to show consistent correlation."""
    print("=" * 70)
    print("DEMO: Multiple Inputs - Consistent Correlation")
    print("=" * 70)
    print()
    
    seed = 42
    inputs = ["hello", "world", "quantum", "geometry"]
    
    results = []
    for input_text in inputs:
        # Machine A
        system_a = initialize_shared_system(seed)
        basin_a = process_to_basin(system_a, input_text, max_steps=50)
        hash_a = BasinSignatureGenerator.compute_basin_hash(basin_a)
        
        # Machine B
        system_b = initialize_shared_system(seed)
        basin_b = process_to_basin(system_b, input_text, max_steps=50)
        hash_b = BasinSignatureGenerator.compute_basin_hash(basin_b)
        
        # Verify
        result = CorrelationVerifier.verify_correlation(basin_a, basin_b)
        results.append((input_text, result.correlated, hash_a == hash_b))
        
        status = "✅" if result.correlated else "❌"
        print(f"{status} '{input_text}': "
              f"Correlated={result.correlated}, "
              f"Hash match={hash_a == hash_b}")
    
    print()
    correlated_count = sum(1 for _, corr, _ in results if corr)
    print(f"Summary: {correlated_count}/{len(results)} inputs showed correlation")
    print()


def demo_determinism():
    """Demo to prove determinism: same seed + same input = same result."""
    print("=" * 70)
    print("DEMO: Determinism Proof")
    print("=" * 70)
    print()
    
    seed = 42
    input_text = "test determinism"
    
    # Run same process multiple times
    signatures = []
    for run in range(3):
        system = initialize_shared_system(seed)
        signature = process_to_basin(system, input_text, max_steps=50)
        hash_str = BasinSignatureGenerator.compute_basin_hash(signature)
        signatures.append((signature, hash_str))
        print(f"Run {run + 1}: Hash = {hash_str}")
    
    # Check all are identical
    all_same = all(s[1] == signatures[0][1] for s in signatures)
    print()
    if all_same:
        print("✅ PROOF: All runs produced identical results!")
        print("   This proves deterministic evolution.")
    else:
        print("⚠️  WARNING: Results differ (non-deterministic behavior detected)")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ENTANGLED BASINS - IDEA A DEMO")
    print("=" * 70)
    print("\nThis demonstrates classical hidden-variable model:")
    print("- Same seed + same input → same basin")
    print("- No communication needed during evolution")
    print("- Apparent 'non-local correlation' from shared structure")
    print()
    
    # Run demos
    demo_basic_correlation()
    demo_multiple_inputs()
    demo_determinism()
    
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)

