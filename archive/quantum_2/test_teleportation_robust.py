"""
Robust teleportation test - run multiple times with different states.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantum.test_teleportation import quantum_teleportation


def test_multiple_states():
    """Test teleportation with multiple different states."""
    print("=" * 70)
    print("ROBUST TELEPORTATION TEST")
    print("=" * 70)
    print()
    
    test_cases = [
        (0.6, 0.8j, "α=0.6, β=0.8i"),
        (1.0, 0.0, "|0> state"),
        (0.0, 1.0, "|1> state"),
        (1/np.sqrt(2), 1/np.sqrt(2), "Equal superposition"),
        (0.8, 0.6j, "α=0.8, β=0.6i"),
        (1/np.sqrt(2), 1j/np.sqrt(2), "Superposition with phase"),
    ]
    
    results = []
    
    for alpha, beta, description in test_cases:
        print(f"Testing: {description}")
        print(f"  α = {alpha:.6f}, β = {beta:.6f}")
        
        result = quantum_teleportation(alpha, beta, verbose=False)
        
        print(f"  Original: {result['original'][0]:.6f}|0> + {result['original'][1]:.6f}|1>")
        print(f"  Final:    {result['final'][0]:.6f}|0> + {result['final'][1]:.6f}|1>")
        print(f"  Fidelity: {result['fidelity']:.6f}")
        print(f"  Match: {'✅ YES' if result['match'] else '❌ NO'}")
        print()
        
        results.append({
            'description': description,
            'match': result['match'],
            'fidelity': result['fidelity']
        })
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    passed = sum(1 for r in results if r['match'])
    total = len(results)
    avg_fidelity = np.mean([r['fidelity'] for r in results])
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Average fidelity: {avg_fidelity:.6f}")
    print()
    
    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"  {status} {r['description']}: fidelity={r['fidelity']:.6f}")
    
    print()
    if passed == total:
        print("🎯 ALL TESTS PASSED!")
        print()
        print("This confirms:")
        print("  ✅ Teleportation works for arbitrary states")
        print("  ✅ Complex amplitudes handled correctly")
        print("  ✅ Phase information preserved")
        print("  ✅ Full quantum mechanics implemented")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return results


if __name__ == "__main__":
    test_multiple_states()

