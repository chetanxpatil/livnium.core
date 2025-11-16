#!/usr/bin/env python3
"""
Test script for Ramsey Number Solver

Tests the solver on known Ramsey number problems.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# Add archive path for hierarchical system
archive_path = project_root / "archive" / "pre_core_systems"
sys.path.insert(0, str(archive_path))

from ramsey_number_solver import solve_ramsey_problem


def test_known_problems():
    """Test on known Ramsey number problems."""
    
    print("\n" + "=" * 70)
    print("RAMSEY NUMBER SOLVER - Test Suite")
    print("=" * 70)
    
    # Test cases: (n, k) where we try to prove R(k,k) > n
    test_cases = [
        (10, 3),  # R(3,3) = 6, so R(3,3) > 10 is false, but good for testing
        (20, 4),  # R(4,4) = 18, so R(4,4) > 20 is false, but good for testing
        (40, 5),  # R(5,5) is unknown, but known to be between 43-48
        (45, 5),  # Challenging case
    ]
    
    results = []
    
    for n, k in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Testing: R({k},{k}) > {n}?")
        print(f"{'=' * 70}")
        
        try:
            result = solve_ramsey_problem(n, k, num_omcubes=5000)
            results.append({
                'n': n,
                'k': k,
                'success': result is not None,
                'result': result
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'n': n,
                'k': k,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} R({result['k']},{result['k']}) > {result['n']}: "
              f"{'FOUND' if result['success'] else 'NOT FOUND'}")
    
    return results


if __name__ == '__main__':
    test_known_problems()

