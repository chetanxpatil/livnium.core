"""
Quick test of SAT solver with a simple CNF formula.

Tests the SAT encoding and solving pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.sat.sat_solver_livnium import solve_sat_livnium, parse_cnf_file


def create_test_cnf():
    """Create a simple test CNF file."""
    # Simple 3-SAT: (x1 OR x2 OR x3) AND (NOT x1 OR x2 OR NOT x3) AND (x1 OR NOT x2 OR x3)
    # This is satisfiable, e.g., x1=True, x2=True, x3=True
    
    cnf_content = """c Simple test CNF
p cnf 3 3
1 2 3 0
-1 2 -3 0
1 -2 3 0
"""
    
    test_file = Path(__file__).parent / "test.cnf"
    with open(test_file, 'w') as f:
        f.write(cnf_content)
    
    return test_file


def main():
    print("="*60)
    print("Testing SAT Solver")
    print("="*60)
    
    # Create test CNF
    test_cnf = create_test_cnf()
    print(f"\nCreated test CNF: {test_cnf}")
    
    # Parse it
    num_vars, clauses = parse_cnf_file(str(test_cnf))
    print(f"  Variables: {num_vars}")
    print(f"  Clauses: {len(clauses)}")
    print(f"  Clauses: {clauses}")
    
    # Solve with Livnium
    print("\nSolving with Livnium...")
    result = solve_sat_livnium(
        str(test_cnf),
        max_steps=200,
        max_time=30.0,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Solved: {result['solved']}")
    print(f"Satisfiable: {result['satisfiable']}")
    print(f"Time: {result['time']:.3f}s")
    print(f"Steps: {result['steps']}")
    print(f"Satisfied clauses: {result['num_satisfied_clauses']}/{result['total_clauses']}")
    
    if result['assignment']:
        print(f"\nAssignment:")
        for var_id in sorted(result['assignment'].keys()):
            print(f"  x{var_id} = {result['assignment'][var_id]}")
    
    # Cleanup
    test_cnf.unlink()
    print(f"\nCleaned up {test_cnf}")


if __name__ == '__main__':
    main()

