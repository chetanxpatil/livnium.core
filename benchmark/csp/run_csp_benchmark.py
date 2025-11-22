"""
CSP Benchmark: Compare Livnium vs python-constraint

Runs CSPLib-style problems and compares results.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.csp.csp_solver_livnium import solve_csp_livnium, CSPProblem


def load_csp_from_json(json_path: Path) -> CSPProblem:
    """Load CSP problem from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return CSPProblem(
        variables=data['variables'],
        constraints=data['constraints']
    )


def solve_with_python_constraint(csp: CSPProblem, timeout: float = 60.0) -> Dict[str, Any]:
    """
    Solve using python-constraint library.
    
    Returns:
        Dictionary with results
    """
    try:
        from constraint import Problem
        
        start_time = time.time()
        
        # Create constraint problem
        problem = Problem()
        
        # Add variables
        for var_name, domain in csp.variables.items():
            problem.addVariable(var_name, domain)
        
        # Add constraints
        for constraint in csp.constraints:
            constraint_type = constraint.get('type', 'custom')
            vars_involved = constraint.get('vars', [])
            
            if constraint_type == 'all_different':
                problem.addConstraint(lambda *args: len(args) == len(set(args)), vars_involved)
            
            elif constraint_type == 'equal':
                problem.addConstraint(lambda *args: len(set(args)) == 1, vars_involved)
            
            elif constraint_type == 'not_equal':
                if len(vars_involved) == 2:
                    problem.addConstraint(lambda a, b: a != b, vars_involved)
                else:
                    problem.addConstraint(lambda *args: len(args) == len(set(args)), vars_involved)
            
            elif constraint_type == 'diagonal':
                # Diagonal constraint for N-Queens
                row_i = constraint.get('row_i')
                row_j = constraint.get('row_j')
                if row_i is not None and row_j is not None:
                    qi_var = f"Q{row_i+1}"
                    qj_var = f"Q{row_j+1}"
                    # Use default parameters to capture values (not references) in closure
                    problem.addConstraint(
                        lambda qi, qj, ri=row_i, rj=row_j: abs(qi - qj) != abs(ri - rj),
                        [qi_var, qj_var]
                    )
            
            elif constraint_type == 'custom':
                fn = constraint.get('fn')
                if fn:
                    problem.addConstraint(fn, vars_involved)
        
        # Solve
        solutions = problem.getSolutions()
        
        elapsed = time.time() - start_time
        
        assignment = None
        if solutions:
            assignment = solutions[0]  # Take first solution
        
        # Check constraint satisfaction
        all_satisfied, num_satisfied = csp.check_all_constraints(assignment) if assignment else (False, 0)
        
        return {
            'solved': True,
            'satisfiable': len(solutions) > 0,
            'assignment': assignment,
            'time': elapsed,
            'num_solutions': len(solutions),
            'num_satisfied_constraints': num_satisfied,
            'total_constraints': len(csp.constraints)
        }
    except ImportError:
        return {
            'solved': False,
            'error': 'python-constraint not installed. Install with: pip install python-constraint'
        }
    except Exception as e:
        return {
            'solved': False,
            'error': str(e)
        }


def run_benchmark(
    csp_files: List[Path],
    max_steps: int = 1000,
    max_time: float = 60.0,
    verbose: bool = False,
    use_recursive: bool = False,
    recursive_depth: int = 2
) -> Dict[str, Any]:
    """
    Run benchmark on a set of CSP files.
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'livnium': [],
        'python_constraint': [],
        'summary': {}
    }
    
    print(f"\n{'='*70}")
    print(f"CSP Benchmark: {len(csp_files)} problems")
    print(f"{'='*70}\n")
    
    for i, csp_file in enumerate(csp_files, 1):
        print(f"[{i}/{len(csp_files)}] {csp_file.name}")
        
        # Load CSP problem
        try:
            csp = load_csp_from_json(csp_file)
            print(f"  Variables: {csp.num_vars}, Constraints: {len(csp.constraints)}")
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
        
        # Solve with Livnium
        print("  Livnium...", end=" ", flush=True)
        livnium_result = solve_csp_livnium(
            csp,
            max_steps=max_steps,
            max_time=max_time,
            verbose=verbose,
            use_recursive=use_recursive,
            recursive_depth=recursive_depth
        )
        livnium_result['file'] = csp_file.name
        livnium_result['num_vars'] = csp.num_vars
        livnium_result['num_constraints'] = len(csp.constraints)
        results['livnium'].append(livnium_result)
        
        if livnium_result['solved']:
            print(f"✓ {livnium_result['time']:.3f}s, "
                  f"{livnium_result['num_satisfied_constraints']}/{livnium_result['total_constraints']} constraints")
        else:
            print("✗ Failed")
        
        # Solve with python-constraint
        print("  python-constraint...", end=" ", flush=True)
        constraint_result = solve_with_python_constraint(csp, timeout=max_time)
        constraint_result['file'] = csp_file.name
        constraint_result['num_vars'] = csp.num_vars
        constraint_result['num_constraints'] = len(csp.constraints)
        results['python_constraint'].append(constraint_result)
        
        if constraint_result.get('solved'):
            print(f"✓ {constraint_result['time']:.3f}s, "
                  f"{constraint_result.get('num_solutions', 0)} solutions")
        else:
            print(f"✗ {constraint_result.get('error', 'Failed')}")
        
        print()
    
    # Compute summary statistics
    livnium_solved = sum(1 for r in results['livnium'] if r.get('solved', False))
    constraint_solved = sum(1 for r in results['python_constraint'] if r.get('solved', False))
    
    livnium_times = [r['time'] for r in results['livnium'] if r.get('solved', False)]
    constraint_times = [r['time'] for r in results['python_constraint'] if r.get('solved', False)]
    
    results['summary'] = {
        'total_problems': len(csp_files),
        'livnium': {
            'solved': livnium_solved,
            'unsolved': len(csp_files) - livnium_solved,
            'avg_time': sum(livnium_times) / len(livnium_times) if livnium_times else None,
            'min_time': min(livnium_times) if livnium_times else None,
            'max_time': max(livnium_times) if livnium_times else None,
        },
        'python_constraint': {
            'solved': constraint_solved,
            'unsolved': len(csp_files) - constraint_solved,
            'avg_time': sum(constraint_times) / len(constraint_times) if constraint_times else None,
            'min_time': min(constraint_times) if constraint_times else None,
            'max_time': max(constraint_times) if constraint_times else None,
        }
    }
    
    return results


def print_summary(results: Dict[str, Any]):
    """Print benchmark summary."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    summary = results['summary']
    
    print(f"\nTotal Problems: {summary['total_problems']}")
    
    print(f"\n{'Solver':<20} {'Solved':<10} {'Unsolved':<10} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12}")
    print("-" * 70)
    
    liv = summary['livnium']
    avg_time_liv = f"{liv['avg_time']:.3f}s" if liv['avg_time'] is not None else "N/A"
    min_time_liv = f"{liv['min_time']:.3f}s" if liv['min_time'] is not None else "N/A"
    max_time_liv = f"{liv['max_time']:.3f}s" if liv['max_time'] is not None else "N/A"
    
    print(f"{'Livnium':<20} {liv['solved']:<10} {liv['unsolved']:<10} "
          f"{avg_time_liv:<12} {min_time_liv:<12} {max_time_liv:<12}")
    
    pc = summary['python_constraint']
    avg_time_pc = f"{pc['avg_time']:.3f}s" if pc['avg_time'] is not None else "N/A"
    min_time_pc = f"{pc['min_time']:.3f}s" if pc['min_time'] is not None else "N/A"
    max_time_pc = f"{pc['max_time']:.3f}s" if pc['max_time'] is not None else "N/A"
    
    print(f"{'python-constraint':<20} {pc['solved']:<10} {pc['unsolved']:<10} "
          f"{avg_time_pc:<12} {min_time_pc:<12} {max_time_pc:<12}")
    
    print("\n" + "="*70)


def save_results(results: Dict[str, Any], output_path: Path):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CSP Benchmark: Livnium vs python-constraint')
    parser.add_argument('--csp-dir', type=str, help='Directory containing CSP JSON files')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max search steps for Livnium')
    parser.add_argument('--max-time', type=float, default=60.0, help='Max time per problem (seconds)')
    parser.add_argument('--limit', type=int, help='Limit number of problems to test')
    parser.add_argument('--output', type=str, default='benchmark/csp/csp_results.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--use-recursive', action='store_true', default=False, help='Use recursive geometry (default: False)')
    parser.add_argument('--recursive-depth', type=int, default=2, help='Recursive geometry depth (default: 2)')
    
    args = parser.parse_args()
    
    # Determine CSP file directory
    if not args.csp_dir:
        benchmark_dir = Path(__file__).parent
        csp_dir = benchmark_dir / "csplib" / "test"
    else:
        csp_dir = Path(args.csp_dir)
    
    # Find CSP files
    csp_files = sorted(csp_dir.glob("*.json"))
    
    if not csp_files:
        print(f"No CSP JSON files found in {csp_dir}")
        print("Generate test problems first: python benchmark/csp/generate_test_csps.py")
        return
    
    if args.limit:
        csp_files = csp_files[:args.limit]
    
    # Run benchmark
    results = run_benchmark(
        csp_files,
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose,
        use_recursive=args.use_recursive,
        recursive_depth=args.recursive_depth
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)


if __name__ == '__main__':
    main()

