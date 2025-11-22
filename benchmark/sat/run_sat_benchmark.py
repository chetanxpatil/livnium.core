"""
SAT Benchmark: Compare Livnium vs PySAT/MiniSAT

Downloads SATLIB benchmarks and runs comparison.
"""

import os
import sys
import time
import json
import urllib.request
import ssl
import tarfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmark.sat.sat_solver_livnium import solve_sat_livnium, parse_cnf_file


SATLIB_URL = "https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html"
SATLIB_ARCHIVE_URL = "https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf20-91.tar.gz"


def download_satlib_sample(download_dir: Path) -> Path:
    """
    Download a sample of SATLIB benchmarks.
    
    For now, we'll use a small set. You can expand this later.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    
    archive_path = download_dir / "uf20-91.tar.gz"
    extract_dir = download_dir / "uf20-91"
    
    if extract_dir.exists() and any(extract_dir.glob("*.cnf")):
        print(f"SATLIB sample already downloaded at {extract_dir}")
        return extract_dir
    
    print(f"Downloading SATLIB sample from {SATLIB_ARCHIVE_URL}...")
    try:
        # Create SSL context that doesn't verify certificates (for macOS SSL issues)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Download with SSL context
        req = urllib.request.Request(SATLIB_ARCHIVE_URL)
        with urllib.request.urlopen(req, context=ssl_context) as response:
            with open(archive_path, 'wb') as out_file:
                out_file.write(response.read())
        
        print(f"Downloaded to {archive_path}")
        
        # Extract
        print(f"Extracting...")
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(download_dir)
        
        print(f"Extracted to {extract_dir}")
        return extract_dir
    except Exception as e:
        print(f"Error downloading SATLIB: {e}")
        print("\nYou can:")
        print("1. Manually download CNF files and place them in benchmark/satlib/")
        print("2. Generate test CNF files: python benchmark/generate_test_cnfs.py")
        print("3. Use --cnf-dir to specify a directory with CNF files")
        return None


def find_cnf_files(directory: Path) -> List[Path]:
    """Find all CNF files in a directory."""
    cnf_files = list(directory.glob("*.cnf"))
    return sorted(cnf_files)


def solve_with_pysat(cnf_path: Path, timeout: float = 60.0) -> Dict[str, Any]:
    """
    Solve using PySAT (wrapper around MiniSAT).
    
    Returns:
        Dictionary with results
    """
    try:
        from pysat.solvers import Glucose3
        from pysat.formula import CNF
        
        start_time = time.time()
        
        # Parse CNF
        cnf = CNF(from_file=str(cnf_path))
        
        # Solve
        solver = Glucose3(cnf)
        is_sat = solver.solve()
        
        elapsed = time.time() - start_time
        
        assignment = None
        if is_sat:
            model = solver.get_model()
            # Convert to dict: variable_id -> bool
            assignment = {}
            for lit in model:
                var_id = abs(lit)
                assignment[var_id] = lit > 0
        
        solver.delete()
        
        return {
            'solved': True,
            'satisfiable': is_sat,
            'assignment': assignment,
            'time': elapsed,
            'steps': None,  # PySAT doesn't expose step count
            'num_satisfied_clauses': len(cnf.clauses) if is_sat else 0,
            'total_clauses': len(cnf.clauses)
        }
    except ImportError:
        return {
            'solved': False,
            'error': 'PySAT not installed. Install with: pip install python-sat'
        }
    except Exception as e:
        return {
            'solved': False,
            'error': str(e)
        }


def run_benchmark(
    cnf_files: List[Path],
    max_steps: int = 500,
    max_time: float = 60.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run benchmark on a set of CNF files.
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        'livnium': [],
        'pysat': [],
        'summary': {}
    }
    
    print(f"\n{'='*70}")
    print(f"SAT Benchmark: {len(cnf_files)} CNF files")
    print(f"{'='*70}\n")
    
    for i, cnf_path in enumerate(cnf_files, 1):
        print(f"[{i}/{len(cnf_files)}] {cnf_path.name}")
        
        # Parse to get problem size
        try:
            num_vars, clauses = parse_cnf_file(str(cnf_path))
            print(f"  Variables: {num_vars}, Clauses: {len(clauses)}")
        except Exception as e:
            print(f"  Error parsing: {e}")
            continue
        
        # Solve with Livnium
        print("  Livnium...", end=" ", flush=True)
        livnium_result = solve_sat_livnium(
            str(cnf_path),
            max_steps=max_steps,
            max_time=max_time,
            verbose=False
        )
        livnium_result['file'] = cnf_path.name
        livnium_result['num_vars'] = num_vars
        livnium_result['num_clauses'] = len(clauses)
        results['livnium'].append(livnium_result)
        
        if livnium_result['solved']:
            print(f"✓ {livnium_result['time']:.3f}s, "
                  f"{livnium_result['num_satisfied_clauses']}/{livnium_result['total_clauses']} clauses")
        else:
            print("✗ Failed")
        
        # Solve with PySAT
        print("  PySAT...", end=" ", flush=True)
        pysat_result = solve_with_pysat(cnf_path, timeout=max_time)
        pysat_result['file'] = cnf_path.name
        pysat_result['num_vars'] = num_vars
        pysat_result['num_clauses'] = len(clauses)
        results['pysat'].append(pysat_result)
        
        if pysat_result.get('solved'):
            print(f"✓ {pysat_result['time']:.3f}s")
        else:
            print(f"✗ {pysat_result.get('error', 'Failed')}")
        
        print()
    
    # Compute summary statistics
    livnium_solved = sum(1 for r in results['livnium'] if r.get('solved', False))
    pysat_solved = sum(1 for r in results['pysat'] if r.get('solved', False))
    
    livnium_times = [r['time'] for r in results['livnium'] if r.get('solved', False)]
    pysat_times = [r['time'] for r in results['pysat'] if r.get('solved', False)]
    
    results['summary'] = {
        'total_problems': len(cnf_files),
        'livnium': {
            'solved': livnium_solved,
            'unsolved': len(cnf_files) - livnium_solved,
            'avg_time': sum(livnium_times) / len(livnium_times) if livnium_times else None,
            'min_time': min(livnium_times) if livnium_times else None,
            'max_time': max(livnium_times) if livnium_times else None,
        },
        'pysat': {
            'solved': pysat_solved,
            'unsolved': len(cnf_files) - pysat_solved,
            'avg_time': sum(pysat_times) / len(pysat_times) if pysat_times else None,
            'min_time': min(pysat_times) if pysat_times else None,
            'max_time': max(pysat_times) if pysat_times else None,
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
    
    print(f"\n{'Solver':<15} {'Solved':<10} {'Unsolved':<10} {'Avg Time':<12} {'Min Time':<12} {'Max Time':<12}")
    print("-" * 70)
    
    liv = summary['livnium']
    print(f"{'Livnium':<15} {liv['solved']:<10} {liv['unsolved']:<10} "
          f"{liv['avg_time']:.3f}s{' ' if liv['avg_time'] else 'N/A':<8} "
          f"{liv['min_time']:.3f}s{' ' if liv['min_time'] else 'N/A':<8} "
          f"{liv['max_time']:.3f}s{' ' if liv['max_time'] else 'N/A':<8}")
    
    pysat = summary['pysat']
    print(f"{'PySAT':<15} {pysat['solved']:<10} {pysat['unsolved']:<10} "
          f"{pysat['avg_time']:.3f}s{' ' if pysat['avg_time'] else 'N/A':<8} "
          f"{pysat['min_time']:.3f}s{' ' if pysat['min_time'] else 'N/A':<8} "
          f"{pysat['max_time']:.3f}s{' ' if pysat['max_time'] else 'N/A':<8}")
    
    print("\n" + "="*70)


def save_results(results: Dict[str, Any], output_path: Path):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='SAT Benchmark: Livnium vs PySAT')
    parser.add_argument('--download', action='store_true', help='Download SATLIB benchmarks')
    parser.add_argument('--cnf-dir', type=str, help='Directory containing CNF files')
    parser.add_argument('--max-steps', type=int, default=500, help='Max search steps for Livnium')
    parser.add_argument('--max-time', type=float, default=60.0, help='Max time per problem (seconds)')
    parser.add_argument('--limit', type=int, help='Limit number of problems to test')
    parser.add_argument('--output', type=str, default='benchmark/sat/sat_results.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Determine CNF file directory
    benchmark_dir = Path(__file__).parent
    satlib_dir = benchmark_dir / "satlib"
    
    if args.download or not args.cnf_dir:
        print("Downloading SATLIB sample...")
        cnf_dir = download_satlib_sample(satlib_dir)
        if cnf_dir is None:
            print("Failed to download. Please provide --cnf-dir or manually download CNF files.")
            return
    else:
        cnf_dir = Path(args.cnf_dir)
    
    # Find CNF files
    cnf_files = find_cnf_files(cnf_dir)
    
    if not cnf_files:
        print(f"No CNF files found in {cnf_dir}")
        return
    
    if args.limit:
        cnf_files = cnf_files[:args.limit]
    
    # Run benchmark
    results = run_benchmark(
        cnf_files,
        max_steps=args.max_steps,
        max_time=args.max_time,
        verbose=args.verbose
    )
    
    # Print summary
    print_summary(results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)


if __name__ == '__main__':
    main()

