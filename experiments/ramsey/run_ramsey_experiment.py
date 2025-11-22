"""
Run Ramsey Experiment: Solve Ramsey Problems with Dynamic Basin Search

This uses ONLY Dynamic Basin Search (single-universe constraint descent).
No MultiBasin interference - Ramsey is a one-universe problem.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Dict, Any
import argparse

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
import importlib

# Import with space in module name
encoder_module = importlib.import_module('core.encoder.constraint_encoder')
ConstraintEncoder = encoder_module.ConstraintEncoder

# Handle relative imports
try:
    from .ramsey_encoder import RamseyEncoder
    from .ramsey_tension import (
        count_monochromatic_k4,
        compute_ramsey_tension,
        is_valid_ramsey_coloring,
        ramsey_score,
    )
    from .ramsey_dynamic_search import solve_ramsey_dynamic
except ImportError:
    from ramsey_encoder import RamseyEncoder
    from ramsey_tension import (
        count_monochromatic_k4,
        compute_ramsey_tension,
        is_valid_ramsey_coloring,
        ramsey_score,
    )
    from ramsey_dynamic_search import solve_ramsey_dynamic


def solve_ramsey(
    n_vertices: int,
    n_candidates: int = 20,  # Not used (kept for compatibility)
    max_steps: int = 1000,
    verbose: bool = True,
    use_checkpoint: bool = True,
    checkpoint_interval: int = 500,
    visualize: bool = False,
    visualize_interval: int = 50
) -> Dict[str, Any]:
    """
    Solve Ramsey problem for Kₙ using Dynamic Basin Search.
    
    This is the correct approach: single-universe constraint descent.
    No MultiBasin - Ramsey is a one-universe problem.
    
    Args:
        n_vertices: Number of vertices (n for Kₙ)
        n_candidates: Not used (kept for compatibility)
        max_steps: Maximum search steps
        verbose: Print progress
        
    Returns:
        Results dictionary
    """
    if verbose:
        print("="*70)
        print(f"Ramsey Problem: K_{n_vertices}")
        print("="*70)
        print()
    
    # Estimate lattice size needed
    # For Kₙ, we need n(n-1)/2 edges
    n_edges = n_vertices * (n_vertices - 1) // 2
    n_lattice = max(3, int(np.ceil((n_edges) ** (1/3))))
    if n_lattice % 2 == 0:
        n_lattice += 1  # Must be odd
    
    if verbose:
        print(f"  Vertices: {n_vertices}")
        print(f"  Edges: {n_edges}")
        print(f"  Lattice size: {n_lattice}×{n_lattice}×{n_lattice} = {n_lattice**3} cells")
        print()
    
    # Create system
    config = LivniumCoreConfig(
        lattice_size=n_lattice,
        enable_semantic_polarity=True
    )
    system = LivniumCoreSystem(config)
    
    # Create Ramsey encoder
    encoder = RamseyEncoder(system, n_vertices)
    
    # Choose constraint type: K₃ for R(3,3) (n <= 6), K₄ for R(4,4) (n >= 17)
    constraint_type = "k3" if n_vertices <= 6 else "k4"
    
    # Encode constraints as tension fields
    if constraint_type == "k3":
        if verbose:
            print(f"  Encoding {len(encoder.k3_subsets)} K₃ constraints (triangles)...")
        tension_fields = encoder.encode_k3_constraints()
    else:
        if verbose:
            print(f"  Encoding {len(encoder.k4_subsets)} K₄ constraints...")
        tension_fields = encoder.encode_k4_constraints()
    
    if verbose:
        print(f"  Created {len(tension_fields)} tension fields")
        print()
    
    # Setup checkpoint path
    checkpoint_path = None
    if use_checkpoint:
        try:
            from .ramsey_checkpoint import get_checkpoint_path, should_resume
        except ImportError:
            from ramsey_checkpoint import get_checkpoint_path, should_resume
        
        checkpoint_path = str(get_checkpoint_path(n_vertices, constraint_type))
        
        if should_resume(get_checkpoint_path(n_vertices, constraint_type), max_steps):
            if verbose:
                print(f"  📍 Checkpoint found - will resume from saved progress")
    
    # Solve using Dynamic Basin Search (single-universe descent)
    result = solve_ramsey_dynamic(
        system=system,
        encoder=encoder,
        max_steps=max_steps,
        verbose=verbose,
        constraint_type=constraint_type,
        checkpoint_path=checkpoint_path,
        save_checkpoint_interval=checkpoint_interval,
        visualize=visualize,
        visualize_interval=visualize_interval
    )
    
    # Extract results
    final_coloring = result['coloring']
    num_violations = result['violations']
    ramsey_tension = result['tension']
    steps = result['steps']
    is_valid = result['solved']
    total_tension = encoder.constraint_encoder.get_total_tension(system)
    score = ramsey_score(final_coloring, encoder.vertices, constraint_type=constraint_type)
    
    results = {
        'n_vertices': n_vertices,
        'n_edges': n_edges,
        'constraint_type': constraint_type,
        'n_k3s': len(encoder.k3_subsets) if constraint_type == 'k3' else 0,
        'n_k4s': len(encoder.k4_subsets) if constraint_type == 'k4' else 0,
        'solved': is_valid,
        'num_violations': num_violations,
        'ramsey_tension': ramsey_tension,
        'total_tension': total_tension,
        'ramsey_score': score,
        'steps': steps,
        'coloring': final_coloring,
        'best_violations': result.get('best_violations', num_violations),
        'steps_to_best': result.get('steps_to_best', steps)
    }
    
    if verbose:
        print()
        print("="*70)
        print("RESULTS")
        print("="*70)
        # Calculate percentage satisfied
        if constraint_type == 'k3':
            total_constraints = len(encoder.k3_subsets)
            constraint_name = "K₃s (triangles)"
        else:
            total_constraints = len(encoder.k4_subsets)
            constraint_name = "K₄s"
        
        if total_constraints > 0:
            satisfied = total_constraints - num_violations
            percent_satisfied = (satisfied / total_constraints) * 100.0
            best_satisfied = total_constraints - result.get('best_violations', num_violations)
            best_percent = (best_satisfied / total_constraints) * 100.0
        else:
            percent_satisfied = 0.0
            best_percent = 0.0
        
        print(f"  Valid coloring: {is_valid}")
        print(f"  Satisfaction: {percent_satisfied:.2f}% ({satisfied}/{total_constraints} {constraint_name} satisfied)")
        if 'best_violations' in result and result['best_violations'] < num_violations:
            print(f"  Best achieved: {best_percent:.2f}% (at step {result['steps_to_best']})")
        print(f"  Steps: {steps}")
        print()
        
        if is_valid:
            if constraint_type == 'k3':
                print("  ✓ SUCCESS: Found valid 2-coloring with no monochromatic triangle")
            else:
                print("  ✓ SUCCESS: Found valid 2-coloring with no monochromatic K₄")
        else:
            if constraint_type == 'k3':
                print(f"  ⚠️  Found coloring with {num_violations} monochromatic triangle(s)")
            else:
                print(f"  ⚠️  Found coloring with {num_violations} monochromatic K₄(s)")
            if n_vertices == 5:
                print("  Expected: K₅ should be colorable (R(3,3) = 6)")
            elif n_vertices == 6:
                print("  Expected: K₆ must have monochromatic triangle (R(3,3) = 6)")
        print()
    
    return results


def test_ramsey_r33():
    """Test R(3,3) = 6: K₅ should succeed, K₆ should fail."""
    print("\n" + "="*70)
    print("Ramsey R(3,3) Test")
    print("="*70)
    print()
    print("R(3,3) = 6 means:")
    print("  - K₅ can be 2-colored with no monochromatic triangle")
    print("  - K₆ must contain a monochromatic triangle")
    print()
    
    # Import meta-search
    try:
        from .ramsey_meta_search import solve_ramsey_meta
    except ImportError:
        from ramsey_meta_search import solve_ramsey_meta
    
    # Test K₅ (should succeed) - use meta-search with K₃ constraints for R(3,3)
    print("Testing K₅ (should find valid coloring with no monochromatic triangle)...")
    result_k5 = solve_ramsey_meta(
        n_vertices=5,
        n_candidates=10,
        inner_steps=1000,
        outer_iterations=20,
        verbose=False,
        constraint_type='k3'  # R(3,3) uses K₃ (triangles), not K₄
    )
    
    print(f"  K₅: Valid={result_k5['solved']}, Violations={result_k5['violations']}")
    if result_k5['solved']:
        print("  ✓ PASS: K₅ correctly colored (no monochromatic triangle)")
    else:
        print(f"  ⚠️  K₅ should be colorable (found {result_k5['violations']} monochromatic triangle(s))")
    
    print()
    
    # Test K₆ (should fail - must have monochromatic triangle) - use meta-search with K₃
    print("Testing K₆ (should have monochromatic triangle)...")
    result_k6 = solve_ramsey_meta(
        n_vertices=6,
        n_candidates=10,
        inner_steps=1000,
        outer_iterations=20,
        verbose=False,
        constraint_type='k3'  # R(3,3) uses K₃ (triangles)
    )
    
    print(f"  K₆: Valid={result_k6['solved']}, Violations={result_k6['violations']}")
    if not result_k6['solved']:
        print("  ✓ PASS: K₆ correctly has monochromatic triangle")
    else:
        print("  ⚠️  K₆ should have monochromatic triangle")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve Ramsey problems")
    parser.add_argument('--n', type=int, default=5, help='Number of vertices (n for Kₙ)')
    parser.add_argument('--candidates', type=int, default=10, help='Number of candidate universes (meta-search)')
    parser.add_argument('--steps', type=int, default=1000, help='Maximum inner search steps per candidate')
    parser.add_argument('--outer', type=int, default=20, help='Number of outer meta-search iterations')
    parser.add_argument('--meta', action='store_true', help='Use meta-search (outer basin reinforcement loop)')
    parser.add_argument('--test-r33', action='store_true', help='Run R(3,3) test')
    parser.add_argument('--no-checkpoint', action='store_true', help='Disable checkpoint saving/resuming')
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='Save checkpoint every N steps')
    parser.add_argument('--visualize', action='store_true', help='Show live visualization of system state')
    parser.add_argument('--viz-interval', type=int, default=50, help='Update visualization every N steps')
    
    args = parser.parse_args()
    
    if args.test_r33:
        test_ramsey_r33()
    elif args.meta:
        # Use meta-search (outer basin reinforcement loop)
        try:
            from .ramsey_meta_search import solve_ramsey_meta
        except ImportError:
            from ramsey_meta_search import solve_ramsey_meta
        
        solve_ramsey_meta(
            n_vertices=args.n,
            n_candidates=args.candidates,
            inner_steps=args.steps,
            outer_iterations=args.outer,
            verbose=True
        )
    else:
        # Use single-universe dynamic search (inner loop only)
        solve_ramsey(
            n_vertices=args.n,
            n_candidates=args.candidates,
            max_steps=args.steps,
            verbose=True,
            use_checkpoint=not args.no_checkpoint,
            checkpoint_interval=args.checkpoint_interval,
            visualize=args.visualize,
            visualize_interval=args.viz_interval
        )
