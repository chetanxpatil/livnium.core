"""
Compare patterns between debug mode (golden labels) and normal mode.

This script runs training in both modes and compares the geometric signals
to see how golden labels affect the patterns.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5.core.pattern_learner import PatternLearner


def run_training(mode: str, num_samples: int = 1000):
    """Run training in specified mode and return pattern file path."""
    script_path = os.path.join(os.path.dirname(__file__), 'train_v5.py')
    
    if mode == 'debug':
        pattern_file = os.path.join(os.path.dirname(__file__), '..', 'patterns', 'patterns_debug.json')
        cmd = [
            sys.executable, script_path,
            '--clean',
            '--train', str(num_samples),
            '--debug-golden',
            '--learn-patterns',
            '--pattern-file', pattern_file
        ]
    else:  # normal
        pattern_file = os.path.join(os.path.dirname(__file__), '..', 'patterns', 'patterns_normal.json')
        cmd = [
            sys.executable, script_path,
            '--clean',
            '--train', str(num_samples),
            '--learn-patterns',
            '--pattern-file', pattern_file
        ]
    
    print(f"\n{'='*80}")
    print(f"Running training in {mode.upper()} mode...")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running training in {mode} mode:")
        print(result.stderr)
        return None
    
    # Print last 50 lines of output
    lines = result.stdout.split('\n')
    print('\n'.join(lines[-50:]))
    
    return pattern_file


def compare_patterns(debug_file: str, normal_file: str):
    """Compare patterns from debug and normal mode."""
    print(f"\n{'='*80}")
    print("PATTERN COMPARISON: Debug Mode vs Normal Mode")
    print(f"{'='*80}\n")
    
    # Load patterns
    debug_learner = PatternLearner(debug_mode=True)
    debug_learner.load_patterns(debug_file)
    debug_learner.analyze()
    
    normal_learner = PatternLearner(debug_mode=False)
    normal_learner.load_patterns(normal_file)
    normal_learner.analyze()
    
    # Compare key signals for each class
    for label in ['entailment', 'contradiction', 'neutral']:
        if label not in debug_learner.stats or label not in normal_learner.stats:
            continue
        
        print(f"\n{'='*80}")
        print(f"{label.upper()} COMPARISON")
        print(f"{'='*80}\n")
        
        debug_signals = debug_learner.stats[label]['signals']
        normal_signals = normal_learner.stats[label]['signals']
        
        # Key signals to compare
        key_signals = [
            'resonance',
            'divergence',
            'convergence',
            'cold_density',
            'divergence_force',
            'cold_attraction',
            'far_attraction',
            'cold_force',
            'far_force',
            'city_force',
        ]
        
        print(f"{'Signal':<20} {'Debug Mode':<25} {'Normal Mode':<25} {'Difference':<15}")
        print("-" * 85)
        
        for signal_name in key_signals:
            if signal_name not in debug_signals or signal_name not in normal_signals:
                continue
            
            debug_mean = debug_signals[signal_name]['mean']
            normal_mean = normal_signals[signal_name]['mean']
            diff = debug_mean - normal_mean
            
            debug_str = f"{debug_mean:.4f} ± {debug_signals[signal_name]['std']:.4f}"
            normal_str = f"{normal_mean:.4f} ± {normal_signals[signal_name]['std']:.4f}"
            diff_str = f"{diff:+.4f}"
            
            # Highlight significant differences
            if abs(diff) > 0.1:
                diff_str = f"⚠️  {diff_str}"
            
            print(f"{signal_name:<20} {debug_str:<25} {normal_str:<25} {diff_str:<15}")
        
        # Special analysis for forces (which are artificial in debug mode)
        print(f"\n{'─'*85}")
        print("FORCE ANALYSIS:")
        print(f"{'─'*85}")
        print("In DEBUG mode, forces are ARTIFICIALLY set to match golden labels.")
        print("In NORMAL mode, forces come from REAL geometry.")
        print("\nExpected debug forces:")
        if label == 'entailment':
            print("  cold=0.7, far=0.2, city=0.1")
        elif label == 'contradiction':
            print("  cold=0.2, far=0.7, city=0.1")
        else:  # neutral
            print("  cold=0.33, far=0.33, city=0.34")
        
        print(f"\nActual debug forces:")
        print(f"  cold={debug_signals['cold_force']['mean']:.3f}, "
              f"far={debug_signals['far_force']['mean']:.3f}, "
              f"city={debug_signals['city_force']['mean']:.3f}")
        
        print(f"\nActual normal forces:")
        print(f"  cold={normal_signals['cold_force']['mean']:.3f}, "
              f"far={normal_signals['far_force']['mean']:.3f}, "
              f"city={normal_signals['city_force']['mean']:.3f}")
        
        # Calculate gap
        force_gap_cold = debug_signals['cold_force']['mean'] - normal_signals['cold_force']['mean']
        force_gap_far = debug_signals['far_force']['mean'] - normal_signals['far_force']['mean']
        
        print(f"\nForce Gap (ideal - real):")
        print(f"  cold_force gap: {force_gap_cold:+.3f}")
        print(f"  far_force gap: {force_gap_far:+.3f}")
        
        if abs(force_gap_cold) > 0.2 or abs(force_gap_far) > 0.2:
            print("\n⚠️  LARGE GAP DETECTED: Geometry is not producing strong enough forces!")
            print("   Consider boosting cold_density or divergence_force computation.")
    
    # Overall insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")
    
    print("1. GEOMETRIC SIGNALS (resonance, divergence, convergence):")
    print("   These should be SIMILAR in both modes (they come from layers 0-3)")
    print("   If they differ significantly, there's a bug in signal computation.\n")
    
    print("2. FORCES (cold_force, far_force, city_force):")
    print("   - Debug mode: ARTIFICIAL (set to match golden labels)")
    print("   - Normal mode: REAL (computed from geometry)")
    print("   Large gaps indicate geometry needs calibration.\n")
    
    print("3. DIVERGENCE CHECK:")
    e_debug_div = debug_learner.stats['entailment']['signals']['divergence']['mean']
    e_normal_div = normal_learner.stats['entailment']['signals']['divergence']['mean']
    c_debug_div = debug_learner.stats['contradiction']['signals']['divergence']['mean']
    c_normal_div = normal_learner.stats['contradiction']['signals']['divergence']['mean']
    
    print(f"   Entailment divergence:")
    print(f"     Debug: {e_debug_div:.4f} (should be negative)")
    print(f"     Normal: {e_normal_div:.4f} (should be negative)")
    if e_normal_div > 0:
        print(f"     ⚠️  PROBLEM: Normal mode divergence is POSITIVE (should be negative)!")
    
    print(f"\n   Contradiction divergence:")
    print(f"     Debug: {c_debug_div:.4f} (should be positive)")
    print(f"     Normal: {c_normal_div:.4f} (should be positive)")
    if c_normal_div < 0:
        print(f"     ⚠️  PROBLEM: Normal mode divergence is NEGATIVE (should be positive)!")
    
    print(f"\n{'='*80}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare patterns between debug and normal mode')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of training samples to use')
    parser.add_argument('--debug-file', type=str, default=None,
                        help='Path to existing debug patterns file (skip debug training)')
    parser.add_argument('--normal-file', type=str, default=None,
                        help='Path to existing normal patterns file (skip normal training)')
    
    args = parser.parse_args()
    
    # Run training or load existing patterns
    if args.debug_file and os.path.exists(args.debug_file):
        debug_file = args.debug_file
        print(f"Using existing debug patterns: {debug_file}")
    else:
        debug_file = run_training('debug', args.samples)
        if not debug_file:
            print("Failed to generate debug patterns")
            return
    
    if args.normal_file and os.path.exists(args.normal_file):
        normal_file = args.normal_file
        print(f"Using existing normal patterns: {normal_file}")
    else:
        normal_file = run_training('normal', args.samples)
        if not normal_file:
            print("Failed to generate normal patterns")
            return
    
    # Compare patterns
    compare_patterns(debug_file, normal_file)


if __name__ == '__main__':
    main()

