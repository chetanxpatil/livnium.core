"""
Compare patterns between normal mode and inverted label mode.

This reveals which geometric signals are INVARIANT (refuse to flip)
when labels are inverted, showing what the geometry truly believes.
"""

import os
import sys
import json
import numpy as np

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5.core.pattern_learner import PatternLearner


def compare_invariants(normal_file: str, inverted_file: str):
    """Compare patterns to find invariant signals."""
    print("=" * 80)
    print("REVERSE PHYSICS ANALYSIS: Finding Invariant Signals")
    print("=" * 80)
    print("\nThis compares normal vs inverted labels to find what REFUSES to flip.")
    print("Signals that stay stable are TRUE geometric invariants.\n")
    
    # Load patterns
    normal_learner = PatternLearner(debug_mode=False)
    normal_learner.load_patterns(normal_file)
    normal_learner.analyze()
    
    inverted_learner = PatternLearner(debug_mode=True)
    inverted_learner.load_patterns(inverted_file)
    inverted_learner.analyze()
    
    # Compare key signals for each class
    for label in ['entailment', 'contradiction', 'neutral']:
        if label not in normal_learner.stats or label not in inverted_learner.stats:
            continue
        
        print(f"\n{'='*80}")
        print(f"{label.upper()} - INVARIANCE ANALYSIS")
        print(f"{'='*80}\n")
        
        normal_signals = normal_learner.stats[label]['signals']
        inverted_signals = inverted_learner.stats[label]['signals']
        
        # Key signals to check for invariance
        key_signals = [
            'resonance',
            'divergence',
            'convergence',
            'cold_density',
            'divergence_force',
            'cold_attraction',
            'far_attraction',
            'curvature',
        ]
        
        print(f"{'Signal':<20} {'Normal':<15} {'Inverted':<15} {'Difference':<15} {'Status':<15}")
        print("-" * 80)
        
        invariants = []
        artifacts = []
        
        for signal_name in key_signals:
            if signal_name not in normal_signals or signal_name not in inverted_signals:
                continue
            
            normal_mean = normal_signals[signal_name]['mean']
            inverted_mean = inverted_signals[signal_name]['mean']
            diff = abs(normal_mean - inverted_mean)
            relative_diff = diff / (abs(normal_mean) + 1e-6)  # Relative change
            
            normal_str = f"{normal_mean:.4f}"
            inverted_str = f"{inverted_mean:.4f}"
            diff_str = f"{diff:.4f}"
            
            # Classify as invariant or artifact
            if relative_diff < 0.1:  # Less than 10% change
                status = "✓ INVARIANT"
                invariants.append(signal_name)
            elif relative_diff < 0.3:  # Less than 30% change
                status = "~ STABLE"
            else:
                status = "✗ ARTIFACT"
                artifacts.append(signal_name)
            
            print(f"{signal_name:<20} {normal_str:<15} {inverted_str:<15} {diff_str:<15} {status:<15}")
        
        # Summary
        print(f"\n{'─'*80}")
        print("INVARIANCE SUMMARY:")
        print(f"{'─'*80}")
        print(f"✓ INVARIANT signals (refuse to flip): {', '.join(invariants) if invariants else 'None'}")
        print(f"✗ ARTIFACT signals (follow wrong labels): {', '.join(artifacts) if artifacts else 'None'}")
        
        # Special analysis for divergence
        if 'divergence' in normal_signals and 'divergence' in inverted_signals:
            normal_div = normal_signals['divergence']['mean']
            inverted_div = inverted_signals['divergence']['mean']
            
            print(f"\nDIVERGENCE ANALYSIS:")
            print(f"  Normal: {normal_div:.4f}")
            print(f"  Inverted: {inverted_div:.4f}")
            print(f"  Difference: {abs(normal_div - inverted_div):.4f}")
            
            # Check if sign is preserved
            if (normal_div > 0 and inverted_div > 0) or (normal_div < 0 and inverted_div < 0):
                print(f"  ✓ SIGN PRESERVED: Divergence refuses to flip sign!")
                print(f"     This confirms divergence is a TRUE geometric law.")
            else:
                print(f"  ⚠️  SIGN FLIPPED: Divergence changed sign with labels")
                print(f"     This suggests divergence may be label-dependent.")
    
    # Overall insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")
    
    print("1. INVARIANT SIGNALS:")
    print("   Signals that refuse to flip are TRUE geometric laws.")
    print("   These are what the geometry truly believes, regardless of labels.\n")
    
    print("2. ARTIFACT SIGNALS:")
    print("   Signals that flip with labels are not fundamental.")
    print("   These may be label-dependent features or force-based interpretations.\n")
    
    print("3. THE DEEP IDEA:")
    print("   By forcing the system to lie (wrong labels), we discover what cannot lie.")
    print("   The unbreakable signals are the laws of the universe.\n")
    
    print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare normal vs inverted patterns to find invariants')
    parser.add_argument('--normal-file', type=str, 
                        default='experiments/nli_v5/patterns/patterns_normal.json',
                        help='Path to normal mode patterns file')
    parser.add_argument('--inverted-file', type=str,
                        default='experiments/nli_v5/patterns/patterns_inverted.json',
                        help='Path to inverted label patterns file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.normal_file):
        print(f"Normal patterns file not found: {args.normal_file}")
        print("Run: python3 experiments/nli_v5/train_v5.py --clean --train 1000 --learn-patterns --pattern-file {args.normal_file}")
        return
    
    if not os.path.exists(args.inverted_file):
        print(f"Inverted patterns file not found: {args.inverted_file}")
        print("Run: python3 experiments/nli_v5/train_v5.py --clean --train 1000 --invert-labels --learn-patterns --pattern-file {args.inverted_file}")
        return
    
    compare_invariants(args.normal_file, args.inverted_file)


if __name__ == '__main__':
    main()

