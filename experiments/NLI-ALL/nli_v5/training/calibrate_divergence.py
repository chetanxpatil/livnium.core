"""
Calibrate Divergence Equilibrium Threshold K from Pattern Data

This script calibrates the divergence equilibrium threshold K from pattern data
using either neutral-anchored or E/C midpoint method.
"""

import os
import sys
import json
import argparse

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from experiments.nli_v5.core.layers import Layer0Resonance


def calibrate_from_patterns(pattern_file: str, method: str = 'neutral'):
    """
    Calibrate equilibrium threshold K from pattern file.
    
    Args:
        pattern_file: Path to pattern JSON file
        method: 'neutral' (neutral-anchored) or 'midpoint' (E/C midpoint)
    
    Returns:
        Calibrated threshold K
    """
    if not os.path.exists(pattern_file):
        print(f"❌ Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        patterns_data = json.load(f)
    
    stats = patterns_data.get('stats', {})
    
    if method == 'neutral':
        # Option A: Neutral-anchored (makes neutral the rest surface)
        if 'neutral' not in stats:
            print("❌ Neutral stats not found in pattern file")
            return None
        
        neutral_div = stats['neutral']['signals'].get('divergence', {}).get('mean', 0.0)
        current_K = Layer0Resonance.equilibrium_threshold
        
        # Estimate alignment from divergence: alignment = K - divergence
        # For neutral, we want divergence ≈ 0, so K ≈ alignment_neutral
        estimated_alignment = current_K - neutral_div
        
        # New K should make neutral divergence ≈ 0, so K = alignment_neutral
        new_K = estimated_alignment
        
        print(f"Neutral-anchored calibration:")
        print(f"  Current K: {current_K:.4f}")
        print(f"  Neutral divergence: {neutral_div:.4f}")
        print(f"  Estimated neutral alignment: {estimated_alignment:.4f}")
        print(f"  New K: {new_K:.4f}")
        
        return new_K
    
    elif method == 'midpoint':
        # Option B: Midpoint between E and C
        if 'entailment' not in stats or 'contradiction' not in stats:
            print("❌ Entailment or Contradiction stats not found")
            return None
        
        e_div = stats['entailment']['signals'].get('divergence', {}).get('mean', 0.0)
        c_div = stats['contradiction']['signals'].get('divergence', {}).get('mean', 0.0)
        current_K = Layer0Resonance.equilibrium_threshold
        
        # Estimate alignments: alignment = K - divergence
        e_align = current_K - e_div
        c_align = current_K - c_div
        
        # New K is midpoint
        new_K = 0.5 * (e_align + c_align)
        
        print(f"E/C midpoint calibration:")
        print(f"  Current K: {current_K:.4f}")
        print(f"  E divergence: {e_div:.4f} → alignment: {e_align:.4f}")
        print(f"  C divergence: {c_div:.4f} → alignment: {c_align:.4f}")
        print(f"  New K (midpoint): {new_K:.4f}")
        
        return new_K
    
    else:
        print(f"❌ Unknown method: {method}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Calibrate divergence equilibrium threshold')
    parser.add_argument('--pattern-file', type=str, required=True,
                        help='Path to pattern JSON file')
    parser.add_argument('--method', type=str, default='neutral',
                        choices=['neutral', 'midpoint'],
                        help='Calibration method: neutral (neutral-anchored) or midpoint (E/C midpoint)')
    parser.add_argument('--apply', action='store_true',
                        help='Apply calibration to Layer0Resonance class')
    
    args = parser.parse_args()
    
    print("="*80)
    print("DIVERGENCE EQUILIBRIUM THRESHOLD CALIBRATION")
    print("="*80)
    print()
    
    new_K = calibrate_from_patterns(args.pattern_file, args.method)
    
    if new_K is None:
        return
    
    print()
    print("="*80)
    print("CALIBRATION RESULT")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Calibrated K: {new_K:.4f}")
    print()
    
    if args.apply:
        Layer0Resonance.equilibrium_threshold = new_K
        print(f"✅ Applied calibration: K = {new_K:.4f}")
        print()
        print("Note: This change is only in memory. To make it permanent,")
        print("update Layer0Resonance.equilibrium_threshold in layers.py")
    else:
        print("To apply this calibration, run with --apply flag")
        print("Or manually update Layer0Resonance.equilibrium_threshold in layers.py")


if __name__ == '__main__':
    main()

