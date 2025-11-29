"""
Extract and analyze canonical physics fingerprints from golden labels.

This creates the "phase diagram" of what E/C/N geometry SHOULD look like.
"""

import json
import numpy as np
import os
from pathlib import Path


def extract_fingerprints(pattern_file: str):
    """Extract canonical physics fingerprints from golden label patterns."""
    
    if not os.path.exists(pattern_file):
        print(f"Pattern file not found: {pattern_file}")
        return None
    
    with open(pattern_file, 'r') as f:
        data = json.load(f)
    
    fingerprints = {}
    
    for label in ['entailment', 'contradiction', 'neutral']:
        if label not in data.get('patterns', {}):
            continue
        
        patterns = data['patterns'][label]
        if not patterns:
            continue
        
        # Extract all signals
        signals = {
            'alignment': [],
            'divergence': [],
            'convergence': [],
            'resonance': [],
            'cold_density': [],
            'divergence_force': [],
            'cold_attraction': [],
            'far_attraction': [],
            'cold_force': [],
            'far_force': [],
            'city_force': [],
        }
        
        for p in patterns:
            # Compute alignment from divergence (reverse: alignment = 0.38 - divergence)
            div = p.get('divergence', 0.0)
            align = 0.38 - div  # Reverse the divergence formula
            
            signals['alignment'].append(align)
            signals['divergence'].append(div)
            signals['convergence'].append(p.get('convergence', -div))
            signals['resonance'].append(p.get('resonance', 0.0))
            signals['cold_density'].append(p.get('cold_density', 0.0))
            signals['divergence_force'].append(p.get('divergence_force', 0.0))
            signals['cold_attraction'].append(p.get('cold_attraction', 0.0))
            signals['far_attraction'].append(p.get('far_attraction', 0.0))
            signals['cold_force'].append(p.get('cold_force', 0.33))
            signals['far_force'].append(p.get('far_force', 0.33))
            signals['city_force'].append(p.get('city_force', 0.33))
        
        # Compute statistics
        stats = {}
        for signal_name, values in signals.items():
            if not values:
                continue
            
            stats[signal_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
            }
        
        fingerprints[label] = {
            'count': len(patterns),
            'signals': stats
        }
    
    return fingerprints


def print_fingerprints(fingerprints):
    """Print canonical physics fingerprints."""
    print("\n" + "=" * 80)
    print("CANONICAL PHYSICS FINGERPRINTS (from Golden Labels)")
    print("=" * 80)
    print("\nThese are the geometric signatures of correct E/C/N classifications.")
    print("Use these as the 'phase diagram' for decision rules.\n")
    
    key_signals = [
        'alignment',
        'divergence',
        'convergence',
        'resonance',
        'cold_density',
        'divergence_force',
        'cold_attraction',
        'far_attraction',
    ]
    
    for label in ['entailment', 'contradiction', 'neutral']:
        if label not in fingerprints:
            continue
        
        fp = fingerprints[label]
        print(f"\n{label.upper()} (n={fp['count']}):")
        print("-" * 80)
        print(f"{'Signal':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
        print("-" * 80)
        
        for signal_name in key_signals:
            if signal_name in fp['signals']:
                s = fp['signals'][signal_name]
                print(f"{signal_name:<20} {s['mean']:<10.4f} {s['std']:<10.4f} "
                      f"{s['min']:<10.4f} {s['max']:<10.4f} {s['median']:<10.4f}")
    
    # Phase diagram summary
    print("\n" + "=" * 80)
    print("PHASE DIAGRAM SUMMARY")
    print("=" * 80)
    
    e_fp = fingerprints.get('entailment', {}).get('signals', {})
    c_fp = fingerprints.get('contradiction', {}).get('signals', {})
    n_fp = fingerprints.get('neutral', {}).get('signals', {})
    
    if e_fp and c_fp and n_fp:
        print("\n1. DIVERGENCE (x-axis):")
        print(f"   E: {e_fp['divergence']['mean']:.4f} ± {e_fp['divergence']['std']:.4f} (should be negative)")
        print(f"   C: {c_fp['divergence']['mean']:.4f} ± {c_fp['divergence']['std']:.4f} (should be positive)")
        print(f"   N: {n_fp['divergence']['mean']:.4f} ± {n_fp['divergence']['std']:.4f} (should be near zero)")
        
        print("\n2. RESONANCE (y-axis):")
        print(f"   E: {e_fp['resonance']['mean']:.4f} ± {e_fp['resonance']['std']:.4f}")
        print(f"   C: {c_fp['resonance']['mean']:.4f} ± {c_fp['resonance']['std']:.4f}")
        print(f"   N: {n_fp['resonance']['mean']:.4f} ± {n_fp['resonance']['std']:.4f}")
        
        print("\n3. CONVERGENCE (E-axis):")
        print(f"   E: {e_fp['convergence']['mean']:.4f} ± {e_fp['convergence']['std']:.4f} (should be positive)")
        print(f"   C: {c_fp['convergence']['mean']:.4f} ± {c_fp['convergence']['std']:.4f} (should be negative)")
        print(f"   N: {n_fp['convergence']['mean']:.4f} ± {n_fp['convergence']['std']:.4f} (should be near zero)")
        
        print("\n4. DECISION THRESHOLDS (suggested):")
        # Use mean ± 1 std as thresholds
        d_e_mean = e_fp['divergence']['mean']
        d_e_std = e_fp['divergence']['std']
        d_c_mean = c_fp['divergence']['mean']
        d_c_std = c_fp['divergence']['std']
        d_n_mean = n_fp['divergence']['mean']
        d_n_std = n_fp['divergence']['std']
        
        r_e_mean = e_fp['resonance']['mean']
        r_e_std = e_fp['resonance']['std']
        r_c_mean = c_fp['resonance']['mean']
        r_c_std = c_fp['resonance']['std']
        r_n_mean = n_fp['resonance']['mean']
        r_n_std = n_fp['resonance']['std']
        
        print(f"   Divergence thresholds:")
        print(f"     E (high): d < {d_e_mean + d_e_std:.4f}")
        print(f"     C (high): d > {d_c_mean - d_c_std:.4f}")
        print(f"     N (band): {d_n_mean - d_n_std:.4f} < d < {d_n_mean + d_n_std:.4f}")
        
        print(f"\n   Resonance thresholds:")
        print(f"     E (high): r > {r_e_mean - r_e_std:.4f}")
        print(f"     C (mid):  {r_c_mean - r_c_std:.4f} < r < {r_c_mean + r_c_std:.4f}")
        print(f"     N (mid):  {r_n_mean - r_n_std:.4f} < r < {r_n_mean + r_n_std:.4f}")
        
        print("\n5. PHYSICS-BASED DECISION RULES:")
        print("   if divergence > threshold_c_high:")
        print("       predict = CONTRADICTION")
        print("   elif divergence < threshold_e_low AND resonance > threshold_r_e_min:")
        print("       predict = ENTAILMENT")
        print("   elif abs(divergence) < threshold_n_band:")
        print("       predict = NEUTRAL")
        print("   else:")
        print("       fallback to force-based decision")


def save_fingerprints(fingerprints, output_file: str):
    """Save fingerprints to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(fingerprints, f, indent=2)
    print(f"\n✓ Fingerprints saved to: {output_file}")


def main():
    pattern_file = 'experiments/nli_v5/patterns/patterns_canonical.json'
    
    if not os.path.exists(pattern_file):
        print(f"Pattern file not found: {pattern_file}")
        print("Run: python3 experiments/nli_v5/train_v5.py --clean --train 1000 --debug-golden --learn-patterns --pattern-file {pattern_file}")
        return
    
    fingerprints = extract_fingerprints(pattern_file)
    
    if fingerprints:
        print_fingerprints(fingerprints)
        save_fingerprints(fingerprints, 'experiments/nli_v5/planet_output/physics_fingerprints.json')


if __name__ == '__main__':
    main()

