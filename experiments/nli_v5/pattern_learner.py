"""
Pattern Learner: Use Golden Labels to Understand Geometry

When we feed golden labels, we can record what geometric signals (resonance, divergence, etc.)
correspond to correct classifications. This helps us understand what the geometry SHOULD look like.

This is reverse engineering: if we know the answer (golden label) and we know what forces
produce that answer, we can learn what patterns lead to correct classification.
"""

import json
import numpy as np
from typing import Dict, List
from collections import defaultdict
from pathlib import Path


class PatternLearner:
    """
    Records and analyzes geometric patterns from golden labels.
    
    For each example with a golden label, records:
    - Resonance
    - Divergence
    - Convergence
    - Cold density
    - Divergence force
    - Cold attraction
    - Far attraction
    - Forces (cold_force, far_force, city_force)
    
    Then analyzes what patterns correlate with correct classification.
    """
    
    def __init__(self, debug_mode: bool = False):
        self.patterns = defaultdict(list)  # label -> list of signal dicts
        self.stats = {}  # Computed statistics
        self.debug_mode = debug_mode  # Track if we're in debug mode
    
    def record(self, golden_label: str, layer_states: Dict):
        """
        Record geometric signals for a golden label.
        
        Args:
            golden_label: 'entailment', 'contradiction', or 'neutral'
            layer_states: Output from classifier.classify().layer_states
        """
        # Extract key signals from layer states
        signals = {
            'resonance': layer_states.get('resonance', 0.0),
            'divergence': layer_states.get('divergence', 0.0),
            'convergence': layer_states.get('convergence', 0.0),
            'cold_density': layer_states.get('cold_density', 0.0),
            'divergence_force': layer_states.get('divergence_force', layer_states.get('distance', 0.0)),
            'cold_attraction': layer_states.get('cold_attraction', 0.0),
            'far_attraction': layer_states.get('far_attraction', 0.0),
            'cold_force': layer_states.get('cold_force', 0.33),
            'far_force': layer_states.get('far_force', 0.33),
            'city_force': layer_states.get('city_force', 0.33),
            'curvature': layer_states.get('curvature', 0.0),
            'word_opposition': layer_states.get('word_opposition', 0.0),
            'learned_contradiction': layer_states.get('learned_contradiction', 0.0),
        }
        
        self.patterns[golden_label].append(signals)
    
    def analyze(self) -> Dict:
        """
        Analyze patterns and compute statistics per class.
        
        Returns:
            Dict with statistics for each label
        """
        stats = {}
        
        for label in ['entailment', 'contradiction', 'neutral']:
            if label not in self.patterns or not self.patterns[label]:
                stats[label] = {'count': 0}
                continue
            
            patterns = self.patterns[label]
            count = len(patterns)
            
            # Compute statistics for each signal
            signal_stats = {}
            for signal_name in patterns[0].keys():
                values = [p[signal_name] for p in patterns]
                signal_stats[signal_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                }
            
            stats[label] = {
                'count': count,
                'signals': signal_stats
            }
        
        self.stats = stats
        return stats
    
    def print_analysis(self):
        """Print human-readable analysis."""
        if not self.stats:
            self.analyze()
        
        print("\n" + "=" * 80)
        if hasattr(self, 'invert_mode') and self.invert_mode:
            mode_str = "REVERSE PHYSICS MODE"
        elif self.debug_mode:
            mode_str = "DEBUG MODE"
        else:
            mode_str = "NORMAL MODE"
        print(f"GEOMETRIC PATTERN ANALYSIS (from Golden Labels) - {mode_str}")
        print("=" * 80)
        
        if hasattr(self, 'invert_mode') and self.invert_mode:
            print("\n⚠️  REVERSE PHYSICS MODE: Labels were INVERTED (E↔C).")
            print("   Forces are set to match INVERTED labels (wrong).")
            print("   Geometric signals (resonance, divergence) are REAL from layers 0-3.")
            print("   This reveals what geometry refuses to change when labels are wrong.")
            print("   Look for signals that stay stable - those are the TRUE invariants.\n")
        elif self.debug_mode:
            print("\n⚠️  DEBUG MODE: Forces are artificially set to match golden labels.")
            print("   Geometric signals (resonance, divergence) are REAL from layers 0-3.")
            print("   This shows what geometry produces vs what forces are needed.\n")
        
        for label in ['entailment', 'contradiction', 'neutral']:
            if label not in self.stats or self.stats[label]['count'] == 0:
                continue
            
            stats = self.stats[label]
            print(f"\n{label.upper()} (n={stats['count']}):")
            print("-" * 80)
            
            signals = stats['signals']
            
            # Key signals to display
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
            
            print(f"{'Signal':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Median':<10}")
            print("-" * 80)
            
            for signal_name in key_signals:
                if signal_name in signals:
                    s = signals[signal_name]
                    print(f"{signal_name:<20} {s['mean']:<10.4f} {s['std']:<10.4f} "
                          f"{s['min']:<10.4f} {s['max']:<10.4f} {s['median']:<10.4f}")
        
        print("\n" + "=" * 80)
        print("KEY INSIGHTS:")
        print("=" * 80)
        
        # Compare classes
        if all(label in self.stats and self.stats[label]['count'] > 0 
               for label in ['entailment', 'contradiction', 'neutral']):
            
            e_stats = self.stats['entailment']['signals']
            c_stats = self.stats['contradiction']['signals']
            n_stats = self.stats['neutral']['signals']
            
            print("\n1. RESONANCE:")
            print(f"   E: {e_stats['resonance']['mean']:.4f} ± {e_stats['resonance']['std']:.4f}")
            print(f"   C: {c_stats['resonance']['mean']:.4f} ± {c_stats['resonance']['std']:.4f}")
            print(f"   N: {n_stats['resonance']['mean']:.4f} ± {n_stats['resonance']['std']:.4f}")
            
            print("\n2. DIVERGENCE:")
            print(f"   E: {e_stats['divergence']['mean']:.4f} ± {e_stats['divergence']['std']:.4f} (should be negative)")
            print(f"   C: {c_stats['divergence']['mean']:.4f} ± {c_stats['divergence']['std']:.4f} (should be positive)")
            print(f"   N: {n_stats['divergence']['mean']:.4f} ± {n_stats['divergence']['std']:.4f} (should be near zero)")
            
            print("\n3. CONVERGENCE:")
            print(f"   E: {e_stats['convergence']['mean']:.4f} ± {e_stats['convergence']['std']:.4f} (should be positive)")
            print(f"   C: {c_stats['convergence']['mean']:.4f} ± {c_stats['convergence']['std']:.4f} (should be negative)")
            print(f"   N: {n_stats['convergence']['mean']:.4f} ± {n_stats['convergence']['std']:.4f} (should be near zero)")
            
            print("\n4. FORCES:")
            print(f"   E: cold={e_stats['cold_force']['mean']:.3f}, far={e_stats['far_force']['mean']:.3f}, city={e_stats['city_force']['mean']:.3f}")
            print(f"   C: cold={c_stats['cold_force']['mean']:.3f}, far={c_stats['far_force']['mean']:.3f}, city={c_stats['city_force']['mean']:.3f}")
            print(f"   N: cold={n_stats['cold_force']['mean']:.3f}, far={n_stats['far_force']['mean']:.3f}, city={n_stats['city_force']['mean']:.3f}")
            
            print("\n5. ATTRACTIONS:")
            print(f"   E: cold_attraction={e_stats['cold_attraction']['mean']:.4f}, far_attraction={e_stats['far_attraction']['mean']:.4f}")
            print(f"   C: cold_attraction={c_stats['cold_attraction']['mean']:.4f}, far_attraction={c_stats['far_attraction']['mean']:.4f}")
            print(f"   N: cold_attraction={n_stats['cold_attraction']['mean']:.4f}, far_attraction={n_stats['far_attraction']['mean']:.4f}")
            
            if self.debug_mode:
                print("\n6. DEBUG MODE INSIGHT:")
                print("   Forces are set to: E(cold=0.7, far=0.2), C(cold=0.2, far=0.7), N(cold=0.33, far=0.33)")
                print("   Compare geometric signals above to see if geometry matches these ideal forces.")
                print("   If divergence is wrong sign or attractions are weak, geometry needs calibration.")
        
        print("\n" + "=" * 80)
    
    def save_patterns(self, filepath: str):
        """Save patterns to JSON file."""
        data = {
            'patterns': {k: v for k, v in self.patterns.items()},
            'stats': self.stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Patterns saved to: {filepath}")
    
    def load_patterns(self, filepath: str):
        """Load patterns from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.patterns = defaultdict(list, data.get('patterns', {}))
        self.stats = data.get('stats', {})
        
        print(f"✓ Patterns loaded from: {filepath}")

