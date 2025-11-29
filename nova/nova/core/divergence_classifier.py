"""
Divergence-Based Classifier: Direct Physics Law Implementation

This implements the Divergence Law directly:
- divergence < 0 → entailment
- divergence ≈ 0 → neutral
- divergence > 0 → contradiction

This bypasses the unsupervised cluster+grammar pipeline and uses pure physics.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import defaultdict


class DivergenceClassifier:
    """
    Pure physics-based classifier using divergence law.
    
    Law: divergence = 0.38 - alignment
    Interpretation:
    - divergence < -delta → entailment
    - |divergence| < delta → neutral (with fracture check)
    - divergence > +delta → contradiction
    """
    
    def __init__(self, 
                 divergence_threshold: float = 0.1,
                 fracture_threshold: float = 0.5):
        """
        Initialize divergence classifier.
        
        Args:
            divergence_threshold: Threshold for E/C/N classification (delta)
            fracture_threshold: Threshold for neutral vs contradiction when divergence is near-zero
        """
        self.divergence_threshold = divergence_threshold
        self.fracture_threshold = fracture_threshold
        
        # Statistics (computed from training data)
        self.stats: Optional[Dict] = None
    
    def compute_statistics(self, 
                          signatures: np.ndarray,
                          labels: np.ndarray,
                          label_names: Dict[int, str] = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}):
        """
        Compute divergence statistics per label from training data.
        
        This helps tune thresholds and verify the law is working.
        
        Args:
            signatures: Array of extended signatures [premise_SW, hypothesis_SW, alignment, divergence, fracture]
            labels: Array of label indices
            label_names: Mapping from label index to name
        """
        # Extract divergence and fracture from signatures
        # Signatures are: [premise_SW (125), hypothesis_SW (125), alignment (1), divergence (1), fracture (1)]
        # Total: 252 dimensions for lattice_size=5
        
        # Find indices for alignment, divergence, fracture (last 3 elements)
        n_features = signatures.shape[1]
        alignment_idx = n_features - 3
        divergence_idx = n_features - 2
        fracture_idx = n_features - 1
        
        stats = defaultdict(lambda: {'divergence': [], 'fracture': [], 'alignment': []})
        
        for i, label_idx in enumerate(labels):
            label_name = label_names[label_idx]
            divergence = signatures[i, divergence_idx]
            fracture = signatures[i, fracture_idx]
            alignment = signatures[i, alignment_idx]
            
            stats[label_name]['divergence'].append(divergence)
            stats[label_name]['fracture'].append(fracture)
            stats[label_name]['alignment'].append(alignment)
        
        # Compute summary statistics
        self.stats = {}
        for label_name, values in stats.items():
            self.stats[label_name] = {
                'divergence': {
                    'mean': np.mean(values['divergence']),
                    'std': np.std(values['divergence']),
                    'median': np.median(values['divergence']),
                    'min': np.min(values['divergence']),
                    'max': np.max(values['divergence']),
                    'percentile_25': np.percentile(values['divergence'], 25),
                    'percentile_75': np.percentile(values['divergence'], 75)
                },
                'fracture': {
                    'mean': np.mean(values['fracture']),
                    'std': np.std(values['fracture']),
                    'median': np.median(values['fracture'])
                },
                'alignment': {
                    'mean': np.mean(values['alignment']),
                    'std': np.std(values['alignment']),
                    'median': np.median(values['alignment'])
                }
            }
        
        return self.stats
    
    def print_statistics(self):
        """Print computed statistics in readable format."""
        if self.stats is None:
            print("No statistics computed. Call compute_statistics() first.")
            return
        
        print("\n" + "=" * 70)
        print("Divergence Law Statistics (per label)")
        print("=" * 70)
        print()
        
        for label_name in ['entailment', 'contradiction', 'neutral']:
            if label_name not in self.stats:
                continue
            
            stats = self.stats[label_name]
            print(f"{label_name.upper()}:")
            print(f"  Divergence: mean={stats['divergence']['mean']:.4f}, "
                  f"std={stats['divergence']['std']:.4f}, "
                  f"median={stats['divergence']['median']:.4f}")
            print(f"    Range: [{stats['divergence']['min']:.4f}, {stats['divergence']['max']:.4f}]")
            print(f"    IQR: [{stats['divergence']['percentile_25']:.4f}, "
                  f"{stats['divergence']['percentile_75']:.4f}]")
            print(f"  Fracture: mean={stats['fracture']['mean']:.4f}, "
                  f"median={stats['fracture']['median']:.4f}")
            print(f"  Alignment: mean={stats['alignment']['mean']:.4f}, "
                  f"median={stats['alignment']['median']:.4f}")
            print()
        
        # Check if law is working
        print("Law Verification:")
        entail_div = self.stats['entailment']['divergence']['median']
        neutral_div = self.stats['neutral']['divergence']['median']
        contra_div = self.stats['contradiction']['divergence']['median']
        
        if entail_div < neutral_div < contra_div:
            print("  ✓ Law working: entailment < neutral < contradiction (divergence)")
        else:
            print(f"  ⚠ Law violation: order is {entail_div:.4f} < {neutral_div:.4f} < {contra_div:.4f}")
        print()
    
    def auto_tune_thresholds(self):
        """
        Automatically tune thresholds based on computed statistics.
        
        Uses percentiles to find natural boundaries.
        """
        if self.stats is None:
            raise ValueError("Must compute statistics first")
        
        # Find divergence boundaries
        entail_div = self.stats['entailment']['divergence']['percentile_75']  # Upper bound for entailment
        neutral_low = self.stats['neutral']['divergence']['percentile_25']  # Lower bound for neutral
        neutral_high = self.stats['neutral']['divergence']['percentile_75']  # Upper bound for neutral
        contra_div = self.stats['contradiction']['divergence']['percentile_25']  # Lower bound for contradiction
        
        # Use median of gaps as threshold
        self.divergence_threshold = max(
            abs(entail_div - neutral_low),
            abs(neutral_high - contra_div)
        ) / 2
        
        # Fracture threshold: median of neutral (low fracture) vs contradiction (high fracture)
        neutral_fracture = self.stats['neutral']['fracture']['median']
        contra_fracture = self.stats['contradiction']['fracture']['median']
        self.fracture_threshold = (neutral_fracture + contra_fracture) / 2
        
        print(f"Auto-tuned thresholds:")
        print(f"  Divergence threshold: {self.divergence_threshold:.4f}")
        print(f"  Fracture threshold: {self.fracture_threshold:.4f}")
        print()
    
    def classify(self, divergence: float, fracture: float) -> str:
        """
        Classify based on divergence law.
        
        Law:
        - divergence < -threshold → entailment
        - |divergence| < threshold → neutral (with fracture check)
        - divergence > +threshold → contradiction
        
        Args:
            divergence: Divergence value (0.38 - alignment)
            fracture: Fracture value (geometric inconsistency)
            
        Returns:
            'entailment', 'contradiction', or 'neutral'
        """
        if divergence < -self.divergence_threshold:
            return "entailment"
        elif divergence > self.divergence_threshold:
            return "contradiction"
        else:
            # Near-zero divergence: neutral vs soft contradiction
            # High fracture suggests contradiction despite low divergence
            if fracture > self.fracture_threshold:
                return "contradiction"
            else:
                return "neutral"
    
    def classify_from_signature(self, signature: np.ndarray) -> str:
        """
        Classify from extended signature.
        
        Args:
            signature: Extended signature [premise_SW, hypothesis_SW, alignment, divergence, fracture]
            
        Returns:
            'entailment', 'contradiction', or 'neutral'
        """
        # Extract divergence and fracture (last 2 elements)
        n_features = signature.shape[0]
        divergence_idx = n_features - 2
        fracture_idx = n_features - 1
        
        divergence = signature[divergence_idx]
        fracture = signature[fracture_idx]
        
        return self.classify(divergence, fracture)

