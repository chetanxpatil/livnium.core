"""
Divergence V2: Structure-Aware Divergence

Redesigned divergence that depends on actual sequence structure,
not just self-comparison. Enables discovery of real invariants.
"""

import numpy as np
import math
from typing import List, Tuple, Dict
from experiments.rule30.diagnostics import create_sequence_vectors


def divergence_v2_transition_based(sequence: List[int], window_size: int = 5) -> float:
    """
    Candidate 1: Transition-Based Divergence
    
    Measures divergence from bit transitions and local pattern structure.
    
    Args:
        sequence: Binary sequence
        window_size: Window size for local analysis
        
    Returns:
        Divergence value that varies with sequence structure
    """
    if len(sequence) < window_size + 1:
        return 0.0
    
    vectors = create_sequence_vectors(sequence)
    
    if len(vectors) < window_size + 1:
        return 0.0
    
    divergence_values = []
    
    for i in range(len(vectors) - window_size):
        # Get two consecutive windows
        window1 = vectors[i:i + window_size]
        window2 = vectors[i + 1:i + 1 + window_size]
        
        # Compute mean vectors
        mean1 = np.mean([v for v in window1 if np.linalg.norm(v) > 1e-6], axis=0)
        mean2 = np.mean([v for v in window2 if np.linalg.norm(v) > 1e-6], axis=0)
        
        if np.linalg.norm(mean1) < 1e-6 or np.linalg.norm(mean2) < 1e-6:
            continue
        
        # Normalize
        unit1 = mean1 / np.linalg.norm(mean1)
        unit2 = mean2 / np.linalg.norm(mean2)
        
        # Compute angle between consecutive windows
        cos_sim = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
        theta = np.arccos(cos_sim)
        
        # Weight by transition count in this region
        local_seq = sequence[i:i + window_size + 1]
        transitions = sum(1 for j in range(len(local_seq) - 1) if local_seq[j] != local_seq[j+1])
        transition_weight = transitions / len(local_seq)
        
        # Divergence from transition structure
        divergence = theta * transition_weight
        
        divergence_values.append(divergence)
    
    if not divergence_values:
        return 0.0
    
    return float(np.mean(divergence_values))


def divergence_v2_neighborhood(sequence: List[int], window_size: int = 5) -> float:
    """
    Candidate 2: Neighborhood Divergence
    
    Compares neighboring windows to capture local structure changes.
    
    Args:
        sequence: Binary sequence
        window_size: Window size for comparison
        
    Returns:
        Divergence value based on neighborhood comparisons
    """
    if len(sequence) < 2 * window_size:
        return 0.0
    
    vectors = create_sequence_vectors(sequence)
    
    if len(vectors) < 2 * window_size:
        return 0.0
    
    divergence_values = []
    
    for i in range(len(vectors) - 2 * window_size + 1):
        # Get two adjacent windows
        window1 = vectors[i:i + window_size]
        window2 = vectors[i + window_size:i + 2 * window_size]
        
        # Compute mean vectors
        mean1 = np.mean([v for v in window1 if np.linalg.norm(v) > 1e-6], axis=0)
        mean2 = np.mean([v for v in window2 if np.linalg.norm(v) > 1e-6], axis=0)
        
        if np.linalg.norm(mean1) < 1e-6 or np.linalg.norm(mean2) < 1e-6:
            continue
        
        # Normalize
        unit1 = mean1 / np.linalg.norm(mean1)
        unit2 = mean2 / np.linalg.norm(mean2)
        
        # Angle between neighborhoods
        cos_sim = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
        theta = np.arccos(cos_sim)
        theta_norm = theta / math.pi
        
        # Equilibrium reference
        equilibrium_angle_deg = 41.2
        theta_eq = equilibrium_angle_deg * (math.pi / 180.0)
        theta_eq_norm = theta_eq / math.pi
        
        # Divergence from equilibrium
        divergence = (theta_norm - theta_eq_norm) * 2.5
        
        divergence_values.append(divergence)
    
    if not divergence_values:
        return 0.0
    
    return float(np.mean(divergence_values))


def divergence_v2_pattern_weighted(sequence: List[int], window_size: int = 5) -> float:
    """
    Candidate 3: Pattern-Weighted Angular Divergence
    
    Weights divergence by local pattern structure.
    
    Args:
        sequence: Binary sequence
        window_size: Window size for pattern analysis
        
    Returns:
        Pattern-weighted divergence value
    """
    if len(sequence) < window_size:
        return 0.0
    
    vectors = create_sequence_vectors(sequence)
    
    if len(vectors) < window_size:
        return 0.0
    
    divergence_values = []
    
    for i in range(len(vectors) - window_size + 1):
        window_vecs = vectors[i:i + window_size]
        local_seq = sequence[i:i + window_size]
        
        # Compute mean vector
        mean_vec = np.mean([v for v in window_vecs if np.linalg.norm(v) > 1e-6], axis=0)
        
        if np.linalg.norm(mean_vec) < 1e-6:
            continue
        
        # Pattern features
        density = sum(local_seq) / len(local_seq)
        transitions = sum(1 for j in range(len(local_seq) - 1) if local_seq[j] != local_seq[j+1])
        transition_rate = transitions / (len(local_seq) - 1) if len(local_seq) > 1 else 0
        
        # Pattern weight (combines density and transitions)
        pattern_weight = density * (1 + transition_rate)
        
        # Base angle (from vector direction)
        unit_vec = mean_vec / np.linalg.norm(mean_vec)
        # Use first component as angle proxy
        angle_proxy = np.arctan2(unit_vec[1] if len(unit_vec) > 1 else 0, unit_vec[0])
        angle_norm = (angle_proxy + math.pi) / (2 * math.pi)  # Normalize to [0, 1]
        
        # Equilibrium reference
        equilibrium_angle_deg = 41.2
        theta_eq_norm = (equilibrium_angle_deg / 180.0)
        
        # Weighted divergence
        divergence = (angle_norm - theta_eq_norm) * pattern_weight * 2.5
        
        divergence_values.append(divergence)
    
    if not divergence_values:
        return 0.0
    
    return float(np.mean(divergence_values))


def test_divergence_v2_candidates(sequences: List[List[int]], names: List[str] = None) -> Dict:
    """
    Test all three divergence V2 candidates on sequences.
    
    Args:
        sequences: List of sequences to test
        names: Optional names for sequences
        
    Returns:
        Dict with results for each candidate
    """
    if names is None:
        names = [f"Sequence {i+1}" for i in range(len(sequences))]
    
    results = {
        'transition_based': [],
        'neighborhood': [],
        'pattern_weighted': []
    }
    
    print("Testing Divergence V2 Candidates:")
    print("="*70)
    
    for i, (seq, name) in enumerate(zip(sequences, names)):
        div1 = divergence_v2_transition_based(seq)
        div2 = divergence_v2_neighborhood(seq)
        div3 = divergence_v2_pattern_weighted(seq)
        
        results['transition_based'].append(div1)
        results['neighborhood'].append(div2)
        results['pattern_weighted'].append(div3)
        
        print(f"\n{name}:")
        print(f"  Transition-based:  {div1:.9f}")
        print(f"  Neighborhood:       {div2:.9f}")
        print(f"  Pattern-weighted:   {div3:.9f}")
    
    # Check variation
    print("\n" + "="*70)
    print("Variation Analysis:")
    print("="*70)
    
    for candidate_name, values in results.items():
        if values:
            std_val = np.std(values)
            min_val = min(values)
            max_val = max(values)
            range_val = max_val - min_val
            
            print(f"\n{candidate_name}:")
            print(f"  Std deviation: {std_val:.9f}")
            print(f"  Range: [{min_val:.9f}, {max_val:.9f}]")
            print(f"  Spread: {range_val:.9f}")
            
            if std_val > 1e-6:
                print(f"  ✓ VARIES across sequences")
            else:
                print(f"  ✗ Constant (no variation)")
    
    return results

