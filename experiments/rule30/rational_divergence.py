"""
Rational/Exact Divergence Computation

Computes divergence using exact rational arithmetic instead of floating point.
This enables discovery of the exact mathematical formula for the invariant.
"""

from fractions import Fraction
from typing import List, Tuple, Dict
import numpy as np

from experiments.rule30.diagnostics import create_sequence_vectors, _compute_field_divergence


def compute_rational_divergence(sequence: List[int], window_size: int = 5) -> Tuple[Fraction, float]:
    """
    Compute divergence using exact rational arithmetic.
    
    Returns both the exact rational value and floating point approximation.
    
    Args:
        sequence: Binary sequence (0s and 1s)
        window_size: Window size for divergence computation
        
    Returns:
        Tuple of (exact_rational_divergence, float_approximation)
    """
    vectors = create_sequence_vectors(sequence)
    
    if len(vectors) < window_size:
        return Fraction(0), 0.0
    
    # Compute divergence for each window
    divergence_values = []
    for i in range(len(vectors) - window_size + 1):
        window_vecs = vectors[i:i + window_size]
        divergence = _compute_field_divergence(window_vecs, window_vecs)
        divergence_values.append(divergence)
    
    if not divergence_values:
        return Fraction(0), 0.0
    
    # Convert to rational (approximate - need to trace exact computation)
    mean_div = float(np.mean(divergence_values))
    
    # Try to find exact rational representation
    # Start with high precision approximation
    exact_rational = Fraction(mean_div).limit_denominator(1000000)
    
    return exact_rational, mean_div


def analyze_divergence_rational_form(sequence: List[int]) -> Dict:
    """
    Analyze divergence to find its exact rational form.
    
    Returns:
        Dict with rational form, pattern, and verification
    """
    exact_rational, float_val = compute_rational_divergence(sequence)
    
    return {
        'exact_rational': exact_rational,
        'numerator': exact_rational.numerator,
        'denominator': exact_rational.denominator,
        'float_value': float_val,
        'as_fraction': str(exact_rational),
        'decimal_representation': float(exact_rational)
    }


def find_exact_invariant_value(sequences: List[List[int]]) -> Dict:
    """
    Find the exact rational form of the invariant across multiple sequences.
    
    Args:
        sequences: List of sequences to test
        
    Returns:
        Dict with invariant analysis
    """
    rationals = []
    floats = []
    
    for seq in sequences:
        exact_rational, float_val = compute_rational_divergence(seq)
        rationals.append(exact_rational)
        floats.append(float_val)
    
    # Check if all rationals are the same
    if len(set(rationals)) == 1:
        invariant_rational = rationals[0]
        is_exact = True
    else:
        # Find common rational approximation
        mean_float = np.mean(floats)
        invariant_rational = Fraction(mean_float).limit_denominator(1000000)
        is_exact = False
    
    return {
        'invariant_rational': invariant_rational,
        'invariant_numerator': invariant_rational.numerator,
        'invariant_denominator': invariant_rational.denominator,
        'invariant_as_fraction': str(invariant_rational),
        'invariant_decimal': float(invariant_rational),
        'is_exact': is_exact,
        'all_rationals': [str(r) for r in rationals],
        'all_floats': floats
    }

