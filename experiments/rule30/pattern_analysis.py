"""
Pattern Frequency Analysis

Analyzes how divergence relates to pattern frequencies in sequences.
Goal: Find exact linear combination D(s) = Σ α_p · freq_p(s)
"""

from typing import List, Dict, Tuple
from collections import Counter
import numpy as np
from scipy import stats


def count_patterns(sequence: List[int], pattern_length: int = 3) -> Dict[str, int]:
    """
    Count all patterns of given length in sequence.
    
    Args:
        sequence: Binary sequence
        pattern_length: Length of patterns to count (default: 3)
        
    Returns:
        Dict mapping pattern strings to counts
    """
    if len(sequence) < pattern_length:
        return {}
    
    patterns = []
    for i in range(len(sequence) - pattern_length + 1):
        pattern = tuple(sequence[i:i + pattern_length])
        patterns.append(pattern)
    
    counts = Counter(patterns)
    
    # Convert to string keys
    result = {}
    for pattern, count in counts.items():
        key = ''.join(str(b) for b in pattern)
        result[key] = count
    
    return result


def compute_pattern_frequencies(sequence: List[int], pattern_length: int = 3) -> Dict[str, float]:
    """
    Compute normalized frequencies of patterns.
    
    Returns:
        Dict mapping pattern strings to frequencies (0.0 to 1.0)
    """
    counts = count_patterns(sequence, pattern_length)
    total = sum(counts.values())
    
    if total == 0:
        return {}
    
    frequencies = {pattern: count / total for pattern, count in counts.items()}
    return frequencies


def analyze_divergence_vs_patterns(
    sequences: List[List[int]],
    divergences: List[float],
    pattern_length: int = 3
) -> Dict:
    """
    Analyze if divergence is a linear combination of pattern frequencies.
    
    Tries to fit: D(s) = Σ α_p · freq_p(s)
    
    Args:
        sequences: List of sequences
        divergences: List of divergence values for each sequence
        pattern_length: Length of patterns to analyze
        
    Returns:
        Dict with regression results and coefficients
    """
    # Collect pattern frequencies for all sequences
    all_patterns = set()
    pattern_freqs_list = []
    
    for seq in sequences:
        freqs = compute_pattern_frequencies(seq, pattern_length)
        pattern_freqs_list.append(freqs)
        all_patterns.update(freqs.keys())
    
    # Ensure all sequences have same pattern keys (pad with 0)
    all_patterns = sorted(all_patterns)
    
    # Build feature matrix
    X = []
    for freqs in pattern_freqs_list:
        row = [freqs.get(pattern, 0.0) for pattern in all_patterns]
        X.append(row)
    
    X = np.array(X)
    y = np.array(divergences)
    
    # Linear regression
    if len(all_patterns) > 0 and X.shape[0] > X.shape[1]:
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.T[0] if X.shape[1] == 1 else X.mean(axis=1), y)
            
            # Try multivariate regression if we have multiple patterns
            if X.shape[1] > 1:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(X, y)
                coefficients = reg.coef_
                intercept = reg.intercept_
                r_squared = reg.score(X, y)
            else:
                coefficients = [slope] if X.shape[1] == 1 else []
                r_squared = r_value ** 2
            
            # Check if fit is exact (within numerical precision)
            y_pred = X @ coefficients + intercept if len(coefficients) > 0 else intercept + X.mean(axis=1) * slope
            residuals = y - y_pred
            max_residual = np.max(np.abs(residuals))
            is_exact = max_residual < 1e-10
            
            return {
                'patterns': all_patterns,
                'coefficients': coefficients.tolist() if hasattr(coefficients, 'tolist') else list(coefficients),
                'intercept': float(intercept),
                'r_squared': float(r_squared),
                'max_residual': float(max_residual),
                'is_exact': is_exact,
                'formula': _format_formula(all_patterns, coefficients, intercept),
                'residuals': residuals.tolist()
            }
        except Exception as e:
            return {'error': str(e)}
    
    return {'error': 'Insufficient data for regression'}


def _format_formula(patterns: List[str], coefficients: List[float], intercept: float) -> str:
    """Format regression result as mathematical formula."""
    terms = []
    for pattern, coeff in zip(patterns, coefficients):
        if abs(coeff) > 1e-10:
            terms.append(f"{coeff:.6f} * freq('{pattern}')")
    
    if abs(intercept) > 1e-10:
        terms.append(f"{intercept:.6f}")
    
    if not terms:
        return "0"
    
    return " + ".join(terms)


def find_invariant_pattern_combination(
    rule30_sequences: List[List[int]],
    random_sequences: List[List[int]] = None
) -> Dict:
    """
    Find the exact pattern combination that gives the invariant.
    
    Tests Rule 30 sequences and optionally random sequences to find
    the linear combination that is conserved.
    
    Args:
        rule30_sequences: List of Rule 30 center column sequences
        random_sequences: Optional list of random sequences for comparison
        
    Returns:
        Dict with invariant formula and verification
    """
    # Compute divergences
    from experiments.rule30.validate_invariant_direct import compute_direct_divergence
    
    rule30_divs = [compute_direct_divergence(seq) for seq in rule30_sequences]
    
    # Analyze pattern frequencies
    result = analyze_divergence_vs_patterns(rule30_sequences, rule30_divs)
    
    # Verify invariance
    if 'formula' in result:
        # Check if formula gives constant value for Rule 30
        rule30_freqs = [compute_pattern_frequencies(seq) for seq in rule30_sequences]
        predicted_divs = []
        
        for freqs in rule30_freqs:
            pred = result['intercept']
            for pattern, coeff in zip(result['patterns'], result['coefficients']):
                pred += coeff * freqs.get(pattern, 0.0)
            predicted_divs.append(pred)
        
        result['predicted_divergences'] = predicted_divs
        result['actual_divergences'] = rule30_divs
        result['invariant_value'] = np.mean(rule30_divs)
        result['is_invariant'] = np.std(predicted_divs) < 1e-10
    
    return result

