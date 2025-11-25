"""
Divergence V3: Parameterized Pattern-Based Divergence

Defines divergence as a weighted sum over 3-bit windows:
D3(s) = Σ_p w_p · freq_p(s)

This is a parameterized form that we can solve for invariance using linear algebra.
"""

from typing import Dict, List, Tuple

Pattern = Tuple[int, int, int]


def enumerate_patterns() -> List[Pattern]:
    """Return all 3-bit patterns as tuples (0/1)."""
    patterns = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                patterns.append((a, b, c))
    return patterns


def pattern_counts_3(row: List[int], cyclic: bool = True) -> Dict[Pattern, int]:
    """
    Count occurrences of each 3-bit window in a row.
    
    Args:
        row: Binary row
        cyclic: If True, use wraparound (cyclic boundary)
        
    Returns:
        Dict mapping pattern tuples to counts
    """
    n = len(row)
    patterns = enumerate_patterns()
    counts = {p: 0 for p in patterns}
    
    for i in range(n):
        if cyclic:
            a = row[(i - 1) % n]
            b = row[i]
            c = row[(i + 1) % n]
        else:
            a = row[i - 1] if i > 0 else 0
            b = row[i]
            c = row[i + 1] if i < n - 1 else 0
        
        pattern = (a, b, c)
        counts[pattern] += 1
    
    return counts


def pattern_frequencies_3(row: List[int], cyclic: bool = True) -> Dict[Pattern, float]:
    """
    Compute normalized frequencies of 3-bit patterns.
    
    Args:
        row: Binary row
        cyclic: Use cyclic boundary conditions
        
    Returns:
        Dict mapping pattern tuples to frequencies (0.0 to 1.0)
    """
    counts = pattern_counts_3(row, cyclic=cyclic)
    n = len(row)
    
    if n == 0:
        return {p: 0.0 for p in enumerate_patterns()}
    
    frequencies = {pattern: count / n for pattern, count in counts.items()}
    return frequencies


def divergence_v3(row: List[int], weights: Dict[Pattern, float], cyclic: bool = True) -> float:
    """
    General 3-pattern divergence: D3(row) = Σ_p w_p · freq_p(row)
    
    Args:
        row: Binary row
        weights: Dict mapping patterns to weights
        cyclic: Use cyclic boundary conditions
        
    Returns:
        Divergence value
    """
    frequencies = pattern_frequencies_3(row, cyclic=cyclic)
    
    total = 0.0
    for pattern, freq in frequencies.items():
        w = weights.get(pattern, 0.0)
        total += w * freq
    
    return total


def format_weights(weights: Dict[Pattern, float]) -> str:
    """Format weights as a readable formula."""
    terms = []
    for pattern, w in sorted(weights.items()):
        if abs(w) > 1e-6:
            pattern_str = ''.join(str(b) for b in pattern)
            if abs(w - 1.0) < 1e-6:
                terms.append(f"freq('{pattern_str}')")
            elif abs(w + 1.0) < 1e-6:
                terms.append(f"-freq('{pattern_str}')")
            else:
                terms.append(f"{w:.6f}*freq('{pattern_str}')")
    
    if not terms:
        return "0"
    
    return " + ".join(terms)

