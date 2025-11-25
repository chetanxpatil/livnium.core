"""
Rule 30 Algebra: Core Update Rule

Exposes Rule 30's update rule in a way we can use symbolically.
This is the "physics" primitive that all divergence functions must respect.
"""

from typing import Tuple, List


# Truth table for Rule 30: (left, center, right) -> new bit
# Rule 30: 111→0, 110→0, 101→0, 100→1, 011→1, 010→1, 001→1, 000→0
RULE30_TABLE = {
    (1, 1, 1): 0,
    (1, 1, 0): 0,
    (1, 0, 1): 0,
    (1, 0, 0): 1,
    (0, 1, 1): 1,
    (0, 1, 0): 1,
    (0, 0, 1): 1,
    (0, 0, 0): 0,
}


def rule30_step(row: List[int], cyclic: bool = True) -> List[int]:
    """
    One step of Rule 30 on a finite row.
    
    Args:
        row: Current row as list of 0s and 1s
        cyclic: If True, use wraparound (cyclic boundary). 
                If False, pad with zeros at boundaries.
        
    Returns:
        Next row after applying Rule 30
    """
    n = len(row)
    nxt = []
    
    for i in range(n):
        if cyclic:
            left = row[(i - 1) % n]
            center = row[i]
            right = row[(i + 1) % n]
        else:
            left = row[i - 1] if i > 0 else 0
            center = row[i]
            right = row[i + 1] if i < n - 1 else 0
        
        nxt.append(RULE30_TABLE[(left, center, right)])
    
    return nxt


def rule30_evolve(row: List[int], steps: int, cyclic: bool = True) -> List[List[int]]:
    """
    Evolve a row under Rule 30 for multiple steps.
    
    Args:
        row: Initial row
        steps: Number of evolution steps
        cyclic: Use cyclic boundary conditions
        
    Returns:
        List of rows (history of evolution)
    """
    history = [row.copy()]
    current = row.copy()
    
    for _ in range(steps):
        current = rule30_step(current, cyclic=cyclic)
        history.append(current.copy())
    
    return history


def get_rule30_table() -> dict:
    """Return the Rule 30 truth table."""
    return RULE30_TABLE.copy()

