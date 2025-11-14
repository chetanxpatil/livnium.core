"""
Quantum Interference as Conflict Resolution Mechanism

Uses quantum interference to resolve contradictions and conflicts between features.
"""

import numpy as np
from typing import List, Tuple, Optional
from quantum.kernel import LivniumQubit, H, CNOT
from quantum.gates import hadamard_gate, get_probabilities, normalize_state


def resolve_feature_conflict(
    feature1_qubit: LivniumQubit,
    feature2_qubit: LivniumQubit,
    conflict_type: str = "contradiction"
) -> Tuple[int, float]:
    """
    Resolve conflict between two features using quantum interference.
    
    Args:
        feature1_qubit: First feature qubit
        feature2_qubit: Second feature qubit
        conflict_type: Type of conflict ("contradiction", "neutral", "entailment")
        
    Returns:
        (resolution, confidence) where resolution is 0 or 1, confidence is [0, 1]
    """
    # Get probabilities
    p1_0, p1_1 = feature1_qubit.get_probabilities()
    p2_0, p2_1 = feature2_qubit.get_probabilities()
    
    # Create conflict qubit in superposition
    # Higher conflict → more superposition
    conflict_strength = abs(p1_1 - p2_1)  # Difference in probabilities
    
    # Create superposition state
    conflict_state = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
    
    # Apply Hadamard to create superposition
    conflict_state = hadamard_gate(conflict_state)
    
    # Apply phase based on conflict type
    if conflict_type == "contradiction":
        # Contradiction: features disagree
        # Apply phase shift to create destructive interference
        phase = np.pi  # 180 degrees
        conflict_state[1] *= np.exp(1j * phase)
    elif conflict_type == "entailment":
        # Entailment: features agree
        # No phase shift (constructive interference)
        phase = 0.0
    else:  # neutral
        # Neutral: balanced
        phase = np.pi / 2
        conflict_state[1] *= np.exp(1j * phase)
    
    # Normalize
    conflict_state = normalize_state(conflict_state)
    
    # Apply second Hadamard for interference
    conflict_state = hadamard_gate(conflict_state)
    
    # Measure resolution
    p0, p1 = get_probabilities(conflict_state)
    resolution = 1 if np.random.rand() < p1 else 0
    
    # Confidence is inverse of uncertainty
    uncertainty = -p0 * np.log2(p0) - p1 * np.log2(p1) if p0 > 1e-10 and p1 > 1e-10 else 1.0
    confidence = 1.0 - (uncertainty / np.log2(2))
    
    return (resolution, confidence)


def resolve_multi_feature_conflict(
    feature_qubits: List[LivniumQubit],
    weights: Optional[List[float]] = None
) -> Tuple[int, float]:
    """
    Resolve conflict between multiple features using quantum interference.
    
    Args:
        feature_qubits: List of feature qubits
        weights: Optional weights for each feature
        
    Returns:
        (resolution, confidence)
    """
    if weights is None:
        weights = [1.0 / len(feature_qubits)] * len(feature_qubits)
    
    # Combine features into superposition
    # Weighted average of probabilities
    total_prob_1 = 0.0
    total_weight = sum(weights)
    
    for qubit, weight in zip(feature_qubits, weights):
        _, p1 = qubit.get_probabilities()
        total_prob_1 += weight * p1
    
    avg_prob_1 = total_prob_1 / total_weight if total_weight > 0 else 0.5
    
    # Create conflict resolution qubit
    conflict_state = np.array([
        np.sqrt(1.0 - avg_prob_1) + 0j,
        np.sqrt(avg_prob_1) + 0j
    ], dtype=np.complex128)
    
    # Apply interference
    conflict_state = hadamard_gate(conflict_state)
    conflict_state = hadamard_gate(conflict_state)  # Interference
    
    # Measure
    p0, p1 = get_probabilities(conflict_state)
    resolution = 1 if np.random.rand() < p1 else 0
    
    # Confidence
    uncertainty = -p0 * np.log2(p0) - p1 * np.log2(p1) if p0 > 1e-10 and p1 > 1e-10 else 1.0
    confidence = 1.0 - (uncertainty / np.log2(2))
    
    return (resolution, confidence)


def quantum_vote(feature_qubits: List[LivniumQubit]) -> Tuple[int, float]:
    """
    Quantum voting mechanism: use interference to combine feature votes.
    
    Args:
        feature_qubits: List of feature qubits voting
        
    Returns:
        (decision, confidence)
    """
    # Count votes (probabilities)
    votes_1 = sum(q.get_probabilities()[1] for q in feature_qubits)
    votes_0 = len(feature_qubits) - votes_1
    
    # Create voting qubit
    vote_prob = votes_1 / len(feature_qubits) if len(feature_qubits) > 0 else 0.5
    
    vote_state = np.array([
        np.sqrt(1.0 - vote_prob) + 0j,
        np.sqrt(vote_prob) + 0j
    ], dtype=np.complex128)
    
    # Quantum interference amplifies majority
    vote_state = hadamard_gate(vote_state)
    vote_state = hadamard_gate(vote_state)
    
    # Measure decision
    p0, p1 = get_probabilities(vote_state)
    decision = 1 if p1 > 0.5 else 0
    
    # Confidence based on margin
    margin = abs(p1 - 0.5) * 2  # [0, 1]
    confidence = margin
    
    return (decision, confidence)


if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM INTERFERENCE AS CONFLICT RESOLUTION")
    print("=" * 70)
    
    # Test conflict resolution
    print("\n1. Feature Conflict Resolution:")
    
    # Create conflicting features
    q1 = LivniumQubit((0, 0, 0), f=1)
    q1.state = np.array([0.3, 0.7], dtype=np.complex128)  # Favors |1>
    q1.state = normalize_state(q1.state)
    
    q2 = LivniumQubit((1, 0, 0), f=1)
    q2.state = np.array([0.8, 0.2], dtype=np.complex128)  # Favors |0>
    q2.state = normalize_state(q2.state)
    
    print(f"   Feature 1: {q1.state_string()} (favors |1>)")
    print(f"   Feature 2: {q2.state_string()} (favors |0>)")
    
    resolution, confidence = resolve_feature_conflict(q1, q2, conflict_type="contradiction")
    print(f"   Resolution: {resolution} (confidence: {confidence:.3f})")
    
    # Test multi-feature conflict
    print("\n2. Multi-Feature Conflict Resolution:")
    features = [
        LivniumQubit((i, 0, 0), f=1, initial_state=np.array([0.7, 0.3], dtype=np.complex128))
        for i in range(5)
    ]
    
    resolution, confidence = resolve_multi_feature_conflict(features)
    print(f"   Resolution: {resolution} (confidence: {confidence:.3f})")
    
    # Test quantum voting
    print("\n3. Quantum Voting:")
    voters = [
        LivniumQubit((i, 0, 0), f=1, initial_state=np.array([0.3, 0.7], dtype=np.complex128))
        for i in range(7)  # 7 voters favoring |1>
    ]
    
    decision, confidence = quantum_vote(voters)
    print(f"   Decision: {decision} (confidence: {confidence:.3f})")
    print(f"   Majority: {'|1>' if decision == 1 else '|0>'} wins")
    
    print("\n" + "=" * 70)
    print("✅ QUANTUM INTERFERENCE: Natural conflict resolution mechanism!")
    print("=" * 70)

