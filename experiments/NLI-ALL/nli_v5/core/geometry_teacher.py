"""
Geometry Teacher: Let Geometry Classify, Not Labels

This module implements the "geometry-first" philosophy:
- Geometry is stable and invariant
- Geometry produces meaning, labeling describes it
- Train classifier to read geometry, not force it
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .layers import Layer0Resonance, LayerState


@dataclass
class GeometryLabel:
    """Label produced by geometry, not dataset."""
    label: str  # 'entailment', 'contradiction', 'neutral'
    confidence: float
    divergence: float
    resonance: float
    curvature: float
    stability: float
    reason: str  # Why geometry chose this label


class GeometryTeacher:
    """
    Geometry Teacher: Uses raw geometry to classify.
    
    Philosophy: Geometry is the sensor, not the generator.
    We read what geometry tells us, not force it to match labels.
    """
    
    # Geometry zone thresholds (calibrated from invariant laws)
    DIVERGENCE_E_THRESHOLD = -0.1  # Negative divergence → Entailment
    DIVERGENCE_C_THRESHOLD = +0.1  # Positive divergence → Contradiction
    RESONANCE_E_THRESHOLD = 0.50   # High resonance → Entailment (when div < 0)
    
    def __init__(self):
        """Initialize geometry teacher."""
        self.layer0 = Layer0Resonance()
    
    def classify_from_geometry(
        self,
        encoded_pair,
        state: Optional[LayerState] = None
    ) -> GeometryLabel:
        """
        Classify based purely on geometry, ignoring dataset labels.
        
        This is the "geometry-first" classification:
        1. Compute raw geometry signals
        2. Let geometry zones determine classification
        3. Return geometry's label
        
        Args:
            encoded_pair: Encoded premise-hypothesis pair
            state: Optional pre-computed layer state
        
        Returns:
            GeometryLabel with geometry's classification
        """
        # Compute raw geometry (Layer 0)
        if state is None:
            state = LayerState()
        
        # Get resonance
        resonance = encoded_pair.get_resonance()
        state.resonance = resonance
        
        # Get word vectors for divergence computation
        p_vecs, h_vecs = encoded_pair.get_word_vectors()
        
        # Compute opposition first (needed for divergence)
        opposition = self.layer0._compute_opposition_field(p_vecs, h_vecs)
        
        # Compute divergence using angle-based method
        divergence = self.layer0._compute_field_divergence(
            p_vecs, h_vecs, opposition
        )
        
        # Compute curvature (gradient of divergence)
        # Simplified: use divergence magnitude as proxy
        curvature = abs(divergence) * 0.5
        
        # Compute stability (valley depth)
        stability = resonance - curvature - (divergence * 2.0)
        
        # Geometry zone classification
        geom_label, confidence, reason = self._classify_zone(
            divergence, resonance, curvature, stability
        )
        
        return GeometryLabel(
            label=geom_label,
            confidence=confidence,
            divergence=divergence,
            resonance=resonance,
            curvature=curvature,
            stability=stability,
            reason=reason
        )
    
    def _classify_zone(
        self,
        divergence: float,
        resonance: float,
        curvature: float,
        stability: float
    ) -> Tuple[str, float, str]:
        """
        Classify based on geometry zones.
        
        Rules (from invariant laws):
        - Negative divergence + High resonance → Entailment
        - Positive divergence → Contradiction
        - Near-zero divergence → Neutral
        
        Returns:
            (label, confidence, reason)
        """
        # Strong Entailment zone
        if divergence < self.DIVERGENCE_E_THRESHOLD and resonance > self.RESONANCE_E_THRESHOLD:
            confidence = min(1.0, abs(divergence) * 2.0 + (resonance - 0.5) * 2.0)
            reason = f"Strong inward pull (div={divergence:.3f}) + high resonance ({resonance:.3f})"
            return ("entailment", confidence, reason)
        
        # Strong Contradiction zone
        if divergence > self.DIVERGENCE_C_THRESHOLD:
            confidence = min(1.0, divergence * 2.0)
            reason = f"Strong outward push (div={divergence:.3f})"
            return ("contradiction", confidence, reason)
        
        # Weak Entailment (negative div but low resonance)
        if divergence < 0:
            confidence = min(0.7, abs(divergence) * 3.0)
            reason = f"Weak inward pull (div={divergence:.3f}), resonance={resonance:.3f}"
            return ("entailment", confidence, reason)
        
        # Neutral zone (near-zero divergence)
        if abs(divergence) < 0.12:
            confidence = 0.5 + (0.12 - abs(divergence)) * 2.0
            reason = f"Balanced forces (div={divergence:.3f})"
            return ("neutral", confidence, reason)
        
        # Weak Contradiction (positive but small)
        confidence = min(0.7, divergence * 3.0)
        reason = f"Weak outward push (div={divergence:.3f})"
        return ("contradiction", confidence, reason)
    
    def compare_with_dataset(
        self,
        geom_label: GeometryLabel,
        dataset_label: str
    ) -> Dict:
        """
        Compare geometry label with dataset label.
        
        Returns analysis of agreement/disagreement.
        """
        agreement = (geom_label.label == dataset_label)
        
        return {
            "agreement": agreement,
            "geometry_label": geom_label.label,
            "dataset_label": dataset_label,
            "geometry_confidence": geom_label.confidence,
            "geometry_reason": geom_label.reason,
            "divergence": geom_label.divergence,
            "resonance": geom_label.resonance,
            "stability": geom_label.stability
        }


def compute_geometry_labels(
    examples: list,
    encoder,
    show_progress: bool = True
) -> list:
    """
    Compute geometry labels for a list of examples.
    
    Args:
        examples: List of dicts with 'premise', 'hypothesis', 'label'
        encoder: ChainEncoder instance
        show_progress: Show progress bar
    
    Returns:
        List of GeometryLabel objects
    """
    from tqdm import tqdm
    
    teacher = GeometryTeacher()
    geom_labels = []
    
    iterator = tqdm(examples) if show_progress else examples
    
    for example in iterator:
        # Encode
        pair = encoder.encode_pair(
            example['premise'],
            example['hypothesis']
        )
        
        # Get geometry label
        geom_label = teacher.classify_from_geometry(pair)
        geom_labels.append(geom_label)
    
    return geom_labels


def analyze_geometry_dataset_alignment(
    examples: list,
    encoder,
    show_progress: bool = True
) -> Dict:
    """
    Analyze alignment between geometry labels and dataset labels.
    
    Returns statistics on:
    - Agreement rate
    - Per-class agreement
    - Disagreement patterns
    - Geometry confidence distribution
    """
    teacher = GeometryTeacher()
    geom_labels = compute_geometry_labels(examples, encoder, show_progress)
    
    # Compare with dataset
    comparisons = []
    for example, geom_label in zip(examples, geom_labels):
        comp = teacher.compare_with_dataset(geom_label, example['label'])
        comparisons.append(comp)
    
    # Statistics
    total = len(comparisons)
    agreements = sum(1 for c in comparisons if c['agreement'])
    agreement_rate = agreements / total if total > 0 else 0.0
    
    # Per-class statistics
    class_stats = {}
    for label in ['entailment', 'contradiction', 'neutral']:
        class_examples = [c for c in comparisons if c['dataset_label'] == label]
        if class_examples:
            class_agreements = sum(1 for c in class_examples if c['agreement'])
            class_stats[label] = {
                "total": len(class_examples),
                "agreements": class_agreements,
                "agreement_rate": class_agreements / len(class_examples),
                "avg_confidence": np.mean([c['geometry_confidence'] for c in class_examples])
            }
    
    # Disagreement analysis
    disagreements = [c for c in comparisons if not c['agreement']]
    disagreement_patterns = {}
    for d in disagreements:
        pattern = f"{d['geometry_label']}→{d['dataset_label']}"
        disagreement_patterns[pattern] = disagreement_patterns.get(pattern, 0) + 1
    
    return {
        "total_examples": total,
        "agreements": agreements,
        "agreement_rate": agreement_rate,
        "class_statistics": class_stats,
        "disagreement_patterns": disagreement_patterns,
        "avg_geometry_confidence": np.mean([c['geometry_confidence'] for c in comparisons]),
        "comparisons": comparisons
    }

