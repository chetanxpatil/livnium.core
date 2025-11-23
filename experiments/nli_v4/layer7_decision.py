"""
Layer 7: Decision Layer

Force-Competition Classifier

Planet Physics:
- Cold (E) vs Far (C) compete like two forces
- City (N) forms when forces are weak or balanced
- No argmax — geometry decides
"""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ClassificationResult:
    label: str                    # 'entailment', 'contradiction', 'neutral' (for backward compatibility)
    basin_index: int             # 0=cold, 1=far, 2=city (geometry-discovered cluster)
    confidence: float
    scores: Dict[str, float]
    is_moksha: bool
    layer_states: Dict


class Layer7Decision:
    """Pure force-competition final decision layer."""
    
    def __init__(self, use_rule_engine: bool = False, rule_engine=None, use_auto_classifier: bool = True):
        """
        Initialize Layer 7 decision layer.
        
        Args:
            use_rule_engine: If True, use symbolic rule engine instead of default logic
            rule_engine: Optional RuleEngine instance (creates default if None and use_rule_engine=True)
            use_auto_classifier: If True, use auto-discovered geometry classifier (recommended)
        """
        self.weak_force_threshold = 0.05     # Too small → city
        self.balance_threshold = 0.15        # Forces close → city
        
        # Auto classifier (geometry's discovered law - recommended)
        self.use_auto_classifier = use_auto_classifier
        if use_auto_classifier:
            from .layer7_auto_classifier import Layer7AutoClassifier, AutoRuleSet
            self.auto_classifier = Layer7AutoClassifier()
        else:
            self.auto_classifier = None
        
        # Rule engine (optional - for geometry-aligned symbolic rules)
        self.use_rule_engine = use_rule_engine and not use_auto_classifier  # Auto classifier takes precedence
        if use_rule_engine and rule_engine is None:
            from .rule_engine import RuleEngine
            self.rule_engine = RuleEngine.from_hand_tuned_rules()
        else:
            self.rule_engine = rule_engine

    def compute(self, x: Dict[str, float]) -> ClassificationResult:
        """Final classification based on force competition."""
        
        # Priority 1: Auto classifier (geometry's discovered law)
        if self.use_auto_classifier and self.auto_classifier is not None:
            return self._compute_with_auto_classifier(x)
        
        # Priority 2: Rule engine (symbolic rules)
        if self.use_rule_engine and self.rule_engine is not None:
            return self._compute_with_rules(x)
        
        # Priority 3: Default force-competition logic
        return self._compute_default(x)
    
    def _compute_with_auto_classifier(self, x: Dict[str, float]) -> ClassificationResult:
        """
        Compute classification using auto-discovered geometry classifier.
        
        This is geometry's own law - discovered, not hand-written.
        """
        # Extract forces
        cold = x.get('cold_attraction', x.get('e_attraction', 0.0))
        far = x.get('far_attraction', x.get('c_attraction', 0.0))
        e_score = x.get('e_score', 0.0)
        c_score = x.get('c_score', 0.0)
        n_score = x.get('n_score', 0.0)
        
        # Compute basin forces (normalized)
        total_force = cold + far + n_score
        if total_force > 0:
            cold_force = cold / total_force
            far_force = far / total_force
            city_force = n_score / total_force
        else:
            cold_force = far_force = city_force = 0.33
        
        max_force = max(cold, far)
        force_ratio = abs(cold - far) / max_force if max_force > 1e-6 else 0.0
        
        # Compute basin_conf from forces (this is the key signal)
        # High basin_conf = clear signal, low = uncertain
        # Use force dominance: how much does the strongest force exceed the others?
        max_basin_force = max(cold_force, far_force, city_force)
        second_basin_force = sorted([cold_force, far_force, city_force])[-2]
        force_dominance = max_basin_force - second_basin_force  # How much stronger is the winner?
        
        # basin_conf = strength of signal (0-1)
        # High when one force clearly dominates, low when forces are balanced
        basin_conf = max_basin_force + (force_dominance * 0.5)  # Boost for clear dominance
        basin_conf = min(1.0, basin_conf)  # Cap at 1.0
        
        # Build features for auto classifier
        features = {
            'basin_conf': float(basin_conf),
            'cold_force': float(cold_force),
            'far_force': float(far_force),
            'city_force': float(city_force),
            'resonance': float(x.get('resonance', 0.0)),
            'cold_attraction': float(cold),
            'far_attraction': float(far),
            'city_pull': float(x.get('city_pull', 0.0)),
            'max_force': float(max_force),
            'force_ratio': float(force_ratio)
        }
        
        # Classify using geometry's discovered law
        label_enc, confidence, rule_desc = self.auto_classifier.classify(features)
        
        # Map E/N/C to full labels
        label_map = {'E': 'entailment', 'N': 'neutral', 'C': 'contradiction'}
        label = label_map.get(label_enc, 'neutral')
        
        # Map to basin index
        basin_map = {'E': 0, 'C': 1, 'N': 2}
        basin_index = basin_map.get(label_enc, 2)
        
        # Normalize scores
        total = e_score + c_score + n_score
        if total > 0:
            e_score /= total
            c_score /= total
            n_score /= total
        
        scores = {
            'entailment': float(e_score),
            'contradiction': float(c_score),
            'neutral': float(n_score),
        }
        
        layer_states = {
            'resonance': float(x.get('resonance', 0.0)),
            'cold_attraction': float(cold),
            'far_attraction': float(far),
            'max_force': float(max_force),
            'attraction_ratio': float(force_ratio),
            'city_pull': x.get('city_pull', 0.0),
            'route': x.get('route', 'unknown'),
            'is_stable': x.get('is_stable', False),
            'decision_rule': 'auto_classifier',
            'rule_used': rule_desc,
            'basin_conf': float(basin_conf),
            'basin_forces': {
                'basin_0_cold': float(cold_force),
                'basin_1_far': float(far_force),
                'basin_2_city': float(city_force)
            }
        }
        
        return ClassificationResult(
            label=label,
            basin_index=basin_index,
            confidence=confidence,
            scores=scores,
            is_moksha=x.get('is_moksha', False),
            layer_states=layer_states
        )
    
    def _compute_with_rules(self, x: Dict[str, float]) -> ClassificationResult:
        """Compute classification using symbolic rule engine."""
        # Extract features for rule engine
        cold = x.get('cold_attraction', x.get('e_attraction', 0.0))
        far = x.get('far_attraction', x.get('c_attraction', 0.0))
        e_score = x.get('e_score', 0.0)
        c_score = x.get('c_score', 0.0)
        n_score = x.get('n_score', 0.0)
        
        # Compute basin forces
        total_force = cold + far + n_score
        if total_force > 0:
            cold_force = cold / total_force
            far_force = far / total_force
            city_force = n_score / total_force
        else:
            cold_force = far_force = city_force = 0.33
        
        max_force = max(cold, far)
        force_ratio = abs(cold - far) / max_force if max_force > 1e-6 else 0.0
        
        # Build features dict for rule engine
        features = {
            'basin_id': 0,  # Will be set after classification
            'basin_conf': 0.5,  # Will be computed
            'cold_attraction': cold,
            'far_attraction': far,
            'city_pull': x.get('city_pull', 0.0),
            'cold_force': cold_force,
            'far_force': far_force,
            'city_force': city_force,
            'resonance': x.get('resonance', 0.0),
            'curvature': x.get('curvature', 0.0),
            'max_force': max_force,
            'force_ratio': force_ratio,
            'cold_density': x.get('cold_density', 0.0),
            'distance': x.get('distance', 0.0),
            'e_score': e_score,
            'c_score': c_score,
            'n_score': n_score,
            'is_stable': x.get('is_stable', False),
            'is_moksha': x.get('is_moksha', False),
            'route': x.get('route', 'unknown')
        }
        
        # Classify using rule engine
        label_enc, confidence, rule_desc = self.rule_engine.classify(features)
        
        # Map E/N/C to full labels
        label_map = {'E': 'entailment', 'N': 'neutral', 'C': 'contradiction'}
        label = label_map.get(label_enc, 'neutral')
        
        # Map to basin index
        basin_map = {'E': 0, 'C': 1, 'N': 2}
        basin_index = basin_map.get(label_enc, 2)
        
        # Update features with computed values
        features['basin_id'] = basin_index
        features['basin_conf'] = confidence
        
        # Normalize scores
        total = e_score + c_score + n_score
        if total > 0:
            e_score /= total
            c_score /= total
            n_score /= total
        
        scores = {
            'entailment': float(e_score),
            'contradiction': float(c_score),
            'neutral': float(n_score),
        }
        
        layer_states = {
            'resonance': float(x.get('resonance', 0.0)),
            'cold_attraction': float(cold),
            'far_attraction': float(far),
            'max_force': float(max_force),
            'attraction_ratio': float(force_ratio),
            'city_pull': x.get('city_pull', 0.0),
            'route': x.get('route', 'unknown'),
            'is_stable': x.get('is_stable', False),
            'decision_rule': 'symbolic_rules',
            'rule_used': rule_desc,
            'basin_forces': {
                'basin_0_cold': float(cold_force),
                'basin_1_far': float(far_force),
                'basin_2_city': float(city_force)
            }
        }
        
        return ClassificationResult(
            label=label,
            basin_index=basin_index,
            confidence=confidence,
            scores=scores,
            is_moksha=x.get('is_moksha', False),
            layer_states=layer_states
        )
    
    def _compute_default(self, x: Dict[str, float]) -> ClassificationResult:
        """
        Default force-competition logic.
        
        Layer 7 is now AUTHORITATIVE - it makes the final label decision
        based purely on forces, not pre-decided labels from geometry.
        """
        # Force values coming from Layer 6 (pure forces, no labels)
        cold = x.get('cold_attraction', x.get('e_attraction', 0.0))
        far = x.get('far_attraction', x.get('c_attraction', 0.0))
        city_pull = x.get('city_pull', 0.0)

        e_score = x.get('e_score', 0.0)
        c_score = x.get('c_score', 0.0)
        n_score = x.get('n_score', 0.0)

        is_moksha = x.get('is_moksha', False)
        resonance = x.get('resonance', 0.0)

        max_force = max(cold, far)
        
        # Compute basin forces (normalized)
        total_force = cold + far + n_score
        if total_force > 0:
            cold_force = cold / total_force
            far_force = far / total_force
            city_force = n_score / total_force
        else:
            cold_force = far_force = city_force = 0.33
        
        # ============================================================
        # LAYER 7 AUTHORITATIVE DECISION (based on forces only)
        # ============================================================
        # Geometry provides forces, Layer 7 decides the label
        # This is the separation: geometry = meaning, rules = interpretation
        
        # Rule 1: Weak forces → Neutral
        if max_force < self.weak_force_threshold:
            label = 'neutral'
            basin_index = 2  # City basin
            confidence = max(n_score, 0.5)
        else:
            # Ratio of difference (0 = perfectly balanced)
            ratio = abs(cold - far) / max_force if max_force > 1e-6 else 0.0

            # Rule 2: Balanced forces → Neutral
            if ratio < self.balance_threshold:
                label = 'neutral'
                basin_index = 2  # City basin
                balance_conf = 1.0 - (ratio / self.balance_threshold) if self.balance_threshold > 0 else 0.5
                confidence = max(n_score, balance_conf)
            # Rule 3: City pull dominates → Neutral
            elif city_force > 0.6:
                label = 'neutral'
                basin_index = 2  # City basin
                confidence = city_force
            # Rule 4: One side wins (Cold vs Far)
            else:
                if cold > far:
                    label = 'entailment'
                    basin_index = 0  # Cold basin
                    confidence = max(e_score, cold_force)
                else:
                    label = 'contradiction'
                    basin_index = 1  # Far basin
                    confidence = max(c_score, far_force)

        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Normalize scores
        total = e_score + c_score + n_score
        if total > 0:
            e_score /= total
            c_score /= total
            n_score /= total

        scores = {
            'entailment': float(e_score),
            'contradiction': float(c_score),
            'neutral': float(n_score),
        }
        
        layer_states = {
            'resonance': float(resonance),
            'cold_attraction': float(cold),
            'far_attraction': float(far),
            'max_force': float(max_force),
            'attraction_ratio': float(abs(cold - far) / max_force) if max_force > 1e-6 else 0.0,
            'city_pull': x.get('city_pull', 0.0),
            'route': x.get('route', 'unknown'),
            'is_stable': x.get('is_stable', False),
            'decision_rule': 'force_competition',
            # Basin forces (geometry-discovered)
            'basin_forces': {
                'basin_0_cold': float(cold_force),
                'basin_1_far': float(far_force),
                'basin_2_city': float(city_force)
            }
        }
        
        return ClassificationResult(
            label=label,  # Backward compatibility
            basin_index=basin_index,  # Geometry-discovered cluster
            confidence=confidence,
            scores=scores,
            is_moksha=is_moksha,
            layer_states=layer_states
        )
    
    def decide(self, cold_force: float, far_force: float, city_force: float) -> str:
        """
        Simple force-competition rule to get E/N/C label.
        
        This is the clean truth: meaning = strongest force direction.
        - Cold force → Entailment (E)
        - Far force → Contradiction (C)
        - City force → Neutral (N)
        
        Args:
            cold_force: Normalized cold attraction force
            far_force: Normalized far attraction force
            city_force: Normalized city pull force
            
        Returns:
            'E', 'N', or 'C' label
        """
        # Simple rule: strongest force wins
        if city_force > max(cold_force, far_force):
            return "N"
        elif cold_force > far_force:
            return "E"
        else:
            return "C"
    
    def get_state(self) -> Dict:
        return {}
