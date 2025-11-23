"""
Layered Livnium Classifier: Complete 7-Layer Architecture

Each layer builds on the one below. Gravity shapes everything.
No manual tuning. Self-correcting.
"""

from typing import Set, Dict
from dataclasses import dataclass
import numpy as np

from experiments.nli_v3.chain_encoder import ChainEncodedPair

from .layer0_resonance import Layer0Resonance
from .layer1_curvature import Layer1Curvature
from .layer2_basin import Layer2Basin
from .layer3_valley import Layer3Valley
from .layer4_meta_routing import Layer4MetaRouting
from .layer5_temporal_stability import Layer5TemporalStability
from .layer6_semantic_memory import Layer6SemanticMemory
from .layer7_decision import Layer7Decision, ClassificationResult
from .auto_physics import AutoPhysicsEngine
from .autonomous_meaning_engine import AutonomousMeaningEngine


class LayeredLivniumClassifier:
    """
    Layered Livnium Classifier - 7-layer geological architecture.
    
    Each layer builds on the one below. Gravity shapes everything.
    No manual tuning. Self-correcting.
    """
    
    def __init__(self, encoded_pair: ChainEncodedPair):
        """
        Initialize layered classifier.
        
        Args:
            encoded_pair: Chain-encoded premise-hypothesis pair
        """
        self.encoded_pair = encoded_pair
        
        # Initialize all layers
        self.layer0 = Layer0Resonance()
        self.layer1 = Layer1Curvature()
        self.layer2 = Layer2Basin()
        self.layer3 = Layer3Valley()
        self.layer4 = Layer4MetaRouting()
        self.layer5 = Layer5TemporalStability()
        self.layer6 = Layer6SemanticMemory()
        # Layer 7 uses auto classifier (geometry's discovered law)
        # This is geometry writing its own rules - discovered, not hand-written
        self.layer7 = Layer7Decision(use_auto_classifier=True)
        
        # Auto-physics engine (self-organizing universe)
        self.auto_physics = AutoPhysicsEngine()
        
        # Autonomous Meaning Engine (full semantic cosmology)
        self.ame = AutonomousMeaningEngine()
    
    def classify(self) -> ClassificationResult:
        """
        Classify using full 7-layer stack.
        
        Returns:
            ClassificationResult with final decision
        """
        # Layer 0: Pure Resonance
        l0_output = self.layer0.compute(self.encoded_pair)
        
        # Layer 1: Curvature
        l1_output = self.layer1.compute(l0_output)
        
        # Layer 2: Basin
        l2_output = self.layer2.compute(l1_output)
        
        # Layer 3: Valley
        l3_output = self.layer3.compute(l2_output)
        
        # Layer 4: Meta Routing
        l4_output = self.layer4.compute(l3_output)
        
        # Layer 5: Temporal Stability
        l5_output = self.layer5.compute(l4_output)
        
        # Layer 6: Semantic Memory
        tokens = set(self.encoded_pair.premise.tokens + self.encoded_pair.hypothesis.tokens)
        l6_output = self.layer6.compute(l5_output, tokens)
        
        # Layer 7: Decision
        result = self.layer7.compute(l6_output)
        
        # Track prediction for auto-physics (use basin_index for unsupervised compatibility)
        # Map basin_index to label for tracking
        basin_to_label = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
        label_for_tracking = basin_to_label.get(result.basin_index, result.label)
        Layer2Basin.track_prediction(label_for_tracking)
        
        # Run auto-physics step (closes thermodynamic loop)
        # This applies the three laws automatically:
        # 1. Automatic entropy injection (scales with imbalance)
        # 2. Repulsion field for contradiction
        # 3. Dynamic basin depth (anti-monopoly)
        physics_state = self.auto_physics.step(self)
        
        # Run Autonomous Meaning Engine (AME) - full semantic cosmology
        # This applies all 7 steps of meaning emergence:
        # 1. Semantic turbulence
        # 2. Competitive word polarity
        # 3. Dynamic basin splitting
        # 4. Curvature-based routing
        # 5. Memory hysteresis
        # 6. Long-range alignment
        # 7. Continuous evolution
        
        # Create sentence hash for hysteresis
        # Use tokens for stable hashing
        p_tokens = ' '.join(sorted(self.encoded_pair.premise.tokens))
        h_tokens = ' '.join(sorted(self.encoded_pair.hypothesis.tokens))
        sentence_hash = f"{p_tokens}|||{h_tokens}"
        
        # Get resonance and curvature for AME
        resonance = l0_output.get('resonance', 0.0)
        curvature = l1_output.get('curvature', 0.0)
        
        # Run AME step
        ame_state = self.ame.step(
            classifier=self,
            resonance=resonance,
            curvature=curvature,
            current_basin=result.basin_index,
            sentence_hash=sentence_hash
        )
        
        # Update result with AME-modified basin
        if ame_state.get('final_basin') != result.basin_index:
            # AME modified the basin assignment
            result.basin_index = ame_state['final_basin']
            # Update label for backward compatibility
            result.label = basin_to_label.get(result.basin_index, 'neutral')
        
        # Store all states in result for monitoring
        result.layer_states.update(physics_state)
        result.layer_states.update(ame_state)
        
        return result
    
    def apply_learning_feedback(self, correct_label: str, learning_strength: float = 1.0):
        """
        Apply learning feedback to all layers.
        
        Args:
            correct_label: Correct label ('entailment', 'contradiction', or 'neutral')
            learning_strength: Learning strength
        """
        label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        correct_idx = label_map.get(correct_label, 2)
        
        # Layer 2: Reinforce basin (E or C only)
        if correct_idx in [0, 1]:
            self.layer2.reinforce(correct_idx, strength=learning_strength)
        
        # Layer 6: Update semantic memory
        tokens = set(self.encoded_pair.premise.tokens + self.encoded_pair.hypothesis.tokens)
        self.layer6.update(tokens, correct_label, strength=learning_strength)
    
    def get_all_states(self) -> dict:
        """Get states from all layers (for debugging)."""
        return {
            'layer0': self.layer0.get_state(),
            'layer1': self.layer1.get_state(),
            'layer2': self.layer2.get_state(),
            'layer3': self.layer3.get_state(),
            'layer4': self.layer4.get_state(),
            'layer5': self.layer5.get_state(),
            'layer6': self.layer6.get_state(),
            'layer7': self.layer7.get_state()
        }
    
    def extract_geometric_features(self, result: ClassificationResult) -> dict:
        """
        Extract geometric features for rule discovery.
        
        This is the "priest of rules" observing the universe.
        Geometry stays wild; we just read its signals.
        
        Args:
            result: ClassificationResult from classify()
            
        Returns:
            Dict of geometric features for rule learning
        """
        layer_states = result.layer_states
        
        # Core forces
        cold_attraction = layer_states.get('cold_attraction', 0.0)
        far_attraction = layer_states.get('far_attraction', 0.0)
        city_pull = layer_states.get('city_pull', 0.0)
        
        # Basin info
        basin_id = result.basin_index
        basin_conf = result.confidence
        
        # Geometry signals
        resonance = layer_states.get('resonance', 0.0)
        # Curvature might be in layer states or need to compute from layer outputs
        # For now, try to get it from states, default to 0.0
        curvature = layer_states.get('curvature', 0.0)
        max_force = layer_states.get('max_force', 0.0)
        force_ratio = layer_states.get('attraction_ratio', 0.0)
        
        # Additional geometry signals that might be available
        cold_density = layer_states.get('cold_density', 0.0)
        distance = layer_states.get('distance', 0.0)
        
        # Basin forces (normalized)
        basin_forces = layer_states.get('basin_forces', {})
        cold_force = basin_forces.get('basin_0_cold', 0.33)
        far_force = basin_forces.get('basin_1_far', 0.33)
        city_force = basin_forces.get('basin_2_city', 0.33)
        
        # Additional signals
        route = layer_states.get('route', 'unknown')
        is_stable = layer_states.get('is_stable', False)
        is_moksha = result.is_moksha
        
        # Scores
        scores = result.scores
        e_score = scores.get('entailment', 0.0)
        c_score = scores.get('contradiction', 0.0)
        n_score = scores.get('neutral', 0.0)
        
        return {
            # Basin assignment
            'basin_id': int(basin_id),
            'basin_conf': float(basin_conf),
            
            # Core forces
            'cold_attraction': float(cold_attraction),
            'far_attraction': float(far_attraction),
            'city_pull': float(city_pull),
            
            # Normalized basin forces
            'cold_force': float(cold_force),
            'far_force': float(far_force),
            'city_force': float(city_force),
            
            # Geometry signals
            'resonance': float(resonance),
            'curvature': float(curvature),
            'max_force': float(max_force),
            'force_ratio': float(force_ratio),  # |cold - far| / max_force
            'cold_density': float(cold_density),
            'distance': float(distance),
            
            # Scores
            'e_score': float(e_score),
            'c_score': float(c_score),
            'n_score': float(n_score),
            
            # Stability signals
            'is_stable': bool(is_stable),
            'is_moksha': bool(is_moksha),
            'route': str(route),
        }

