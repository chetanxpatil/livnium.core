"""
Inference Engine: Logical Inference and Reasoning

Implements entailment, contradiction detection, and causal reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable

from .semantic_processor import SemanticProcessor
from .meaning_graph import MeaningGraph
from ..classical.livnium_core_system import LivniumCoreSystem


class InferenceEngine:
    """
    Inference engine for semantic reasoning.
    
    Performs logical inference, contradiction detection, and causal reasoning.
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize inference engine.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.semantic_processor = SemanticProcessor(core_system)
        self.inference_history: List[Dict] = []
    
    def infer_entailment(self, premise_coords: Tuple[int, int, int],
                        conclusion_coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Infer if premise entails conclusion.
        
        Args:
            premise_coords: Premise cell coordinates
            conclusion_coords: Conclusion cell coordinates
            
        Returns:
            Inference result
        """
        entailment_score = self.semantic_processor.detect_entailment(
            premise_coords, conclusion_coords
        )
        
        result = {
            'premise': premise_coords,
            'conclusion': conclusion_coords,
            'entailment_score': entailment_score,
            'entails': entailment_score > 0.5,
        }
        
        self.inference_history.append(result)
        return result
    
    def detect_contradictions(self) -> List[Dict[str, Any]]:
        """
        Detect all contradictions in lattice.
        
        Returns:
            List of contradiction detections
        """
        contradictions = []
        
        coords_list = list(self.core_system.lattice.keys())
        for i, coords1 in enumerate(coords_list):
            for coords2 in coords_list[i+1:]:
                score = self.semantic_processor.detect_contradiction(coords1, coords2)
                if score > 0.3:
                    contradictions.append({
                        'cell1': coords1,
                        'cell2': coords2,
                        'contradiction_score': score,
                    })
        
        return contradictions
    
    def detect_causal_links(self, 
                           cause_coords: Tuple[int, int, int],
                           effect_coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Detect causal link between cells.
        
        Args:
            cause_coords: Cause cell coordinates
            effect_coords: Effect cell coordinates
            
        Returns:
            Causal link analysis
        """
        cause_cell = self.core_system.get_cell(cause_coords)
        effect_cell = self.core_system.get_cell(effect_coords)
        
        if not cause_cell or not effect_cell:
            return {'causal_strength': 0.0}
        
        # Causal strength based on SW difference and proximity
        sw_diff = cause_cell.symbolic_weight - effect_cell.symbolic_weight
        distance = np.linalg.norm(np.array(cause_coords) - np.array(effect_coords))
        
        # Higher SW causes lower SW nearby
        if sw_diff > 0 and distance < 2:
            causal_strength = (sw_diff / 27.0) * (1.0 / (1.0 + distance))
        else:
            causal_strength = 0.0
        
        return {
            'cause': cause_coords,
            'effect': effect_coords,
            'causal_strength': float(np.clip(causal_strength, 0.0, 1.0)),
            'sw_difference': sw_diff,
            'distance': float(distance),
        }
    
    def propagate_negation(self, coords: Tuple[int, int, int]) -> Dict[Tuple[int, int, int], float]:
        """
        Propagate negation from cell.
        
        Args:
            coords: Cell coordinates
            
        Returns:
            Dictionary mapping coordinates to negation scores
        """
        cell = self.core_system.get_cell(coords)
        if not cell:
            return {}
        
        negation_scores = {}
        
        # Negation propagates to opposite class cells
        for other_coords, other_cell in self.core_system.lattice.items():
            if other_coords != coords and other_cell.cell_class:
                # Opposite classes: Core ↔ Corner, Center ↔ Edge
                if (cell.cell_class.value + other_cell.cell_class.value == 3) or \
                   (cell.cell_class.value + other_cell.cell_class.value == 3 and 
                    abs(cell.cell_class.value - other_cell.cell_class.value) == 2):
                    distance = np.linalg.norm(np.array(coords) - np.array(other_coords))
                    negation_score = 1.0 / (1.0 + distance)
                    negation_scores[other_coords] = float(negation_score)
        
        return negation_scores
    
    def get_inference_statistics(self) -> Dict:
        """Get inference engine statistics."""
        return {
            'total_inferences': len(self.inference_history),
            'entailments': sum(1 for h in self.inference_history if h.get('entails', False)),
            'contradictions': len(self.detect_contradictions()),
        }

