"""
Semantic Processor: Meaning Extraction and Processing

Transforms geometry into concepts and meaning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .feature_extractor import FeatureExtractor
from .meaning_graph import MeaningGraph
from ..classical.livnium_core_system import LivniumCoreSystem


class SemanticProcessor:
    """
    Semantic processor for Livnium Core System.
    
    Extracts meaning from geometric and quantum states.
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize semantic processor.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.feature_extractor = FeatureExtractor(core_system)
        self.meaning_graph = MeaningGraph()
        self.semantic_embeddings: Dict[Tuple[int, int, int], np.ndarray] = {}
    
    def extract_semantic_features(self, coords: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Extract semantic features from cell.
        
        Args:
            coords: Cell coordinates
            
        Returns:
            Feature dictionary
        """
        cell = self.core_system.get_cell(coords)
        if not cell:
            return {}
        
        features = {
            'coordinates': coords,
            'face_exposure': cell.face_exposure,
            'symbolic_weight': cell.symbolic_weight,
            'cell_class': cell.cell_class.name if cell.cell_class else None,
            'symbol': self.core_system.get_symbol(coords),
        }
        
        # Add geometric features
        geometric_features = self.feature_extractor.extract_geometric_features(coords)
        features.update(geometric_features)
        
        return features
    
    def create_semantic_embedding(self, coords: Tuple[int, int, int]) -> np.ndarray:
        """
        Create semantic embedding vector for cell.
        
        Args:
            coords: Cell coordinates
            
        Returns:
            Embedding vector
        """
        features = self.extract_semantic_features(coords)
        
        # Create embedding from features
        embedding = np.array([
            features.get('face_exposure', 0) / 3.0,
            features.get('symbolic_weight', 0) / 27.0,
            features.get('cell_class', 0) / 3.0 if features.get('cell_class') else 0,
            # Add more features as needed
        ], dtype=float)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self.semantic_embeddings[coords] = embedding
        return embedding
    
    def detect_contradiction(self, coords1: Tuple[int, int, int],
                           coords2: Tuple[int, int, int]) -> float:
        """
        Detect contradiction between two cells.
        
        Args:
            coords1: First cell coordinates
            coords2: Second cell coordinates
            
        Returns:
            Contradiction score [0, 1]
        """
        cell1 = self.core_system.get_cell(coords1)
        cell2 = self.core_system.get_cell(coords2)
        
        if not cell1 or not cell2:
            return 0.0
        
        # Check for contradictions
        contradiction = 0.0
        
        # Same coordinates but different classes = contradiction
        if coords1 == coords2 and cell1.cell_class != cell2.cell_class:
            contradiction += 0.5
        
        # Opposite classes nearby = potential contradiction
        if cell1.cell_class.value + cell2.cell_class.value == 3:  # Core + Corner
            distance = np.linalg.norm(np.array(coords1) - np.array(coords2))
            if distance < 2:
                contradiction += 0.3
        
        return min(1.0, contradiction)
    
    def detect_entailment(self, coords1: Tuple[int, int, int],
                         coords2: Tuple[int, int, int]) -> float:
        """
        Detect entailment (if coords1 implies coords2).
        
        Args:
            coords1: First cell (premise)
            coords2: Second cell (conclusion)
            
        Returns:
            Entailment score [0, 1]
        """
        cell1 = self.core_system.get_cell(coords1)
        cell2 = self.core_system.get_cell(coords2)
        
        if not cell1 or not cell2:
            return 0.0
        
        # Higher SW cell entails lower SW cell
        if cell1.symbolic_weight > cell2.symbolic_weight:
            entailment = (cell1.symbolic_weight - cell2.symbolic_weight) / 27.0
            return float(np.clip(entailment, 0.0, 1.0))
        
        return 0.0
    
    def propagate_context(self, context: Dict[str, Any]) -> Dict[Tuple[int, int, int], float]:
        """
        Propagate semantic context across lattice.
        
        Args:
            context: Context dictionary
            
        Returns:
            Dictionary mapping coordinates to relevance scores
        """
        relevance_scores = {}
        
        for coords in self.core_system.lattice.keys():
            features = self.extract_semantic_features(coords)
            relevance = 0.0
            
            # Check feature matches
            for key, value in context.items():
                if key in features and features[key] == value:
                    relevance += 0.3
            
            relevance_scores[coords] = min(1.0, relevance)
        
        return relevance_scores
    
    def get_semantic_summary(self) -> Dict:
        """Get semantic processing summary."""
        return {
            'total_embeddings': len(self.semantic_embeddings),
            'meaning_graph_size': len(self.meaning_graph.nodes),
            'feature_extractor': self.feature_extractor.get_statistics(),
        }

