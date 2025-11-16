"""
Meaning Graph: Symbol-to-Meaning Mapping

Represents semantic relationships between symbols and meanings.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class SemanticNode:
    """
    Node in meaning graph.
    
    Represents a concept or meaning.
    """
    concept: str
    embedding: np.ndarray
    symbols: Set[str] = field(default_factory=set)
    associations: List[str] = field(default_factory=list)
    strength: float = 1.0
    
    def __repr__(self) -> str:
        return f"SemanticNode({self.concept}, symbols={len(self.symbols)})"


class MeaningGraph:
    """
    Graph representing semantic relationships.
    
    Maps symbols to meanings and tracks relationships.
    """
    
    def __init__(self):
        """Initialize meaning graph."""
        self.nodes: Dict[str, SemanticNode] = {}
        self.symbol_to_concepts: Dict[str, Set[str]] = defaultdict(set)
        self.edges: Dict[Tuple[str, str], float] = {}  # (concept1, concept2) -> strength
    
    def add_concept(self, concept: str, embedding: np.ndarray, symbols: List[str] = None):
        """
        Add concept to graph.
        
        Args:
            concept: Concept name
            embedding: Semantic embedding vector
            symbols: Associated symbols
        """
        if concept not in self.nodes:
            self.nodes[concept] = SemanticNode(
                concept=concept,
                embedding=embedding,
                symbols=set(symbols or [])
            )
            
            # Update symbol mappings
            for symbol in (symbols or []):
                self.symbol_to_concepts[symbol].add(concept)
        else:
            # Update existing
            node = self.nodes[concept]
            node.embedding = embedding
            if symbols:
                node.symbols.update(symbols)
                for symbol in symbols:
                    self.symbol_to_concepts[symbol].add(concept)
    
    def link_concepts(self, concept1: str, concept2: str, strength: float = 1.0):
        """
        Link two concepts.
        
        Args:
            concept1: First concept
            concept2: Second concept
            strength: Link strength [0, 1]
        """
        if concept1 in self.nodes and concept2 in self.nodes:
            self.edges[(concept1, concept2)] = strength
            self.edges[(concept2, concept1)] = strength  # Bidirectional
            
            self.nodes[concept1].associations.append(concept2)
            self.nodes[concept2].associations.append(concept1)
    
    def get_concepts_for_symbol(self, symbol: str) -> List[str]:
        """Get concepts associated with symbol."""
        return list(self.symbol_to_concepts.get(symbol, set()))
    
    def get_symbols_for_concept(self, concept: str) -> List[str]:
        """Get symbols associated with concept."""
        if concept in self.nodes:
            return list(self.nodes[concept].symbols)
        return []
    
    def find_similar_concepts(self, concept: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Find concepts similar to given concept.
        
        Args:
            concept: Concept name
            threshold: Similarity threshold
            
        Returns:
            List of (concept, similarity) tuples
        """
        if concept not in self.nodes:
            return []
        
        target_embedding = self.nodes[concept].embedding
        similar = []
        
        for other_concept, node in self.nodes.items():
            if other_concept != concept:
                similarity = float(np.dot(target_embedding, node.embedding))
                if similarity >= threshold:
                    similar.append((other_concept, similarity))
        
        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics."""
        return {
            'num_concepts': len(self.nodes),
            'num_edges': len(self.edges) // 2,  # Bidirectional
            'num_symbols': len(self.symbol_to_concepts),
            'avg_associations': np.mean([len(n.associations) for n in self.nodes.values()]) if self.nodes else 0.0,
        }

