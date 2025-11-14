"""
Quantum Islands Architecture

Implements "quantum islands" pattern: many small entangled groups (1-4 qubits)
instead of one global entangled state. Perfect for Livnium's informational quantum.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from quantum.kernel import LivniumQubit, EntangledPair


class QuantumIsland:
    """
    A small quantum island: 1-4 qubits with local entanglement.
    
    Represents a feature group or semantic concept cluster.
    """
    
    def __init__(self, name: str, qubits: List[LivniumQubit]):
        """
        Initialize a quantum island.
        
        Args:
            name: Island name (e.g., "semantic_features", "structural_features")
            qubits: List of qubits in this island (1-4 qubits)
        """
        if len(qubits) > 4:
            raise ValueError(f"Island {name} has {len(qubits)} qubits. Max 4 for practical use.")
        
        self.name = name
        self.qubits = qubits
        self.entangled_pairs: List[EntangledPair] = []
        
        # Track which qubits are entangled
        self.entanglement_graph: Dict[int, Set[int]] = {i: set() for i in range(len(qubits))}
    
    def entangle_pair(self, idx1: int, idx2: int):
        """
        Entangle two qubits within this island.
        
        Args:
            idx1: Index of first qubit
            idx2: Index of second qubit
        """
        if idx1 >= len(self.qubits) or idx2 >= len(self.qubits):
            raise ValueError(f"Invalid qubit indices: {idx1}, {idx2}")
        
        q1 = self.qubits[idx1]
        q2 = self.qubits[idx2]
        
        # Check if already entangled
        if q1.entangled or q2.entangled:
            return False
        
        # Create entangled pair
        pair = EntangledPair.create_from_qubits(q1, q2)
        self.entangled_pairs.append(pair)
        
        # Update graph
        self.entanglement_graph[idx1].add(idx2)
        self.entanglement_graph[idx2].add(idx1)
        
        return True
    
    def get_state_info(self) -> Dict:
        """Get information about island state."""
        info = {
            'name': self.name,
            'n_qubits': len(self.qubits),
            'n_entangled_pairs': len(self.entangled_pairs),
            'qubit_states': [q.state_string() for q in self.qubits],
        }
        
        if self.entangled_pairs:
            info['entangled_pairs'] = [
                {
                    'pair_idx': i,
                    'state': pair.state_string(),
                    'probabilities': pair.get_probabilities()
                }
                for i, pair in enumerate(self.entangled_pairs)
            ]
        
        return info


class QuantumIslandArchitecture:
    """
    Architecture for managing many quantum islands.
    
    Each island is a small entangled group (1-4 qubits).
    Islands are independent - no global entanglement.
    """
    
    def __init__(self):
        """Initialize quantum island architecture."""
        self.islands: Dict[str, QuantumIsland] = {}
        self.island_graph: Dict[str, List[str]] = {}  # Which islands interact
    
    def create_island(
        self,
        name: str,
        feature_dict: Dict[str, float],
        entanglement_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> QuantumIsland:
        """
        Create a quantum island from features.
        
        Args:
            name: Island name
            feature_dict: Dictionary mapping feature names to values
            entanglement_pairs: Optional list of (feature1, feature2) pairs to entangle
            
        Returns:
            QuantumIsland instance
        """
        # Create qubits for each feature
        qubits = []
        feature_to_idx = {}
        
        for i, (feat_name, value) in enumerate(feature_dict.items()):
            # Normalize value to [0, 1]
            normalized = np.clip(value, 0.0, 1.0)
            alpha = np.sqrt(1.0 - normalized)
            beta = np.sqrt(normalized)
            initial_state = np.array([alpha + 0j, beta + 0j], dtype=np.complex128)
            
            # Create qubit
            position = (i % 3, (i // 3) % 3, i // 9)
            qubit = LivniumQubit(position=position, f=1, initial_state=initial_state)
            qubits.append(qubit)
            feature_to_idx[feat_name] = i
        
        # Create island
        island = QuantumIsland(name, qubits)
        
        # Entangle pairs if specified
        if entanglement_pairs:
            for feat1, feat2 in entanglement_pairs:
                if feat1 in feature_to_idx and feat2 in feature_to_idx:
                    idx1 = feature_to_idx[feat1]
                    idx2 = feature_to_idx[feat2]
                    island.entangle_pair(idx1, idx2)
        
        self.islands[name] = island
        return island
    
    def connect_islands(self, island1_name: str, island2_name: str):
        """
        Connect two islands (for future: cross-island interactions).
        
        Args:
            island1_name: First island name
            island2_name: Second island name
        """
        if island1_name not in self.islands or island2_name not in self.islands:
            raise ValueError(f"Islands {island1_name} or {island2_name} not found")
        
        if island1_name not in self.island_graph:
            self.island_graph[island1_name] = []
        if island2_name not in self.island_graph:
            self.island_graph[island2_name] = []
        
        self.island_graph[island1_name].append(island2_name)
        self.island_graph[island2_name].append(island1_name)
    
    def measure_all_islands(self) -> Dict[str, Dict[str, int]]:
        """
        Measure all islands.
        
        Returns:
            Dictionary mapping island names to feature measurement results
        """
        results = {}
        
        for island_name, island in self.islands.items():
            island_results = {}
            
            # Measure entangled pairs first
            measured_indices = set()
            for pair in island.entangled_pairs:
                r1, r2 = pair.measure()
                # Find indices of qubits in pair
                idx1 = island.qubits.index(pair.q1)
                idx2 = island.qubits.index(pair.q2)
                # Map back to feature names (would need feature_to_idx)
                measured_indices.add(idx1)
                measured_indices.add(idx2)
            
            # Measure independent qubits
            for i, qubit in enumerate(island.qubits):
                if i not in measured_indices:
                    result = qubit.measure()
                    # Map to feature name (would need feature_to_idx)
                    pass
            
            results[island_name] = island_results
        
        return results
    
    def get_architecture_info(self) -> Dict:
        """Get information about the entire architecture."""
        return {
            'n_islands': len(self.islands),
            'islands': {
                name: island.get_state_info()
                for name, island in self.islands.items()
            },
            'island_graph': self.island_graph,
            'total_qubits': sum(len(island.qubits) for island in self.islands.values()),
            'total_entangled_pairs': sum(len(island.entangled_pairs) for island in self.islands.values()),
        }


def create_semantic_island(features: Dict[str, float]) -> QuantumIsland:
    """
    Create a semantic feature island.
    
    Typical features: phi_adjusted, embedding_proximity, semantic_similarity
    """
    architecture = QuantumIslandArchitecture()
    
    # Entangle semantic features
    entanglement_pairs = [
        ('phi_adjusted', 'embedding_proximity'),
    ]
    
    return architecture.create_island(
        'semantic_features',
        features,
        entanglement_pairs=entanglement_pairs
    )


def create_structural_island(features: Dict[str, float]) -> QuantumIsland:
    """
    Create a structural feature island.
    
    Typical features: sw_distribution, concentration, token_overlap
    """
    architecture = QuantumIslandArchitecture()
    
    # Entangle structural features
    entanglement_pairs = [
        ('sw_f1_ratio', 'concentration_f1'),
    ]
    
    return architecture.create_island(
        'structural_features',
        features,
        entanglement_pairs=entanglement_pairs
    )


if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM ISLANDS ARCHITECTURE DEMO")
    print("=" * 70)
    
    # Create semantic island
    semantic_features = {
        'phi_adjusted': 0.5,
        'embedding_proximity': 0.7,
        'semantic_similarity': 0.6,
    }
    
    semantic_island = create_semantic_island(semantic_features)
    print("\n✅ Created Semantic Island:")
    print(f"   Qubits: {len(semantic_island.qubits)}")
    print(f"   Entangled pairs: {len(semantic_island.entangled_pairs)}")
    
    # Create structural island
    structural_features = {
        'sw_f1_ratio': 0.6,
        'concentration_f1': 0.8,
        'token_overlap': 0.5,
    }
    
    structural_island = create_structural_island(structural_features)
    print("\n✅ Created Structural Island:")
    print(f"   Qubits: {len(structural_island.qubits)}")
    print(f"   Entangled pairs: {len(structural_island.entangled_pairs)}")
    
    # Create architecture
    architecture = QuantumIslandArchitecture()
    architecture.islands['semantic'] = semantic_island
    architecture.islands['structural'] = structural_island
    architecture.connect_islands('semantic', 'structural')
    
    print("\n✅ Quantum Island Architecture:")
    info = architecture.get_architecture_info()
    print(f"   Total islands: {info['n_islands']}")
    print(f"   Total qubits: {info['total_qubits']}")
    print(f"   Total entangled pairs: {info['total_entangled_pairs']}")
    print(f"   Island connections: {info['island_graph']}")
    
    print("\n" + "=" * 70)
    print("✅ QUANTUM ISLANDS: Many small quantum systems, not one monster!")
    print("=" * 70)

