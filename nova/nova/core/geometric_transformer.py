"""
Geometric Transformer: Transform Meaning Signatures

Takes query meaning + context meaning → response meaning

This is the core of the reply engine - it transforms geometric signatures
to generate new meanings, not just search for existing ones.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from nova.core.text_to_geometry import TextToGeometry


class GeometricTransformer:
    """
    Transforms geometric signatures to generate response meanings.
    
    Architecture:
    - Input: Query signature + Context signature
    - Transform: Geometric operations (merge, rotate, project)
    - Output: Response signature
    """
    
    def __init__(self, interface: TextToGeometry):
        """Initialize transformer with geometry interface."""
        self.interface = interface
        self.lattice_size = interface.geometry.lattice_size
        self.num_cells = len(interface.geometry.lattice)
    
    def merge_signatures(self, sig1: np.ndarray, sig2: np.ndarray, 
                        alpha: float = 0.6) -> np.ndarray:
        """
        Merge two signatures with weighted combination.
        
        Args:
            sig1: First signature (query)
            sig2: Second signature (context)
            alpha: Weight for sig1 (1-alpha for sig2)
        
        Returns:
            Merged signature
        """
        # Normalize signatures
        sig1_norm = sig1 / (np.linalg.norm(sig1) + 1e-10)
        sig2_norm = sig2 / (np.linalg.norm(sig2) + 1e-10)
        
        # Weighted combination
        merged = alpha * sig1_norm + (1 - alpha) * sig2_norm
        
        # Scale back to original magnitude range
        avg_magnitude = (np.linalg.norm(sig1) + np.linalg.norm(sig2)) / 2
        merged = merged * avg_magnitude
        
        return merged
    
    def transform_signature(self, query_sig: np.ndarray, 
                           context_sig: Optional[np.ndarray] = None,
                           transform_type: str = "merge") -> np.ndarray:
        """
        Transform query signature into response signature.
        
        Args:
            query_sig: Query meaning signature
            context_sig: Optional context signature
            transform_type: Type of transformation ("merge", "rotate", "project")
        
        Returns:
            Transformed response signature
        """
        if context_sig is None:
            # No context - just use query with slight modification
            response_sig = query_sig * 0.9  # Slight attenuation
        else:
            if transform_type == "merge":
                # Merge query and context
                response_sig = self.merge_signatures(query_sig, context_sig, alpha=0.7)
            elif transform_type == "rotate":
                # Rotate signature in geometry space
                # This creates a "shifted" meaning
                rotation_matrix = self._create_rotation_matrix()
                response_sig = rotation_matrix @ query_sig
            elif transform_type == "project":
                # Project query onto context direction
                context_dir = context_sig / (np.linalg.norm(context_sig) + 1e-10)
                projection = np.dot(query_sig, context_dir) * context_dir
                response_sig = 0.5 * query_sig + 0.5 * projection
            else:
                response_sig = self.merge_signatures(query_sig, context_sig)
        
        # Ensure signature is in valid range (non-negative, reasonable magnitude)
        response_sig = np.clip(response_sig, 0, None)  # No negative SW
        
        return response_sig
    
    def _create_rotation_matrix(self) -> np.ndarray:
        """Create a rotation matrix for signature transformation."""
        # Small random rotation in signature space
        angle = np.pi / 12  # 15 degrees
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 2D rotation matrix (for first 2 dimensions)
        # Extended to full signature space
        n = self.num_cells
        rotation = np.eye(n)
        rotation[0, 0] = cos_a
        rotation[0, 1] = -sin_a
        rotation[1, 0] = sin_a
        rotation[1, 1] = cos_a
        
        return rotation
    
    def apply_geometric_operations(self, signature: np.ndarray,
                                  operations: List[str]) -> np.ndarray:
        """
        Apply a sequence of geometric operations to a signature.
        
        Operations:
        - "smooth": Smooth the signature (reduce variance)
        - "amplify": Amplify peaks
        - "normalize": Normalize magnitude
        - "shift": Shift all values
        """
        result = signature.copy()
        
        for op in operations:
            if op == "smooth":
                # Moving average smoothing
                window = 3
                smoothed = np.convolve(result, np.ones(window)/window, mode='same')
                result = 0.7 * result + 0.3 * smoothed
            elif op == "amplify":
                # Amplify peaks (non-linear scaling)
                result = result * (1 + 0.2 * (result / (np.max(result) + 1e-10)))
            elif op == "normalize":
                # Normalize to unit vector
                norm = np.linalg.norm(result)
                if norm > 0:
                    result = result / norm * np.mean(signature)
            elif op == "shift":
                # Shift all values up
                result = result + np.mean(result) * 0.1
        
        return result

