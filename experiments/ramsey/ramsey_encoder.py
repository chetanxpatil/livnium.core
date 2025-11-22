"""
Ramsey Encoder: Map Ramsey Problems to Geometry

Maps complete graphs Kₙ to lattice coordinates and creates
tension fields for K₃ and K₄ constraints.
"""

import numpy as np
from typing import Dict, List, Tuple
from itertools import combinations

from core.classical.livnium_core_system import LivniumCoreSystem
import importlib

# Import with space in module name
encoder_module = importlib.import_module('core.encoder.constraint_encoder')
ConstraintEncoder = encoder_module.ConstraintEncoder
TensionField = encoder_module.TensionField

# Handle relative imports
try:
    from .ramsey_tension import (
        get_all_k3_subsets,
        get_all_k4_subsets,
        get_k3_edges,
        get_k4_edges,
    )
except ImportError:
    from ramsey_tension import (
        get_all_k3_subsets,
        get_all_k4_subsets,
        get_k3_edges,
        get_k4_edges,
    )

Edge = Tuple[int, int]


class RamseyEncoder:
    """
    Bridges abstract Ramsey graph (vertices, edges) to Livnium lattice.

    - Builds edge ↔ lattice-cell mapping
    - Precomputes K3 and K4 subsets
    - Provides methods to encode constraints & colorings
    """

    def __init__(self, system: LivniumCoreSystem, n_vertices: int):
        self.system = system
        self.vertices: List[int] = list(range(n_vertices))
        self.n_vertices = n_vertices

        # All edges of K_n (normalized: i < j)
        self.edges: List[Edge] = [
            (i, j) for i in range(n_vertices) for j in range(i + 1, n_vertices)
        ]

        # Universal constraint encoder
        self.constraint_encoder = ConstraintEncoder(system)

        # Map edges to coordinates (one coord per edge)
        self.edge_to_coords: Dict[Edge, Tuple[int, int, int]] = self._map_edges_to_coordinates()
        self.coords_to_edge = {coords: edge for edge, coords in self.edge_to_coords.items()}
        
        # Store active cells set
        self.active_cells = set(self.edge_to_coords.values())

        # Precompute K3 and K4 subsets
        self.k3_subsets: List[Tuple[int, int, int]] = get_all_k3_subsets(self.vertices)
        self.k4_subsets: List[Tuple[int, int, int, int]] = get_all_k4_subsets(self.vertices)

    def _map_edges_to_coordinates(self) -> Dict[Edge, Tuple[int, int, int]]:
        """
        Map edges to compact region (one cell per edge).
        
        Returns:
            Dictionary mapping edge → coordinate
        """
        n_edges = len(self.edges)
        
        # Use first n_edges cells from lattice
        coords_list = list(self.system.lattice.keys())[:n_edges]
        edge_to_coords = {}
        
        for i, edge in enumerate(self.edges):
            if i < len(coords_list):
                edge_to_coords[edge] = coords_list[i]
            else:
                raise ValueError(
                    f"Not enough cells in lattice ({len(coords_list)}) "
                    f"for {len(self.edges)} edges"
                )
        
        return edge_to_coords

    # ---------- Constraint encoding ----------

    def encode_k3_constraints(self) -> List[TensionField]:
        """
        Encode triangle constraints (for R(3,3)).

        Each triangle gets a tension field over all its edge cells.
        """
        tension_fields = []

        for tri in self.k3_subsets:
            edges = get_k3_edges(tri)
            coords = [self.edge_to_coords[e] for e in edges if e in self.edge_to_coords]

            if len(coords) < 3:
                continue  # Skip if not all edges mapped

            # Create tension function for this K₃
            def create_k3_tension_fn(coords_list):
                def compute_tension(system: LivniumCoreSystem) -> float:
                    # Decode colors using median-based approach
                    sw_values = []
                    for coord in coords_list:
                        cell = system.get_cell(coord)
                        if cell:
                            sw_values.append(cell.symbolic_weight)
                    
                    if len(sw_values) < 3:
                        return 0.0
                    
                    # Use fixed threshold (0.0) for stability
                    colors = [0 if sw < 0.0 else 1 for sw in sw_values]
                    
                    # Check if all same color (monochromatic)
                    if len(set(colors)) == 1:
                        return 1.0  # Monochromatic K₃
                    return 0.0  # Valid K₃
                
                return compute_tension

            field = self.constraint_encoder.encode_custom_constraint(
                constraint_id=f"k3_{tri}",
                involved_coords=coords,
                tension_fn=create_k3_tension_fn(coords),
                description=f"No monochromatic K₃: {tri}"
            )
            tension_fields.append(field)

        return tension_fields

    def encode_k4_constraints(self) -> List[TensionField]:
        """
        Encode K₄ constraints (for R(4,4)).

        Each K₄ gets a tension field over all its edge cells.
        """
        tension_fields = []

        for quad in self.k4_subsets:
            edges = get_k4_edges(quad)
            coords = [self.edge_to_coords[e] for e in edges if e in self.edge_to_coords]

            if len(coords) < 6:
                continue  # Skip if not all edges mapped

            # Create tension function for this K₄
            def create_k4_tension_fn(coords_list):
                def compute_tension(system: LivniumCoreSystem) -> float:
                    # Decode colors using median-based approach
                    sw_values = []
                    for coord in coords_list:
                        cell = system.get_cell(coord)
                        if cell:
                            sw_values.append(cell.symbolic_weight)
                    
                    if len(sw_values) < 6:
                        return 1.0  # High tension if incomplete
                    
                    # Use fixed threshold (0.0) for stability
                    colors = [0 if sw < 0.0 else 1 for sw in sw_values]
                    
                    # Check if all same color (monochromatic)
                    if len(set(colors)) == 1:
                        return 1.0  # Monochromatic K₄
                    return 0.0  # Valid K₄
                
                return compute_tension

            field = self.constraint_encoder.encode_custom_constraint(
                constraint_id=f"k4_{quad}",
                involved_coords=coords,
                tension_fn=create_k4_tension_fn(coords),
                description=f"No monochromatic K₄: {quad}"
            )
            tension_fields.append(field)

        return tension_fields

    # ---------- Coloring encode/decode ----------

    def decode_coloring(self) -> Dict[Edge, int]:
        """
        Read current coloring from SW field.

        Convention:
          - color 0 ↔ SW < 0 (fixed threshold, stable)
          - color 1 ↔ SW ≥ 0
        """
        coloring: Dict[Edge, int] = {}
        
        # Use fixed threshold (0.0) instead of global median for stability
        # This prevents healing one edge from flipping other edges unexpectedly
        threshold = 0.0
        
        for edge, coord in self.edge_to_coords.items():
            cell = self.system.get_cell(coord)
            if cell is None:
                continue
            # Fixed threshold: negative = color 0, non-negative = color 1
            coloring[edge] = 1 if cell.symbolic_weight >= threshold else 0
        
        return coloring

    def encode_coloring(self, coloring: Dict[Edge, int], initial_only: bool = False):
        """
        Write a full coloring into the lattice.

        For each edge:
          - color 0 → SW = -S (strong signal)
          - color 1 → SW = +S (strong signal)
        """
        S = 10.0  # base magnitude

        for edge, color in coloring.items():
            if edge not in self.edge_to_coords:
                continue
            coord = self.edge_to_coords[edge]
            target = +S if color == 1 else -S
            
            cell = self.system.get_cell(coord)
            if cell is None:
                continue
            
            if initial_only:
                # Direct assignment for initialization
                cell.symbolic_weight = target
            else:
                # Strong push toward target (90% target, 10% current)
                cell.symbolic_weight = 0.1 * cell.symbolic_weight + 0.9 * target
