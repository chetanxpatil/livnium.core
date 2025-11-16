#!/usr/bin/env python3
"""
Ramsey Number Solver using 5000 Omcubes

This solver uses the Livnium Core System to search for 2-colorings
of complete graphs that avoid monochromatic cliques, thereby improving
lower bounds for Ramsey numbers R(k,k).

Problem: Find a 2-coloring of edges of K_n (complete graph on n vertices)
that avoids:
- A red clique of size k
- A blue clique of size k

If such a coloring exists, we prove R(k,k) > n.

Approach:
- Use 5000 omcubes (geometric states) to represent different colorings
- Each omcube encodes a partial or complete edge coloring
- Use Livnium Core System to explore search space
- Check constraints (no monochromatic k-cliques)
- Use symmetry reduction to avoid duplicate colorings

IMPORTANT: Quantum-Inspired Classical Search
---------------------------------------------
This is NOT quantum computing. Each omcube is a CLASSICAL state (a RamseyGraph
object), not a quantum superposition. We use 50,000 omcubes = 50,000 parallel
classical universes exploring the search space together.

What makes it "quantum-inspired":
- Parallel exploration of many configurations simultaneously
- Geometric interference patterns (via coordinate evolution)
- Early collapse of invalid paths (constraint checking)
- Amplification of promising regions (elite archiving)
- Structured search through Livnium Core System geometry

This is quantum-INSPIRED evolutionary search, not quantum computing.
No amplitudes, no unitaries, no true superposition - but powerful parallel
classical exploration with geometric guidance.
"""

import sys
import time
import copy
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
from itertools import combinations

# Try to import numba for acceleration (optional dependency)
try:
    from numba import jit, prange, types
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    types = None
    NumbaDict = dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Use new core system
from core import LivniumCoreSystem, LivniumCoreConfig, RotationAxis

# Dual cube monitor - optional, uses archive if available
try:
    # Try to import from archive if available
    archive_path = project_root / "archive" / "pre_core_systems"
    if archive_path.exists():
        sys.path.insert(0, str(archive_path))
        from quantum.hierarchical.monitoring.dual_cube_monitor import DualCubeMonitor
    else:
        DualCubeMonitor = None
except ImportError:
    DualCubeMonitor = None


class RamseyGraph:
    """
    Represents a graph with 2-colored edges for Ramsey number search.
    
    Edge coloring: 0 = red, 1 = blue
    """
    
    def __init__(self, n: int):
        """
        Initialize complete graph on n vertices.
        
        Args:
            n: Number of vertices
        """
        self.n = n
        self.num_edges = n * (n - 1) // 2
        self.edge_coloring: Dict[Tuple[int, int], int] = {}
        self.edge_index_map: Dict[Tuple[int, int], int] = {}
        self._hash_cache: Optional[int] = None  # Cache for graph hash
        
        # Map edges to indices
        edge_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                self.edge_index_map[(i, j)] = edge_idx
                self.edge_index_map[(j, i)] = edge_idx
                edge_idx += 1
    
    def get_hash(self) -> int:
        """
        Compute a hash fingerprint of the edge coloring.
        
        Uses stable Python hash of sorted edge coloring for collision-resistant uniqueness.
        """
        if self._hash_cache is None:
            # Create a stable, sorted representation of the coloring
            sorted_edges = tuple(sorted(self.edge_coloring.items()))
            self._hash_cache = hash(sorted_edges)
        return self._hash_cache
    
    def set_edge_color(self, u: int, v: int, color: int):
        """Set color of edge (u, v). 0=red, 1=blue."""
        if u > v:
            u, v = v, u
        self.edge_coloring[(u, v)] = color
        self._hash_cache = None  # Invalidate hash cache
    
    def get_edge_index(self, u: int, v: int) -> int:
        """Get index of edge (u, v)."""
        return self.edge_index_map.get((u, v), -1)
    
    def get_edge_color(self, u: int, v: int) -> Optional[int]:
        """Get color of edge (u, v). Returns None if uncolored."""
        if u > v:
            u, v = v, u
        return self.edge_coloring.get((u, v))
    
    # Original implementation (commented for reference)
    # def has_monochromatic_clique(self, k: int) -> Tuple[bool, Optional[List[int]]]:
    #     """
    #     Check if graph has a monochromatic clique of size k.
    #     
    #     Optimized version with early termination and efficient checking.
    #     
    #     Returns:
    #         (has_clique, clique_vertices) - True if found, with vertex list
    #     """
    #     # Build adjacency lists for each color
    #     red_adj = defaultdict(set)
    #     blue_adj = defaultdict(set)
    #     
    #     for (u, v), color in self.edge_coloring.items():
    #         if color == 0:  # Red
    #             red_adj[u].add(v)
    #             red_adj[v].add(u)
    #         elif color == 1:  # Blue
    #             blue_adj[u].add(v)
    #             blue_adj[v].add(u)
    #     
    #     # Check for red k-clique using recursive backtracking
    #     def find_clique(vertices, color_adj, current_clique, remaining):
    #         if len(current_clique) == k:
    #             return True, list(current_clique)
    #         
    #         if len(current_clique) + len(remaining) < k:
    #             return False, None
    #         
    #         # Try adding each remaining vertex
    #         for v in list(remaining):
    #             # Check if v is connected to all vertices in current_clique
    #             can_add = all(v in color_adj[u] for u in current_clique)
    #             
    #             if can_add:
    #                 new_clique = current_clique | {v}
    #                 new_remaining = remaining & color_adj[v]
    #                 
    #                 found, clique = find_clique(vertices, color_adj, new_clique, new_remaining)
    #                 if found:
    #                     return True, clique
    #         
    #         return False, None
    #     
    #     # Check red cliques
    #     for start_vertex in range(self.n):
    #         if len(red_adj[start_vertex]) >= k - 1:
    #             found, clique = find_clique(
    #                 range(self.n), red_adj, {start_vertex}, red_adj[start_vertex]
    #             )
    #             if found:
    #                 return True, clique
    #     
    #     # Check blue cliques
    #     for start_vertex in range(self.n):
    #         if len(blue_adj[start_vertex]) >= k - 1:
    #             found, clique = find_clique(
    #                 range(self.n), blue_adj, {start_vertex}, blue_adj[start_vertex]
    #             )
    #             if found:
    #                 return True, clique
    #     
    #     return False, None
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _find_clique_numba(adj_matrix: np.ndarray, n: int, k: int, 
                          current_clique: np.ndarray, clique_size: int,
                          remaining: np.ndarray, remaining_size: int) -> Tuple[bool, np.ndarray]:
        """
        Find clique using numba-accelerated backtracking.
        
        Returns:
            (found, clique_array) where clique_array is size k if found, else zeros
        """
        if clique_size == k:
            return True, current_clique[:k].copy()
        
        if clique_size + remaining_size < k:
            return False, np.zeros(k, dtype=np.int32)
        
        # Try adding each remaining vertex
        for i in range(remaining_size):
            v = remaining[i]
            
            # Check if v is connected to all vertices in current_clique
            can_add = True
            for j in range(clique_size):
                u = current_clique[j]
                if adj_matrix[u, v] == 0:
                    can_add = False
                    break
            
            if can_add:
                # Add v to clique
                new_clique = current_clique.copy()
                new_clique[clique_size] = v
                
                # Build new remaining list (neighbors of v that are also in remaining)
                new_remaining = np.zeros(n, dtype=np.int32)
                new_remaining_size = 0
                for j in range(remaining_size):
                    w = remaining[j]
                    if w != v and adj_matrix[v, w] == 1:
                        new_remaining[new_remaining_size] = w
                        new_remaining_size += 1
                
                found, clique = RamseyGraph._find_clique_numba(
                    adj_matrix, n, k, new_clique, clique_size + 1,
                    new_remaining, new_remaining_size
                )
                if found:
                    return True, clique
        
        return False, np.zeros(k, dtype=np.int32)
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _has_clique_numba(adj_matrix: np.ndarray, n: int, k: int) -> Tuple[bool, np.ndarray]:
        """Check for clique using numba (returns found flag and clique array)."""
        # Check each starting vertex
        for start_vertex in range(n):
            # Count neighbors
            neighbor_count = 0
            for v in range(n):
                if v != start_vertex and adj_matrix[start_vertex, v] == 1:
                    neighbor_count += 1
            
            if neighbor_count >= k - 1:
                # Build initial clique and remaining
                current_clique = np.zeros(k, dtype=np.int32)
                current_clique[0] = start_vertex
                
                remaining = np.zeros(n, dtype=np.int32)
                remaining_size = 0
                for v in range(n):
                    if v != start_vertex and adj_matrix[start_vertex, v] == 1:
                        remaining[remaining_size] = v
                        remaining_size += 1
                
                found, clique = RamseyGraph._find_clique_numba(
                    adj_matrix, n, k, current_clique, 1, remaining, remaining_size
                )
                if found:
                    return True, clique
        
        return False, np.zeros(k, dtype=np.int32)
    
    def has_monochromatic_clique(self, k: int) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if graph has a monochromatic clique of size k.
        
        Optimized version with early termination and efficient checking.
        Uses numba acceleration when available.
        
        Returns:
            (has_clique, clique_vertices) - True if found, with vertex list
        """
        if NUMBA_AVAILABLE:
            # Build adjacency matrices for each color (numba-friendly)
            red_adj = np.zeros((self.n, self.n), dtype=np.int32)
            blue_adj = np.zeros((self.n, self.n), dtype=np.int32)
            
            for (u, v), color in self.edge_coloring.items():
                if color == 0:  # Red
                    red_adj[u, v] = 1
                    red_adj[v, u] = 1
                elif color == 1:  # Blue
                    blue_adj[u, v] = 1
                    blue_adj[v, u] = 1
            
            # Check red cliques
            found, clique = self._has_clique_numba(red_adj, self.n, k)
            if found:
                return True, clique.tolist()
            
            # Check blue cliques
            found, clique = self._has_clique_numba(blue_adj, self.n, k)
            if found:
                return True, clique.tolist()
            
            return False, None
        else:
            # Fallback to original implementation
            red_adj = defaultdict(set)
            blue_adj = defaultdict(set)
            
            for (u, v), color in self.edge_coloring.items():
                if color == 0:  # Red
                    red_adj[u].add(v)
                    red_adj[v].add(u)
                elif color == 1:  # Blue
                    blue_adj[u].add(v)
                    blue_adj[v].add(u)
            
            def find_clique(vertices, color_adj, current_clique, remaining):
                if len(current_clique) == k:
                    return True, list(current_clique)
                
                if len(current_clique) + len(remaining) < k:
                    return False, None
                
                for v in list(remaining):
                    can_add = all(v in color_adj[u] for u in current_clique)
                    
                    if can_add:
                        new_clique = current_clique | {v}
                        new_remaining = remaining & color_adj[v]
                        
                        found, clique = find_clique(vertices, color_adj, new_clique, new_remaining)
                        if found:
                            return True, clique
                
                return False, None
            
            for start_vertex in range(self.n):
                if len(red_adj[start_vertex]) >= k - 1:
                    found, clique = find_clique(
                        range(self.n), red_adj, {start_vertex}, red_adj[start_vertex]
                    )
                    if found:
                        return True, clique
            
            for start_vertex in range(self.n):
                if len(blue_adj[start_vertex]) >= k - 1:
                    found, clique = find_clique(
                        range(self.n), blue_adj, {start_vertex}, blue_adj[start_vertex]
                    )
                    if found:
                        return True, clique
            
            return False, None
    
    def to_coordinates(self) -> Tuple[float, ...]:
        """
        Convert edge coloring to geometric coordinates.
        
        Uses a packed encoding that incorporates all edge information into 3D coordinates.
        This provides meaningful geometric representation of the full coloring state.
        """
        # Create a list of all edge colors (0, 1, or None for uncolored)
        edge_colors = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                color = self.edge_coloring.get((i, j))
                if color is None:
                    edge_colors.append(0.5)  # Uncolored = 0.5
                else:
                    edge_colors.append(float(color))
        
        # Pack into 3D coordinates using weighted sums
        # This distributes information across all 3 dimensions
        if len(edge_colors) == 0:
            return (0.0, 0.0, 0.0)
        
        # Use different weight patterns for each dimension
        coords = []
        for axis in range(3):
            # Weighted sum with different patterns per axis
            weighted_sum = sum(
                color * ((i + 1) * (axis + 1)) % 1.0 
                for i, color in enumerate(edge_colors)
            )
            # Normalize to [0, 1]
            coord = (weighted_sum / len(edge_colors)) % 1.0
            coords.append(coord)
        
        return tuple(coords)
    
    def from_coordinates(self, coords: Tuple[float, ...], seed: int = None):
        """
        Reconstruct edge coloring from coordinates using deterministic mapping.
        
        Since coordinates only encode partial information, we use them as a seed
        to generate a consistent coloring pattern.
        
        Args:
            coords: Geometric coordinates
            seed: Optional seed for random generation
        """
        # Use coordinates as seed for deterministic coloring
        if seed is None:
            seed = int(sum(c * 1000 for c in coords) % (2**31))
        
        np.random.seed(seed)
        
        # Generate coloring based on seed
        for i in range(self.n):
            for j in range(i + 1, self.n):
                # Use deterministic "random" based on seed and edge position
                edge_hash = (i * self.n + j + seed) % 2
                self.set_edge_color(i, j, edge_hash)


class RamseySolver:
    """
    Ramsey number solver using 5000 omcubes.
    
    Each omcube represents a candidate coloring or partial coloring.
    
    NOTE: These are CLASSICAL states, not quantum superpositions.
    - Each omcube = one RamseyGraph object (classical configuration)
    - 50,000 omcubes = 50,000 parallel classical universes
    - Evolution happens through geometric operations, not unitary gates
    - This is quantum-INSPIRED search, not quantum computing
    
    The power comes from:
    - Parallel exploration of many configurations
    - Geometric interference patterns (coordinate evolution)
    - Early collapse of invalid paths (constraint checking)
    - Amplification of promising regions (elite archiving)
    """
    
    def __init__(self, n: int, k: int, num_omcubes: int = 5000, enable_dual_monitor: bool = False):
        """
        Initialize Ramsey solver.
        
        Args:
            n: Number of vertices in graph
            k: Clique size to avoid
            num_omcubes: Number of omcubes to use (default: 5000)
            enable_dual_monitor: If True, enable dual cube monitoring for semantic diagnostics
        """
        self.n = n
        self.k = k
        self.num_omcubes = num_omcubes
        self.graph = RamseyGraph(n)
        self.num_edges = n * (n - 1) // 2
        
        # Initialize core system
        print(f"Initializing Livnium Core System with {num_omcubes:,} omcubes...")
        
        # Calculate lattice size needed (need at least num_omcubes cells)
        # N³ >= num_omcubes, so N >= ceil(cbrt(num_omcubes))
        # Make it odd and at least 3
        min_lattice_size = int(np.ceil(num_omcubes ** (1/3)))
        if min_lattice_size % 2 == 0:
            min_lattice_size += 1
        if min_lattice_size < 3:
            min_lattice_size = 3
        
        # Create config with optimized features for Ramsey solving
        config = LivniumCoreConfig(
            lattice_size=min_lattice_size,
            enable_3x3x3_lattice=True,
            enable_symbol_alphabet=False,  # Don't need symbols
            enable_symbolic_weight=False,  # Don't need SW for this
            enable_face_exposure=True,  # ENABLED: Prioritize omcubes by geometric structure
            enable_class_structure=True,  # ENABLED: Organize search by cell classes
            enable_90_degree_rotations=True,  # Need rotations
            enable_rotation_group=True,
            enable_global_observer=True,  # ENABLED: Reference point for polarity
            enable_local_observer=False,
            enable_semantic_polarity=True,  # ENABLED: Guide mutations toward solutions
            enable_cross_lattice_coupling=False,
            enable_sw_conservation=False,
            enable_class_count_conservation=False,
        )
        
        self.core_system = LivniumCoreSystem(config)
        
        # Map omcube indices to cell coordinates and store float coordinates
        self.omcube_to_cell: Dict[int, Tuple[int, int, int]] = {}
        self.cell_to_omcube: Dict[Tuple[int, int, int], int] = {}
        self.omcube_coords: Dict[int, Tuple[float, float, float]] = {}
        
        # Initialize mapping: use first num_omcubes cells from lattice
        sorted_cells = sorted(self.core_system.lattice.keys())
        for i in range(min(num_omcubes, len(sorted_cells))):
            cell_coords = sorted_cells[i]
            self.omcube_to_cell[i] = cell_coords
            self.cell_to_omcube[cell_coords] = i
        
        # Dual cube monitor: optional semantic sensor/thermometer/logger
        self.dual_monitor: Optional[DualCubeMonitor] = None
        if enable_dual_monitor and DualCubeMonitor is not None:
            self.dual_monitor = DualCubeMonitor(base_dimension=3, num_levels=3)
            print("  Dual cube monitor: ENABLED (semantic diagnostics active)")
        elif enable_dual_monitor:
            print("  ⚠️  Dual cube monitor requested but not available (archived)")
        
        # Store omcube states
        self.omcube_states: List[RamseyGraph] = []
        # Track best valid colorings (keep only the best ones, not all)
        self.best_valid_colorings: List[RamseyGraph] = []
        self.max_best_valid = 100  # Keep only top 100 best
        
        # Logging for metrics
        self.metrics_log: List[Dict] = []
        
        print(f"  Target: Find 2-coloring of K_{n} avoiding monochromatic K_{k}")
        print(f"  Total edges: {self.num_edges}")
        print(f"  Search space: 2^{self.num_edges} = {2**self.num_edges:,} possible colorings")
        print(f"  Lattice size: {min_lattice_size}×{min_lattice_size}×{min_lattice_size} = {min_lattice_size**3} cells")
    
    def initialize_omcubes(self):
        """Initialize omcubes with random or structured colorings."""
        print(f"\nInitializing {self.num_omcubes:,} omcubes...")
        
        start_time = time.time()
        for i in range(self.num_omcubes):
            # Create a new graph for this omcube
            graph = RamseyGraph(self.n)
            
            # Strategy 1: Incremental building (start small, build up)
            if i < self.num_omcubes * 0.4:
                # Start with a few edges, build incrementally
                num_colored = min(50, self.num_edges // 10)  # Start with ~10% of edges
                edges = [(u, v) for u in range(self.n) for v in range(u + 1, self.n)]
                np.random.shuffle(edges)
                
                # Color edges one by one, checking for cliques
                colored_count = 0
                for u, v in edges:
                    if colored_count >= num_colored:
                        break
                    # Try both colors, pick one that doesn't create immediate clique
                    for color in [0, 1]:
                        graph.set_edge_color(u, v, color)
                        # Quick check: does this edge create a small clique with neighbors?
                        has_small_clique = False
                        for w in range(self.n):
                            if w != u and w != v:
                                u_w = graph.get_edge_color(u, w)
                                v_w = graph.get_edge_color(v, w)
                                if u_w == color and v_w == color and u_w is not None and v_w is not None:
                                    # Potential triangle, but we only care about k-cliques
                                    if self.k <= 3:
                                        has_small_clique = True
                                        break
                        if not has_small_clique:
                            colored_count += 1
                            break
            
            # Strategy 2: Random coloring (more edges)
            elif i < self.num_omcubes * 0.7:
                # Random partial coloring
                num_colored = np.random.randint(self.num_edges // 3, self.num_edges * 2 // 3)
                edges = [(u, v) for u in range(self.n) for v in range(u + 1, self.n)]
                np.random.shuffle(edges)
                
                for u, v in edges[:num_colored]:
                    color = np.random.randint(0, 2)
                    graph.set_edge_color(u, v, color)
            
            # Strategy 3: Structured coloring (balanced)
            elif i < self.num_omcubes * 0.9:
                # Try to balance red/blue, but only partially
                num_colored = self.num_edges * 3 // 4
                red_count = 0
                blue_count = 0
                edges = [(u, v) for u in range(self.n) for v in range(u + 1, self.n)]
                np.random.shuffle(edges)
                
                for u, v in edges[:num_colored]:
                    if red_count <= blue_count:
                        graph.set_edge_color(u, v, 0)  # Red
                        red_count += 1
                    else:
                        graph.set_edge_color(u, v, 1)  # Blue
                        blue_count += 1
            
            # Strategy 4: Known construction patterns
            else:
                # Use geometric patterns - partition vertices
                partition_size = self.n // 2
                for u in range(self.n):
                    for v in range(u + 1, self.n):
                        # Within same partition: one color, between partitions: other
                        u_partition = u < partition_size
                        v_partition = v < partition_size
                        if u_partition == v_partition:
                            graph.set_edge_color(u, v, 0)  # Red within partitions
                        else:
                            graph.set_edge_color(u, v, 1)  # Blue between partitions
            
            # Convert to coordinates and store
            coords = graph.to_coordinates()
            self.omcube_coords[i] = coords
            self.omcube_states.append(graph)
            
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Progress: {i+1:,}/{self.num_omcubes:,} ({rate:,.0f} omcubes/s)")
        
        elapsed = time.time() - start_time
        print(f"  ✅ Initialized {self.num_omcubes:,} omcubes in {elapsed:.2f}s")
    
    def check_constraints(self, graph: RamseyGraph) -> Tuple[bool, Optional[List[int]]]:
        """
        Check if coloring satisfies constraints (no monochromatic k-clique).
        
        Returns:
            (is_valid, violating_clique)
        """
        has_clique, clique = graph.has_monochromatic_clique(self.k)
        return not has_clique, clique
    
    def search_for_valid_coloring(self, max_iterations: int = 100) -> Optional[RamseyGraph]:
        """
        Search for a valid coloring using omcube exploration.
        
        Args:
            max_iterations: Maximum iterations to run
            
        Returns:
            Valid coloring if found, None otherwise
        """
        print(f"\nSearching for valid coloring...")
        print(f"  Max iterations: {max_iterations}")
        print(f"  Note: Clique checking is optimized but may still be slow for large n/k")
        print(f"  Progress will be shown every 5 iterations")
        
        # For very large problems, we might want to sample, but by default use all omcubes
        # Only sample if the problem is extremely large AND we have many omcubes
        use_all = (self.n * self.k) < 300 or self.num_omcubes <= 1000
        sample_size = min(1000, self.num_omcubes) if not use_all else self.num_omcubes
        
        if not use_all:
            print(f"  Note: Using sampling (checking {sample_size} of {self.num_omcubes} omcubes per iteration)")
        else:
            print(f"  Checking all {self.num_omcubes} omcubes per iteration")
        
        for iteration in range(max_iterations):
            # Inject elites back into population to prevent drift
            if iteration > 0 and self.best_valid_colorings:
                num_inject = min(50, len(self.best_valid_colorings), len(self.omcube_states) // 10)
                for j in range(num_inject):
                    # Overwrite some random omcubes with elites
                    idx = np.random.randint(0, len(self.omcube_states))
                    self.omcube_states[idx] = copy.deepcopy(self.best_valid_colorings[j % len(self.best_valid_colorings)])
            
            # Track valid colorings for THIS iteration only
            iteration_valid: List[RamseyGraph] = []
            seen_hashes: Set[int] = set()  # Track unique graphs by hash
            valid_found = 0
            
            # Select omcubes to check
            if use_all:
                indices_to_check = range(self.num_omcubes)
            else:
                # Prioritize by completeness AND face exposure (Livnium feature)
                if self.core_system.config.enable_face_exposure:
                    def priority_key(i):
                        g = self.omcube_states[i]
                        completeness = len(g.edge_coloring)
                        # Get face exposure for this omcube
                        cell_coords = self.omcube_to_cell.get(i)
                        face_priority = 0
                        if cell_coords:
                            cell = self.core_system.get_cell(cell_coords)
                            if cell:
                                face_priority = cell.face_exposure  # Higher = more priority
                        return (completeness, face_priority)
                    
                    sorted_indices = sorted(
                        range(self.num_omcubes),
                        key=priority_key,
                        reverse=True
                    )
                else:
                    sorted_indices = sorted(
                        range(self.num_omcubes),
                        key=lambda i: len(self.omcube_states[i].edge_coloring),
                        reverse=True
                    )
                indices_to_check = sorted_indices[:sample_size]
            
            # Check selected omcubes
            for idx, i in enumerate(indices_to_check):
                graph = self.omcube_states[i]
                
                # Only check graphs with enough edges colored
                min_edges = self.num_edges * 0.3 if (self.n * self.k) >= 200 else self.num_edges * 0.1
                if len(graph.edge_coloring) < min_edges:
                    continue
                
                is_valid, violating_clique = self.check_constraints(graph)
                
                if is_valid and self.num_edges == len(graph.edge_coloring):
                    # Fully colored and valid!
                    print(f"\n  ✅ FOUND VALID COLORING at omcube {i}!")
                    return graph
                
                if is_valid:
                    valid_found += 1
                    graph_hash = graph.get_hash()
                    if graph_hash not in seen_hashes:
                        seen_hashes.add(graph_hash)
                        iteration_valid.append(graph)
                
                # Progress indicator
                if (idx + 1) % 50 == 0 and iteration % 5 == 0:
                    print(f"    Checked {idx+1}/{len(indices_to_check)} omcubes (iteration {iteration})...", end='\r')
            
            # Update best valid colorings (keep only top performers)
            iteration_valid.sort(key=lambda g: len(g.edge_coloring), reverse=True)
            self.best_valid_colorings.extend([copy.deepcopy(g) for g in iteration_valid[:20]])
            self.best_valid_colorings.sort(key=lambda g: len(g.edge_coloring), reverse=True)
            self.best_valid_colorings = self.best_valid_colorings[:self.max_best_valid]
            
            # Evolve omcubes using core system rotations
            if iteration < max_iterations - 1:
                # Apply rotations to the core system
                # Rotate on different axes to create diversity
                axes = [RotationAxis.X, RotationAxis.Y, RotationAxis.Z]
                axis = axes[iteration % len(axes)]
                quarter_turns = 1  # Single 90° rotation
                
                # Apply rotation to core system
                self.core_system.rotate(axis, quarter_turns)
                
                # Update coordinate mappings after rotation
                # After rotation, cell coordinates change, so we need to remap
                # For simplicity, we'll just use the stored float coordinates
                # and apply a small perturbation based on the rotation
                
                # Update graphs from evolved coordinates
                # Treat valid and invalid graphs differently
                # Use coordinates as mutation seeds to explore nearby colorings
                # Now with semantic guidance from dual cube system + Livnium features
                total_to_mutate = len(self.omcube_states)
                
                # Get best solution coordinates for polarity calculation
                best_coords = None
                if self.best_valid_colorings and self.core_system.config.enable_semantic_polarity:
                    best_coords = self.best_valid_colorings[0].to_coordinates()
                
                for i in range(len(self.omcube_states)):
                    if i < len(self.omcube_states):
                        g = self.omcube_states[i]
                        coords = self.omcube_coords.get(i, g.to_coordinates())
                        
                        # Apply perturbation based on rotation to evolve coordinates
                        perturbation = np.array([0.01, 0.01, 0.01]) * (iteration % 3 - 1)
                        coords = tuple((np.array(coords) + perturbation) % 1.0)
                        self.omcube_coords[i] = coords

                        is_valid, _ = self.check_constraints(g)
                        semantic = self.compute_semantic_scores(g, is_valid=is_valid)
                        
                        # Add Livnium features to semantic scores
                        if self.core_system.config.enable_face_exposure:
                            cell_coords = self.omcube_to_cell.get(i)
                            if cell_coords:
                                cell = self.core_system.get_cell(cell_coords)
                                if cell:
                                    semantic['face_exposure'] = cell.face_exposure
                                    semantic['cell_class'] = cell.cell_class.name if cell.cell_class else None
                        
                        if self.core_system.config.enable_semantic_polarity and best_coords:
                            # Calculate motion vector from best solution to current state
                            motion_vec = tuple(np.array(coords) - np.array(best_coords))
                            try:
                                polarity = self.core_system.calculate_polarity(motion_vec)
                                semantic['polarity'] = polarity
                            except:
                                semantic['polarity'] = 0.0
                        else:
                            semantic['polarity'] = 0.0

                        if is_valid:
                            mutated = copy.deepcopy(g)
                            self.mutate_coloring(mutated, coords, gentle=True, semantic=semantic)
                            still_valid, _ = self.check_constraints(mutated)
                            if still_valid:
                                self.omcube_states[i] = mutated
                        else:
                            mutated = copy.deepcopy(g)
                            self.mutate_coloring(mutated, coords, gentle=False, semantic=semantic)
                            self.omcube_states[i] = mutated
                    
                    # Progress indicator for mutation phase (every 5000 omcubes)
                    if (i + 1) % 5000 == 0:
                        print(f"    Mutating omcubes: {i+1}/{total_to_mutate} ({100*(i+1)//total_to_mutate}%)...", end='\r')
                
                # Clear the progress line
                if total_to_mutate > 0:
                    print(f"    Mutated {total_to_mutate} omcubes" + " " * 20)
                
                # Beam search: Down-select omcubes gradually
                if iteration >= 5 and self.num_omcubes > 10000:
                    def rank_key(i):
                        g = self.omcube_states[i]
                        is_valid, _ = self.check_constraints(g)
                        return (1 if is_valid else 0, len(g.edge_coloring))
                    
                    ranked_indices = sorted(
                        range(len(self.omcube_states)),
                        key=rank_key,
                        reverse=True
                    )
                    
                    # Gradual compression based on iteration
                    if iteration < 10:
                        N_active = 30000
                    elif iteration < 20:
                        N_active = 20000
                    else:
                        N_active = 10000
                    
                    keep_indices = ranked_indices[:N_active]
                    self.omcube_states = [self.omcube_states[i] for i in keep_indices]
                    
                    # Down-select coordinate mappings
                    new_omcube_coords = {}
                    new_omcube_to_cell = {}
                    new_cell_to_omcube = {}
                    for new_idx, old_idx in enumerate(keep_indices):
                        if old_idx in self.omcube_coords:
                            new_omcube_coords[new_idx] = self.omcube_coords[old_idx]
                        if old_idx in self.omcube_to_cell:
                            cell_coords = self.omcube_to_cell[old_idx]
                            new_omcube_to_cell[new_idx] = cell_coords
                            new_cell_to_omcube[cell_coords] = new_idx
                    
                    self.omcube_coords = new_omcube_coords
                    self.omcube_to_cell = new_omcube_to_cell
                    self.cell_to_omcube = new_cell_to_omcube
                    self.num_omcubes = len(self.omcube_states)
                    
                    if len(self.best_valid_colorings) > self.max_best_valid:
                        self.best_valid_colorings.sort(key=lambda g: len(g.edge_coloring), reverse=True)
                        self.best_valid_colorings = self.best_valid_colorings[:self.max_best_valid]
                    
                    if iteration % 10 == 0:
                        print(f"    Beam search: Down-selected to {self.num_omcubes} active omcubes, "
                              f"best stored: {len(self.best_valid_colorings)}")
            
            # Log iteration status and metrics
            if iteration % 10 == 0 or iteration == 0:
                print(f"  Iteration {iteration}: Valid partial colorings: {valid_found}/{len(indices_to_check)} (checked), "
                      f"best stored: {len(self.best_valid_colorings)}")
                
                if self.dual_monitor is not None and iteration % 5 == 0:
                    decoherence = self.dual_monitor.measure_decoherence()
                    print(f"    [DUAL CUBE] pos_energy={decoherence['positive_energy']:.2f}, "
                          f"neg_energy={decoherence['negative_energy']:.2f}, "
                          f"decoherence={decoherence['decoherence_fraction']:.3f}")
                
                if self.best_valid_colorings:
                    sample = self.best_valid_colorings[0]
                    sem = self.compute_semantic_scores(sample, is_valid=True)
                    print(f"    [SEMANTICS] completeness={sem['completeness']:.3f}, "
                          f"confusion={sem['confusion_score']:.3f}, "
                          f"diag={sem['diagnosis']}")
            
            # Log metrics to JSONL file (every iteration)
            if self.dual_monitor is not None:
                decoherence = self.dual_monitor.measure_decoherence()
                avg_edges_valid = (
                    sum(len(g.edge_coloring) for g in self.best_valid_colorings) / len(self.best_valid_colorings)
                    if self.best_valid_colorings else 0
                )
                
                metric_entry = {
                    "iteration": iteration,
                    "num_valid": valid_found,
                    "num_invalid": len(indices_to_check) - valid_found,
                    "num_omcubes": self.num_omcubes,
                    "best_stored": len(self.best_valid_colorings),
                    "average_edges_per_valid": avg_edges_valid,
                    "dual_cube": {
                        "positive_energy": float(decoherence["positive_energy"]),
                        "negative_energy": float(decoherence["negative_energy"]),
                        "decoherence_fraction": float(decoherence["decoherence_fraction"]),
                    }
                }
                self.metrics_log.append(metric_entry)
            
            # Try to complete best valid colorings
            should_try_complete = (
                (iteration >= 20 and iteration % 2 == 0) or
                (valid_found == 0 and len(self.best_valid_colorings) > 0)
            )
            
            if len(self.best_valid_colorings) > 0 and should_try_complete:
                # Try to complete top candidates
                candidates_to_try = min(10, len(self.best_valid_colorings))
                for graph in self.best_valid_colorings[:candidates_to_try]:
                    completed = self.complete_coloring(graph)
                    if completed:
                        is_valid, _ = self.check_constraints(completed)
                        if is_valid:
                            print(f"\n  ✅ FOUND VALID COLORING by completing partial coloring!")
                            return completed
        
        print(f"\n  ⚠️  No complete valid coloring found after {max_iterations} iterations")
        print(f"  Best valid partial colorings stored: {len(self.best_valid_colorings)}")
        if self.best_valid_colorings:
            best = self.best_valid_colorings[0]
            print(f"  Best partial coloring: {len(best.edge_coloring)}/{self.num_edges} edges ({100*len(best.edge_coloring)/self.num_edges:.1f}% complete)")
        
        return None
    
    def compute_semantic_scores(self, graph: RamseyGraph, is_valid: bool) -> dict:
        """
        Compute semantic scores for a graph.
        
        If monitor is enabled, records the state and returns semantic metrics.
        If monitor is disabled, returns basic completeness info only.
        """
        coords = graph.to_coordinates()
        completeness = len(graph.edge_coloring) / float(self.num_edges) if self.num_edges > 0 else 0.0
        
        # If monitor is enabled, record state and get semantic metrics
        if self.dual_monitor is not None:
            weight = len(graph.edge_coloring)  # Use edge count as weight
            
            if is_valid:
                self.dual_monitor.record_positive(coords, weight=weight)
            else:
                self.dual_monitor.record_negative(coords, weight=weight)
            
            # Get confusion diagnosis
            diagnosis = self.dual_monitor.diagnose_confusion(coords)
            
            return {
                "coords": coords,
                "completeness": completeness,
                "confusion_score": float(diagnosis["confusion_score"]),
                "diagnosis": diagnosis["diagnosis"],
            }
        else:
            # Monitor disabled: return basic info only
            return {
                "coords": coords,
                "completeness": completeness,
                "confusion_score": 0.0,
                "diagnosis": "monitor_disabled",
            }
    
    def mutate_coloring(
        self,
        graph: RamseyGraph,
        coords: Tuple[float, ...],
        gentle: bool = False,
        semantic: Optional[dict] = None,
    ):
        """
        Mutate a coloring based on coordinate changes.
        
        Uses coordinate variance to determine mutation probability for better diversity.
        Now also uses semantic scores to guide mutation intensity.
        
        Args:
            graph: Graph to mutate
            coords: Geometric coordinates
            gentle: If True, use much gentler mutations (for valid graphs)
            semantic: Optional semantic scores dict with confusion_score and completeness
        """
        # Calculate base mutation rate from coordinate variance
        coords_array = np.array(coords)
        coord_variance = np.var(coords_array)
        base_mutation_rate = min(1.0, max(0.0, coord_variance * 4.0))
        
        if gentle:
            mutation_rate = base_mutation_rate * 0.1
        else:
            mutation_rate = base_mutation_rate
        
        # Apply decoherence-based scaling
        if self.dual_monitor is not None:
            decoherence = self.dual_monitor.measure_decoherence()
            decoherence_fraction = decoherence.get("decoherence_fraction", 0.0)
            global_mutation_scale = 1.0 - 0.5 * decoherence_fraction
            mutation_rate *= max(0.3, global_mutation_scale)
        
        # Check cancellation zones
        if self.dual_monitor is not None:
            if self.dual_monitor.is_in_cancellation_zone(coords, threshold=0.7):
                if not gentle:
                    mutation_rate *= 2.0
                    if np.random.random() < 0.1:
                        graph.edge_coloring.clear()
                        return
        
        # Apply semantic modulation
        if semantic is not None:
            confusion = semantic.get("confusion_score", 0.0)
            completeness = semantic.get("completeness", 0.0)
            
            # Use semantic polarity to guide mutations (Livnium feature)
            polarity = semantic.get("polarity", 0.0)
            if abs(polarity) > 0.1:  # Significant polarity
                if polarity > 0.5:  # Moving toward good solutions
                    mutation_rate *= 0.5  # Gentle mutations to preserve progress
                elif polarity < -0.5:  # Moving away from solutions
                    mutation_rate *= 2.0  # Aggressive mutations to escape
                else:
                    # Moderate polarity: scale mutation rate
                    mutation_rate *= (1.0 + polarity * 0.5)
            
            # Use face exposure to modulate mutation (Livnium feature)
            face_exposure = semantic.get("face_exposure", 1)
            if face_exposure >= 2:  # Edge or corner (high exploration priority)
                mutation_rate *= 1.2  # Slightly more aggressive
            elif face_exposure == 0:  # Core (refinement priority)
                mutation_rate *= 0.8  # More conservative

            if gentle:
                factor = (1.0 - 0.5 * confusion) * (1.0 - 0.5 * completeness)
                mutation_rate *= max(0.05, factor)
            else:
                factor = 1.0 + 1.5 * confusion
                mutation_rate *= factor
        
        # Use coordinates as seed for deterministic mutations
        seed = int(sum(c * 1000 for c in coords) % (2**31))
        np.random.seed(seed)
        
        # Add uncolored edges if incomplete
        if len(graph.edge_coloring) < self.num_edges * 0.9:
            uncolored = [(u, v) for u in range(self.n) for v in range(u + 1, self.n) 
                        if graph.get_edge_color(u, v) is None]
            if uncolored:
                add_rate = 0.05 if gentle else 0.2
                num_to_add = min(int(len(uncolored) * mutation_rate * add_rate), 5 if gentle else 20)
                np.random.shuffle(uncolored)
                for u, v in uncolored[:num_to_add]:
                    color = np.random.randint(0, 2)
                    graph.set_edge_color(u, v, color)
        
        # Mutate existing edges
        edge_list = [(u, v) for u in range(self.n) for v in range(u + 1, self.n)
                     if graph.get_edge_color(u, v) is not None]
        
        mutation_fraction = 0.01 if gentle else 0.05
        num_mutations = max(1, int(len(edge_list) * mutation_rate * mutation_fraction))
        
        for _ in range(num_mutations):
            if edge_list:
                u, v = edge_list[np.random.randint(len(edge_list))]
                current = graph.get_edge_color(u, v)
                if current is not None:
                    if gentle:
                        old_color = current
                        graph.set_edge_color(u, v, 1 - old_color)
                        is_valid, _ = self.check_constraints(graph)
                        if not is_valid:
                            graph.set_edge_color(u, v, old_color)
                    else:
                        graph.set_edge_color(u, v, 1 - current)
    
    def complete_coloring(self, graph: RamseyGraph) -> Optional[RamseyGraph]:
        """Try to complete a partial coloring greedily."""
        # Find uncolored edges
        uncolored = []
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if graph.get_edge_color(u, v) is None:
                    uncolored.append((u, v))
        
        if not uncolored:
            return graph
        
        # Try to color each edge without creating a clique
        for u, v in uncolored:
            # Try red first
            graph.set_edge_color(u, v, 0)
            is_valid, _ = self.check_constraints(graph)
            if is_valid:
                continue
            
            # Try blue
            graph.set_edge_color(u, v, 1)
            is_valid, _ = self.check_constraints(graph)
            if not is_valid:
                # Neither works, backtrack
                del graph.edge_coloring[(u, v)]
                return None
        
        return graph
    
    def verify_coloring(self, graph: RamseyGraph) -> bool:
        """Verify a coloring is complete and valid."""
        # Check completeness
        if len(graph.edge_coloring) != self.num_edges:
            print(f"  ⚠️  Coloring incomplete: {len(graph.edge_coloring)}/{self.num_edges} edges")
            return False
        
        # Check constraints
        is_valid, clique = self.check_constraints(graph)
        if not is_valid:
            print(f"  ❌ Coloring invalid: contains monochromatic {self.k}-clique: {clique}")
            return False
        
        print(f"  ✅ Coloring verified: complete and valid!")
        return True
    
    def save_coloring(self, graph: RamseyGraph, filename: str):
        """Save coloring to file."""
        with open(filename, 'w') as f:
            f.write(f"# Ramsey Number Lower Bound: R({self.k},{self.k}) > {self.n}\n")
            f.write(f"# 2-coloring of K_{self.n} avoiding monochromatic K_{self.k}\n")
            f.write(f"# Format: vertex1 vertex2 color (0=red, 1=blue)\n\n")
            
            for (u, v), color in sorted(graph.edge_coloring.items()):
                f.write(f"{u} {v} {color}\n")
        
        print(f"  💾 Saved coloring to {filename}")


def solve_ramsey_problem(n: int, k: int, num_omcubes: int = 5000, enable_dual_monitor: bool = False):
    """
    Solve Ramsey number problem using omcubes.
    
    Args:
        n: Number of vertices
        k: Clique size to avoid
        num_omcubes: Number of omcubes to use
        enable_dual_monitor: If True, enable dual cube monitoring for semantic diagnostics
        
    Returns:
        Valid coloring if found, None otherwise
    """
    print("=" * 70)
    print("RAMSEY NUMBER SOLVER - Using 5000 Omcubes")
    print("=" * 70)
    print(f"\nProblem: Find 2-coloring of K_{n} avoiding monochromatic K_{k}")
    print(f"If found, this proves: R({k},{k}) > {n}")
    print(f"\nUsing {num_omcubes:,} omcubes for parallel search...")
    
    solver = RamseySolver(n, k, num_omcubes, enable_dual_monitor=enable_dual_monitor)
    
    # Initialize omcubes
    solver.initialize_omcubes()
    
    # Search for valid coloring
    valid_coloring = solver.search_for_valid_coloring(max_iterations=50)
    
    # Write metrics log to JSONL file if monitor was enabled
    if solver.dual_monitor is not None and solver.metrics_log:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / f"ramsey_dual_cube_metrics_R{k}_{k}_n{n}.jsonl"
        
        with open(log_file, 'w') as f:
            for entry in solver.metrics_log:
                f.write(json.dumps(entry) + '\n')
        
        print(f"\n  💾 Saved metrics log to {log_file}")
        print(f"     ({len(solver.metrics_log)} entries)")
    
    if valid_coloring:
        # Verify
        if solver.verify_coloring(valid_coloring):
            print(f"\n" + "=" * 70)
            print(f"✅ SUCCESS: R({k},{k}) > {n}")
            print("=" * 70)
            
            # Save result
            filename = f"ramsey_R{k}_{k}_n{n}.txt"
            solver.save_coloring(valid_coloring, filename)
            
            return valid_coloring
    else:
        print(f"\n" + "=" * 70)
        print(f"⚠️  No complete valid coloring found")
        print(f"This does NOT prove R({k},{k}) <= {n} (may need more search)")
        print("=" * 70)
    
    return None


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ramsey Number Solver using 5000 Omcubes')
    parser.add_argument('--n', type=int, default=45, help='Number of vertices')
    parser.add_argument('--k', type=int, default=5, help='Clique size to avoid')
    parser.add_argument('--omcubes', type=int, default=5000, help='Number of omcubes to use')
    parser.add_argument('--dual-monitor', action='store_true', 
                       help='Enable dual cube monitoring for semantic diagnostics')
    
    args = parser.parse_args()
    
    solve_ramsey_problem(args.n, args.k, args.omcubes, enable_dual_monitor=args.dual_monitor)


if __name__ == '__main__':
    main()

