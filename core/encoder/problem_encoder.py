"""
Universal Problem Encoder: Convert Any Problem to Geometry

This is the main interface for encoding problems into geometric patterns:
- Constraints → Tension fields (energy landscape)
- Solutions → Basins (candidate attractors)

The encoder produces:
1. Tension fields (from constraints)
2. Candidate basins (from solution space)

These are then passed to multi-basin search.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from core.classical.livnium_core_system import LivniumCoreSystem
from .constraint_encoder import ConstraintEncoder, TensionField


@dataclass
class EncodedProblem:
    """
    Result of encoding a problem into geometry.
    
    Contains:
    - Tension fields (from constraints)
    - Candidate basins (from solution space)
    - Variable mappings (for decoding)
    """
    tension_fields: List[TensionField]
    candidate_basins: List[List[Tuple[int, int, int]]]
    variable_mappings: Dict[str, List[Tuple[int, int, int]]]
    problem_type: str
    metadata: Dict[str, Any]


class UniversalProblemEncoder:
    """
    Universal encoder that converts any problem into geometric patterns.
    
    Architecture:
    - Constraints → Tension fields (energy landscape)
    - Solutions → Basins (candidate attractors)
    """
    
    def __init__(self, system: LivniumCoreSystem):
        """
        Initialize universal problem encoder.
        
        Args:
            system: LivniumCoreSystem
        """
        self.system = system
        self.constraint_encoder = ConstraintEncoder(system)
    
    def encode(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode a problem into geometry.
        
        Args:
            problem: Problem specification with:
                - 'type': Problem type (e.g., 'graph_coloring', 'sat', 'ramsey')
                - 'variables': Variable definitions
                - 'constraints': Constraint definitions
                - 'candidates': Candidate solutions (optional)
                - Other problem-specific fields
        
        Returns:
            EncodedProblem with tension fields and candidate basins
        """
        problem_type = problem.get('type', 'unknown')
        
        # Route to specific encoder based on problem type
        if problem_type == 'graph_coloring':
            return self._encode_graph_coloring(problem)
        elif problem_type == 'sat':
            return self._encode_sat(problem)
        elif problem_type == 'ramsey':
            return self._encode_ramsey(problem)
        elif problem_type == 'constraint_satisfaction':
            return self._encode_constraint_satisfaction(problem)
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def _encode_graph_coloring(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode standard Graph (Vertex) Coloring problem.
        
        Constraint: Adjacent vertices cannot have the same color.
        Variables: Vertices
        Constraints: Edges (adjacency relationships)
        Solutions: Valid vertex colorings
        
        Args:
            problem: Graph coloring specification with:
                - 'vertices': List of vertex identifiers
                - 'edges': List of (u, v) edge tuples
                - 'n_candidates': Number of candidate colorings (default: 10)
        
        Returns:
            EncodedProblem
        """
        vertices = problem.get('vertices', [])
        edges = problem.get('edges', [])
        
        # 1. Variables are VERTICES, not edges
        variable_mappings = {}
        coords_list = list(self.system.lattice.keys())
        
        for i, vertex in enumerate(vertices):
            if i < len(coords_list):
                # Map vertex to a coordinate
                variable_mappings[vertex] = [coords_list[i]]
        
        # 2. Constraints are EDGES (checking adjacency)
        tension_fields = []
        
        for u, v in edges:
            if u not in variable_mappings or v not in variable_mappings:
                continue
            
            u_coords = variable_mappings[u]
            v_coords = variable_mappings[v]
            involved_coords = u_coords + v_coords
            
            # Tension function: High if colors match (adjacent vertices same color)
            def create_adjacency_tension_fn(u_coords, v_coords):
                def compute_tension(system: LivniumCoreSystem) -> float:
                    # Get colors (using threshold approach)
                    # Assumes single coordinate per var for simplicity
                    cell1 = system.get_cell(u_coords[0]) if u_coords else None
                    cell2 = system.get_cell(v_coords[0]) if v_coords else None
                    
                    if not cell1 or not cell2:
                        return 0.0
                    
                    # Simple binary coloring check (Red/Blue)
                    # You can expand this for k-coloring later
                    color1 = 0 if cell1.symbolic_weight < 10.0 else 1
                    color2 = 0 if cell2.symbolic_weight < 10.0 else 1
                    
                    # VIOLATION if colors are equal (adjacent vertices same color)
                    return 1.0 if color1 == color2 else 0.0
                
                return compute_tension
            
            field = self.constraint_encoder.encode_custom_constraint(
                constraint_id=f"edge_{u}_{v}",
                involved_coords=involved_coords,
                tension_fn=create_adjacency_tension_fn(u_coords, v_coords),
                description=f"Adjacent vertices {u}-{v} must differ"
            )
            tension_fields.append(field)
        
        # Generate candidate basins (valid colorings)
        # For now, generate random colorings
        candidate_basins = []
        n_candidates = problem.get('n_candidates', 10)
        import random
        
        for _ in range(n_candidates):
            # Random coloring: assign random colors to vertices
            basin_coords = []
            for vertex in vertices:
                if vertex in variable_mappings:
                    basin_coords.extend(variable_mappings[vertex])
            candidate_basins.append(basin_coords)
        
        return EncodedProblem(
            tension_fields=tension_fields,
            candidate_basins=candidate_basins,
            variable_mappings=variable_mappings,
            problem_type='graph_coloring',
            metadata={'vertices': vertices, 'edges': edges}
        )
    
    def _encode_sat(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode SAT problem.
        
        Constraints: Clauses (tension if clause unsatisfied)
        Solutions: Variable assignments (basins)
        
        Args:
            problem: SAT specification with:
                - 'clauses': List of clauses, each clause is list of literals (ints)
                - 'num_vars': Number of variables
                - 'n_candidates': Number of candidate assignments (default: 20)
        
        Returns:
            EncodedProblem
        """
        clauses = problem.get('clauses', [])
        num_vars = problem.get('num_vars', 0)
        n_candidates = problem.get('n_candidates', 20)
        
        if not clauses or num_vars == 0:
            raise ValueError("SAT problem must have clauses and num_vars")
        
        # Map variables to coordinates
        # Each variable gets one coordinate (or multiple for redundancy)
        coords_list = list(self.system.lattice.keys())
        variable_mappings = {}
        
        # Assign coordinates to variables
        for var_id in range(1, num_vars + 1):  # Variables are 1-indexed
            if (var_id - 1) < len(coords_list):
                var_coords = [coords_list[var_id - 1]]
            else:
                # Wrap around if more variables than coordinates
                var_coords = [coords_list[(var_id - 1) % len(coords_list)]]
            variable_mappings[f"var_{var_id}"] = var_coords
            variable_mappings[f"var_{-var_id}"] = var_coords  # Negation uses same coords
        
        # Encode clauses as tension fields
        tension_fields = []
        
        for clause_idx, clause in enumerate(clauses):
            # Get coordinates for literals in this clause
            clause_coords = []
            for literal in clause:
                var_id = abs(literal)
                if f"var_{var_id}" in variable_mappings:
                    clause_coords.extend(variable_mappings[f"var_{var_id}"])
            
            if not clause_coords:
                continue
            
            # Create tension function: high tension if clause is unsatisfied
            def create_clause_tension_fn(clause_literals, literal_coords_map):
                def compute_tension(system: LivniumCoreSystem) -> float:
                    # Check if clause is satisfied
                    clause_satisfied = False
                    
                    for literal in clause_literals:
                        var_id = abs(literal)
                        if f"var_{var_id}" not in literal_coords_map:
                            continue
                        
                        coords = literal_coords_map[f"var_{var_id}"]
                        if not coords:
                            continue
                        
                        # Get cell value (symbolic weight)
                        cell = system.get_cell(coords[0])
                        if not cell:
                            continue
                        
                        # Decode assignment: SW < 10 = False, SW >= 10 = True
                        var_value = cell.symbolic_weight >= 10.0
                        
                        # Check if literal is satisfied
                        if literal > 0 and var_value:
                            clause_satisfied = True
                            break
                        elif literal < 0 and not var_value:
                            clause_satisfied = True
                            break
                    
                    # Tension = 1.0 if clause unsatisfied, 0.0 if satisfied
                    return 1.0 if not clause_satisfied else 0.0
                
                return compute_tension
            
            # Create mapping for this clause
            literal_coords_map = {}
            for literal in clause:
                var_id = abs(literal)
                if f"var_{var_id}" in variable_mappings:
                    literal_coords_map[f"var_{var_id}"] = variable_mappings[f"var_{var_id}"]
            
            field = self.constraint_encoder.encode_custom_constraint(
                constraint_id=f"clause_{clause_idx}",
                involved_coords=clause_coords,
                tension_fn=create_clause_tension_fn(clause, literal_coords_map),
                description=f"Clause {clause_idx}: {clause}"
            )
            tension_fields.append(field)
        
        # Generate candidate basins (variable assignments)
        candidate_basins = []
        import random
        
        for _ in range(n_candidates):
            # Random assignment: each variable gets True/False
            basin_coords = []
            for var_id in range(1, num_vars + 1):
                if f"var_{var_id}" in variable_mappings:
                    basin_coords.extend(variable_mappings[f"var_{var_id}"])
            
            # Initialize cells with random assignments
            for coords in basin_coords:
                cell = self.system.get_cell(coords)
                if cell:
                    # Random assignment: < 10 = False, >= 10 = True
                    cell.symbolic_weight = random.choice([5.0, 15.0])
            
            candidate_basins.append(basin_coords)
        
        return EncodedProblem(
            tension_fields=tension_fields,
            candidate_basins=candidate_basins,
            variable_mappings=variable_mappings,
            problem_type='sat',
            metadata={
                'num_vars': num_vars,
                'num_clauses': len(clauses),
                'clauses': clauses
            }
        )
    
    def _encode_ramsey(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode Ramsey problem (Edge Coloring with Triangle/K₄ constraints).
        
        Constraints: No monochromatic triangles (K₃) or K₄ cliques
        Variables: Edges (edge coloring)
        Solutions: Valid 2-colorings of edges
        
        Args:
            problem: Ramsey specification with:
                - 'vertices': List of vertex identifiers
                - 'edges': List of (u, v) edge tuples
                - 'constraint_type': 'k3' (triangles) or 'k4' (4-cliques), default 'k3'
                - 'n_candidates': Number of candidate colorings (default: 10)
        
        Returns:
            EncodedProblem
        """
        vertices = problem.get('vertices', [])
        edges = problem.get('edges', [])
        constraint_type = problem.get('constraint_type', 'k3')  # 'k3' or 'k4'
        
        # Map variables (edges) to coordinates
        variable_mappings = {}
        coords_list = list(self.system.lattice.keys())
        
        edge_to_coords = {}
        for i, edge in enumerate(edges):
            if i < len(coords_list):
                coords = [coords_list[i]]
                edge_to_coords[edge] = coords
                variable_mappings[f"edge_{edge}"] = coords
        
        # Encode constraints: No monochromatic triangles (K₃) or K₄
        tension_fields = []
        
        if constraint_type == 'k3':
            # Find all triangles
            from itertools import combinations
            triangles = list(combinations(vertices, 3))
            
            for triangle in triangles:
                # Get edges of triangle
                triangle_edges = [
                    (triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[0], triangle[2])
                ]
                
                # Get coordinates for triangle edges
                triangle_coords = []
                for edge in triangle_edges:
                    # Handle both (u,v) and (v,u) edge orderings
                    if edge in edge_to_coords:
                        triangle_coords.extend(edge_to_coords[edge])
                    elif (edge[1], edge[0]) in edge_to_coords:
                        triangle_coords.extend(edge_to_coords[(edge[1], edge[0])])
                
                if not triangle_coords:
                    continue
                
                # Create tension field: high tension if all edges same color
                def create_triangle_tension_fn(edge_coords_list):
                    def compute_tension(system: LivniumCoreSystem) -> float:
                        # Get SW values (colors) for edges
                        sw_values = []
                        for coords in edge_coords_list:
                            cell = system.get_cell(coords)
                            if cell:
                                # Decode color: < 10 = color 0, >= 10 = color 1
                                color = 0 if cell.symbolic_weight < 10.0 else 1
                                sw_values.append(color)
                        
                        if len(sw_values) < 3:
                            return 0.0
                        
                        # Tension = 1.0 if all same color (monochromatic), 0.0 otherwise
                        if len(set(sw_values)) == 1:
                            return 1.0  # Monochromatic triangle
                        return 0.0  # Valid coloring
                    
                    return compute_tension
                
                field = self.constraint_encoder.encode_custom_constraint(
                    constraint_id=f"triangle_{triangle}",
                    involved_coords=triangle_coords,
                    tension_fn=create_triangle_tension_fn(triangle_coords),
                    description=f"No monochromatic triangle: {triangle}"
                )
                tension_fields.append(field)
        
        elif constraint_type == 'k4':
            # Find all K₄ cliques (4-vertex complete subgraphs)
            from itertools import combinations
            k4_cliques = list(combinations(vertices, 4))
            
            for clique in k4_cliques:
                # Get all 6 edges of K₄
                clique_edges = list(combinations(clique, 2))
                
                # Get coordinates for clique edges
                clique_coords = []
                for edge in clique_edges:
                    if edge in edge_to_coords:
                        clique_coords.extend(edge_to_coords[edge])
                    elif (edge[1], edge[0]) in edge_to_coords:
                        clique_coords.extend(edge_to_coords[(edge[1], edge[0])])
                
                if not clique_coords:
                    continue
                
                # Create tension field: high tension if all edges same color
                def create_k4_tension_fn(edge_coords_list):
                    def compute_tension(system: LivniumCoreSystem) -> float:
                        sw_values = []
                        for coords in edge_coords_list:
                            cell = system.get_cell(coords)
                            if cell:
                                color = 0 if cell.symbolic_weight < 10.0 else 1
                                sw_values.append(color)
                        
                        if len(sw_values) < 6:
                            return 0.0
                        
                        # Tension = 1.0 if all same color (monochromatic K₄)
                        if len(set(sw_values)) == 1:
                            return 1.0
                        return 0.0
                    
                    return compute_tension
                
                field = self.constraint_encoder.encode_custom_constraint(
                    constraint_id=f"k4_{clique}",
                    involved_coords=clique_coords,
                    tension_fn=create_k4_tension_fn(clique_coords),
                    description=f"No monochromatic K₄: {clique}"
                )
                tension_fields.append(field)
        
        # Generate candidate basins (valid colorings)
        candidate_basins = []
        n_candidates = problem.get('n_candidates', 10)
        import random
        
        for _ in range(n_candidates):
            # Random coloring: assign random colors to edges
            basin_coords = []
            for edge, coords in edge_to_coords.items():
                basin_coords.extend(coords)
            candidate_basins.append(basin_coords)
        
        return EncodedProblem(
            tension_fields=tension_fields,
            candidate_basins=candidate_basins,
            variable_mappings=variable_mappings,
            problem_type='ramsey',
            metadata={
                'vertices': vertices,
                'edges': edges,
                'constraint_type': constraint_type
            }
        )
    
    def _encode_constraint_satisfaction(
        self,
        problem: Dict[str, Any]
    ) -> EncodedProblem:
        """
        Encode general constraint satisfaction problem.
        
        Args:
            problem: CSP specification with:
                - 'variables': Dict[str, List[Any]] - variable names to domains
                - 'constraints': List[Dict] - constraint definitions
                - 'n_candidates': Number of candidate solutions (default: 20)
        
        Returns:
            EncodedProblem
        """
        variables = problem.get('variables', {})
        constraints = problem.get('constraints', [])
        n_candidates = problem.get('n_candidates', 20)
        
        if not variables or not constraints:
            raise ValueError("CSP problem must have variables and constraints")
        
        # Map variables to coordinates
        coords_list = list(self.system.lattice.keys())
        variable_mappings = {}
        
        for i, var_name in enumerate(variables.keys()):
            if i < len(coords_list):
                var_coords = [coords_list[i]]
            else:
                var_coords = [coords_list[i % len(coords_list)]]
            variable_mappings[var_name] = var_coords
        
        # Encode constraints as tension fields
        tension_fields = []
        
        for constraint_idx, constraint in enumerate(constraints):
            constraint_type = constraint.get('type', 'custom')
            vars_involved = constraint.get('vars', [])
            
            # Get coordinates for variables in this constraint
            constraint_coords = []
            for var_name in vars_involved:
                if var_name in variable_mappings:
                    constraint_coords.extend(variable_mappings[var_name])
            
            if not constraint_coords:
                continue
            
            # Create tension function based on constraint type
            def create_constraint_tension_fn(constraint_def, var_map, var_domains):
                def compute_tension(system: LivniumCoreSystem) -> float:
                    # Decode assignment from system
                    assignment = {}
                    for var_name in constraint_def.get('vars', []):
                        if var_name not in var_map:
                            continue
                        coords = var_map[var_name]
                        if not coords:
                            continue
                        cell = system.get_cell(coords[0])
                        if not cell:
                            continue
                        
                        # Map SW to domain value
                        domain = var_domains.get(var_name, [])
                        if domain:
                            sw_value = int(cell.symbolic_weight) % len(domain)
                            assignment[var_name] = domain[sw_value]
                    
                    # Check constraint satisfaction
                    constraint_type = constraint_def.get('type', 'custom')
                    vars_involved = constraint_def.get('vars', [])
                    
                    if not all(v in assignment for v in vars_involved):
                        return 1.0  # Missing variables = violation
                    
                    values = [assignment[v] for v in vars_involved]
                    
                    if constraint_type == 'all_different':
                        # All different: tension = 0 if all different, >0 if duplicates
                        num_unique = len(set(values))
                        num_total = len(values)
                        if num_unique == num_total:
                            return 0.0
                        # Tension proportional to number of duplicates
                        return float(num_total - num_unique) / num_total
                    
                    elif constraint_type == 'equal':
                        # All equal: tension = 0 if all equal, >0 if different
                        num_unique = len(set(values))
                        if num_unique == 1:
                            return 0.0
                        return float(num_unique - 1) / len(values)
                    
                    elif constraint_type == 'not_equal':
                        # Not equal: tension = 0 if all different, >0 if any equal
                        if len(values) == len(set(values)):
                            return 0.0
                        return 1.0
                    
                    elif constraint_type == 'custom':
                        # Custom constraint function
                        fn = constraint_def.get('fn')
                        if fn:
                            is_satisfied = fn(assignment)
                            return 0.0 if is_satisfied else 1.0
                    
                    return 0.0
                
                return compute_tension
            
            field = self.constraint_encoder.encode_custom_constraint(
                constraint_id=f"constraint_{constraint_idx}",
                involved_coords=constraint_coords,
                tension_fn=create_constraint_tension_fn(constraint, variable_mappings, variables),
                description=f"Constraint {constraint_idx}: {constraint_type}"
            )
            tension_fields.append(field)
        
        # Generate candidate basins
        candidate_basins = []
        import random
        
        for _ in range(n_candidates):
            basin_coords = []
            for var_name in variables.keys():
                if var_name in variable_mappings:
                    basin_coords.extend(variable_mappings[var_name])
            
            # Initialize cells with random domain values
            for var_name, coords in variable_mappings.items():
                domain = variables[var_name]
                if domain and coords:
                    cell = self.system.get_cell(coords[0])
                    if cell:
                        # Set SW to a value that maps to a random domain element
                        domain_idx = random.randint(0, len(domain) - 1)
                        # Map domain index to SW (use range 0-100, then modulo)
                        cell.symbolic_weight = float(domain_idx * 10)
            
            candidate_basins.append(basin_coords)
        
        return EncodedProblem(
            tension_fields=tension_fields,
            candidate_basins=candidate_basins,
            variable_mappings=variable_mappings,
            problem_type='constraint_satisfaction',
            metadata={
                'variables': list(variables.keys()),
                'num_constraints': len(constraints),
                'constraints': constraints
            }
        )

