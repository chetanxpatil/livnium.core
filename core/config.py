"""
Configuration for Livnium Core System with feature switches.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LivniumCoreConfig:
    """
    Configuration for Livnium Core System with feature switches.
    
    All features can be enabled/disabled independently.
    """
    # Core Structure
    enable_3x3x3_lattice: bool = True  # A1: Canonical Spatial Alphabet
    enable_symbol_alphabet: bool = True  # 27-symbol alphabet (Σ = {0, a...z})
    
    # Symbolic Weight
    enable_symbolic_weight: bool = True  # A3: Symbolic Weight Law (SW = 9·f)
    enable_face_exposure: bool = True  # Face exposure calculation (f ∈ {0,1,2,3})
    enable_class_structure: bool = True  # Core/Center/Edge/Corner classes
    
    # Dynamic Law
    enable_90_degree_rotations: bool = True  # A4: Only 90° quarter-turns
    enable_rotation_group: bool = True  # 24-element rotation group
    
    # Observer System
    enable_global_observer: bool = True  # A2: Global Observer at (0,0,0)
    enable_local_observer: bool = True  # A6: Local Observer designation
    enable_observer_coordinates: bool = True  # Observer-based coordinate system
    
    # Semantic Polarity
    enable_semantic_polarity: bool = True  # A5: cos(θ) between motion and observer
    
    # Cross-Lattice Coupling
    enable_cross_lattice_coupling: bool = True  # A7: Wreath-product transformations
    
    # Quantum Features
    enable_quantum: bool = False  # Enable quantum layer
    enable_superposition: bool = False  # Complex amplitudes per cell
    enable_quantum_gates: bool = False  # Unitary gate operations
    enable_entanglement: bool = False  # Multi-cell entanglement
    enable_measurement: bool = False  # Born rule + collapse
    enable_geometry_quantum_coupling: bool = False  # Geometry ↔ Quantum mapping
    
    # Memory Layer
    enable_memory: bool = False  # Enable memory layer
    enable_working_memory: bool = False  # Working memory per cell
    enable_long_term_memory: bool = False  # Long-term memory consolidation
    enable_memory_coupling: bool = False  # Memory-geometry coupling
    
    # Reasoning Layer
    enable_reasoning: bool = False  # Enable reasoning layer
    enable_search: bool = False  # Search engine
    enable_rules: bool = False  # Rule engine
    enable_problem_solving: bool = False  # Problem solver
    
    # Semantic Layer
    enable_semantic: bool = False  # Enable semantic layer
    enable_feature_extraction: bool = False  # Feature extraction
    enable_meaning_graph: bool = False  # Meaning graph
    enable_inference: bool = False  # Inference engine
    
    # Meta Layer
    enable_meta: bool = False  # Enable meta layer
    enable_introspection: bool = False  # Introspection
    enable_anomaly_detection: bool = False  # Anomaly detection
    enable_calibration: bool = False  # Auto-calibration
    
    # Runtime
    enable_runtime: bool = False  # Enable runtime orchestrator
    enable_episodes: bool = False  # Episode management
    
    # Recursive Geometry (Layer 0)
    enable_recursive_geometry: bool = False  # Enable recursive geometry engine
    recursive_max_depth: int = 3  # Maximum recursion depth
    recursive_subdivision_rule: str = "default"  # Subdivision rule
    enable_moksha: bool = False  # Enable fixed-point convergence (moksha)
    moksha_convergence_threshold: float = 0.999  # Convergence threshold for moksha
    moksha_stability_window: int = 10  # Stability window for convergence check
    
    # Invariants
    enable_sw_conservation: bool = True  # ΣSW = 486 conservation
    enable_class_count_conservation: bool = True  # Class counts {1,6,12,8} conservation
    
    # Hierarchical Extension
    enable_hierarchical_extension: bool = False  # Level-0 (macro) + Level-1 (micro)
    hierarchical_macro_size: int = 3  # Macro lattice size (3×3×3)
    hierarchical_micro_size: int = 3  # Micro lattice size (3×3×3)
    
    # Lattice Size
    lattice_size: int = 3  # N×N×N lattice (must be odd, ≥ 3)
    # Supports: N=3, 5, 7, 9, 11, ... (any odd integer ≥ 3)
    
    # Equilibrium Constant
    equilibrium_constant: float = 10.125  # K = 10.125 (for 3×3×3)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.lattice_size < 3:
            raise ValueError(f"lattice_size must be >= 3, got {self.lattice_size}")
        if self.lattice_size % 2 == 0:
            raise ValueError(f"lattice_size must be odd (3, 5, 7, 9, ...), got {self.lattice_size}")
        
        if self.enable_symbolic_weight and not self.enable_face_exposure:
            raise ValueError("enable_symbolic_weight requires enable_face_exposure")
        
        if self.enable_class_structure and not self.enable_face_exposure:
            raise ValueError("enable_class_structure requires enable_face_exposure")
        
        if self.enable_semantic_polarity and not self.enable_global_observer:
            raise ValueError("enable_semantic_polarity requires enable_global_observer")
        
        if self.enable_local_observer and not self.enable_global_observer:
            raise ValueError("enable_local_observer requires enable_global_observer")
        
        if self.enable_sw_conservation and not self.enable_symbolic_weight:
            raise ValueError("enable_sw_conservation requires enable_symbolic_weight")
        
        if self.enable_class_count_conservation and not self.enable_class_structure:
            raise ValueError("enable_class_count_conservation requires enable_class_structure")
        
        # Quantum feature dependencies
        if self.enable_quantum_gates and not self.enable_superposition:
            raise ValueError("enable_quantum_gates requires enable_superposition")
        if self.enable_entanglement and not self.enable_superposition:
            raise ValueError("enable_entanglement requires enable_superposition")
        if self.enable_measurement and not self.enable_superposition:
            raise ValueError("enable_measurement requires enable_superposition")
        if self.enable_geometry_quantum_coupling and not self.enable_quantum:
            raise ValueError("enable_geometry_quantum_coupling requires enable_quantum")
        
        # Memory dependencies
        if self.enable_long_term_memory and not self.enable_memory:
            raise ValueError("enable_long_term_memory requires enable_memory")
        if self.enable_memory_coupling and not self.enable_memory:
            raise ValueError("enable_memory_coupling requires enable_memory")
        
        # Reasoning dependencies
        if self.enable_rules and not self.enable_reasoning:
            raise ValueError("enable_rules requires enable_reasoning")
        if self.enable_problem_solving and not self.enable_reasoning:
            raise ValueError("enable_problem_solving requires enable_reasoning")
        
        # Semantic dependencies
        if self.enable_meaning_graph and not self.enable_semantic:
            raise ValueError("enable_meaning_graph requires enable_semantic")
        if self.enable_inference and not self.enable_semantic:
            raise ValueError("enable_inference requires enable_semantic")
        
        # Meta dependencies
        if self.enable_introspection and not self.enable_meta:
            raise ValueError("enable_introspection requires enable_meta")
        if self.enable_anomaly_detection and not self.enable_meta:
            raise ValueError("enable_anomaly_detection requires enable_meta")
        if self.enable_calibration and not self.enable_meta:
            raise ValueError("enable_calibration requires enable_meta")
        
        # Runtime dependencies
        if self.enable_episodes and not self.enable_runtime:
            raise ValueError("enable_episodes requires enable_runtime")

