"""
Geometry ↔ Quantum Coupling: Livnium-Specific Integration

Maps geometric properties (face exposure, symbolic weight, polarity) to quantum state.
This is the "magic sauce" that makes Livnium unique.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .quantum_cell import QuantumCell
from .quantum_gates import QuantumGates, GateType
from ..classical.livnium_core_system import LivniumCoreSystem, CellClass


class GeometryQuantumCoupling:
    """
    Couples geometric properties to quantum state.
    
    Rules:
    - Face exposure → entanglement connections
    - Symbolic Weight → amplitude strength
    - Polarity → phase
    - Observer → measurement basis
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize geometry-quantum coupling.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
    
    def initialize_quantum_state_from_geometry(self, 
                                              cell: QuantumCell,
                                              geometric_cell) -> QuantumCell:
        """
        Initialize quantum state based on geometric properties.
        
        Rules:
        - Face exposure f → superposition strength
        - Symbolic Weight SW → amplitude magnitude
        - Class → initial state
        
        Args:
            cell: Quantum cell to initialize
            geometric_cell: Geometric cell from core system
            
        Returns:
            Initialized quantum cell
        """
        f = geometric_cell.face_exposure
        sw = geometric_cell.symbolic_weight
        
        # Rule: Higher SW → stronger superposition
        # Normalize SW to [0, 1] range (max SW = 27 for corners)
        sw_normalized = sw / 27.0
        
        # Initialize superposition: |ψ⟩ = √(1-p)|0⟩ + √p|1⟩
        # where p depends on SW
        p = sw_normalized * 0.5  # Scale down for stability
        alpha_0 = np.sqrt(1 - p)
        alpha_1 = np.sqrt(p)
        
        cell.set_state_vector([alpha_0, alpha_1])
        return cell
    
    def apply_polarity_to_phase(self, cell: QuantumCell,
                               polarity: float) -> QuantumCell:
        """
        Apply semantic polarity to quantum phase.
        
        Rule: Polarity → phase shift
        - Positive polarity (+1) → phase = 0
        - Negative polarity (-1) → phase = π
        - Neutral (0) → phase = π/2
        
        Args:
            cell: Quantum cell
            polarity: Semantic polarity value [-1, 1]
            
        Returns:
            Cell with phase applied
        """
        # Map polarity to phase: [-1, 1] → [π, 0]
        phase = (1 - polarity) * np.pi / 2
        
        # Apply phase gate
        phase_gate = QuantumGates.phase(phase)
        cell.apply_unitary(phase_gate)
        
        return cell
    
    def face_exposure_to_entanglement_strength(self, 
                                               face_exposure: int) -> float:
        """
        Map face exposure to entanglement strength.
        
        Rule: Higher face exposure → stronger entanglement
        
        Args:
            face_exposure: Face exposure value (0-3)
            
        Returns:
            Entanglement strength [0, 1]
        """
        # Linear mapping: f ∈ {0,1,2,3} → strength ∈ {0, 0.33, 0.67, 1.0}
        return face_exposure / 3.0
    
    def symbolic_weight_to_amplitude_modulation(self,
                                               sw: float) -> float:
        """
        Map symbolic weight to amplitude modulation factor.
        
        Rule: Higher SW → stronger amplitudes
        
        Args:
            sw: Symbolic weight value
            
        Returns:
            Amplitude modulation factor [0, 1]
        """
        # Normalize SW (max = 27 for corners)
        return min(1.0, sw / 27.0)
    
    def observer_dependent_measurement_basis(self,
                                           cell: QuantumCell,
                                           observer_coords: Tuple[int, int, int]) -> np.ndarray:
        """
        Create measurement basis dependent on observer position.
        
        Rule: Observer position → rotated measurement basis
        
        Args:
            cell: Quantum cell
            observer_coords: Observer coordinates
            
        Returns:
            Measurement basis (rotation matrix)
        """
        # Calculate vector from cell to observer
        cell_array = np.array(cell.coordinates)
        obs_array = np.array(observer_coords)
        direction = obs_array - cell_array
        
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            # Observer at cell: use standard basis
            return np.eye(2, dtype=complex)
        
        direction = direction / norm
        
        # Map direction to rotation angle
        # Use X-component to determine rotation about Y-axis
        theta = np.arccos(np.clip(direction[0], -1, 1))
        
        # Create rotation gate
        return QuantumGates.rotation_y(theta)
    
    def geometric_rotation_to_quantum_gate(self,
                                         rotation_axis: str,
                                         quarter_turns: int) -> np.ndarray:
        """
        Map geometric rotation to quantum gate.
        
        Rule: 90° geometric rotation → quantum rotation gate
        
        Args:
            rotation_axis: Rotation axis ("X", "Y", "Z")
            quarter_turns: Number of quarter-turns
            
        Returns:
            Quantum gate (unitary matrix)
        """
        # Map geometric rotation to quantum rotation
        angle = quarter_turns * np.pi / 2
        
        if rotation_axis.upper() == "X":
            return QuantumGates.rotation_x(angle)
        elif rotation_axis.upper() == "Y":
            return QuantumGates.rotation_y(angle)
        elif rotation_axis.upper() == "Z":
            return QuantumGates.rotation_z(angle)
        else:
            raise ValueError(f"Unknown rotation axis: {rotation_axis}")
    
    def class_to_initial_state(self, cell_class: CellClass) -> np.ndarray:
        """
        Map cell class to initial quantum state.
        
        Rule:
        - Core (f=0) → |0⟩
        - Centers (f=1) → (|0⟩ + |1⟩)/√2
        - Edges (f=2) → (|0⟩ + i|1⟩)/√2
        - Corners (f=3) → |1⟩
        
        Args:
            cell_class: Cell class
            
        Returns:
            Initial state vector
        """
        if cell_class == CellClass.CORE:
            return np.array([1.0, 0.0], dtype=complex)  # |0⟩
        elif cell_class == CellClass.CENTER:
            return np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)  # (|0⟩ + |1⟩)/√2
        elif cell_class == CellClass.EDGE:
            return np.array([1/np.sqrt(2), 1j/np.sqrt(2)], dtype=complex)  # (|0⟩ + i|1⟩)/√2
        elif cell_class == CellClass.CORNER:
            return np.array([0.0, 1.0], dtype=complex)  # |1⟩
        else:
            return np.array([1.0, 0.0], dtype=complex)  # Default: |0⟩
    
    def update_quantum_from_geometry(self, 
                                    quantum_cells: Dict[Tuple[int, int, int], QuantumCell]):
        """
        Update all quantum states based on current geometry.
        
        This is the main coupling function that keeps quantum and geometry in sync.
        
        Args:
            quantum_cells: Dictionary of quantum cells
        """
        for coords, quantum_cell in quantum_cells.items():
            geometric_cell = self.core_system.get_cell(coords)
            
            if geometric_cell:
                # Update based on geometric properties
                initial_state = self.class_to_initial_state(geometric_cell.cell_class)
                quantum_cell.set_state_vector(initial_state)
                
                # Apply SW modulation
                sw_factor = self.symbolic_weight_to_amplitude_modulation(geometric_cell.symbolic_weight)
                state = quantum_cell.get_state_vector()
                state = state * np.sqrt(sw_factor)
                quantum_cell.set_state_vector(state)
                
                # Apply polarity to phase if observer exists
                if self.core_system.global_observer:
                    motion_vec = np.array(coords, dtype=float)
                    polarity = self.core_system.calculate_polarity(motion_vec)
                    self.apply_polarity_to_phase(quantum_cell, polarity)

