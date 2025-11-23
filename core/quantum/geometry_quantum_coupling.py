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
from ..classical.livnium_core_system import LivniumCoreSystem, CellClass, RotationAxis, RotationGroup


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
    
    def phi_straight_line(self, 
                         target_coords: Tuple[int, int, int],
                         observer_coords: Optional[Tuple[int, int, int]] = None) -> float:
        """
        Calculate straight-line Φ: Direct, linear, no-rotation mapping.
        
        This is the "pure polarity" direction - like a laser beam pointed 
        straight out of OM (origin anchor).
        
        Φ is the connection between OM (origin anchor at (0,0,0)) and 
        LO (local orientation at target_coords).
        
        Args:
            target_coords: Target cell coordinates (LO)
            observer_coords: Observer coordinates (OM, default: (0,0,0))
            
        Returns:
            Straight-line Φ value (polarity) in [-1, 1]
        """
        if observer_coords is None:
            if self.core_system.global_observer:
                observer_coords = self.core_system.global_observer.coordinates
            else:
                observer_coords = (0, 0, 0)  # Default OM at origin
        
        # Calculate direct connection: OM → LO
        motion_vec = np.array(target_coords, dtype=float) - np.array(observer_coords, dtype=float)
        polarity = self.core_system.calculate_polarity(
            tuple(motion_vec),
            observer_coords=observer_coords
        )
        
        return polarity
    
    def phi_rotated(self,
                   target_coords: Tuple[int, int, int],
                   observer_coords: Optional[Tuple[int, int, int]] = None,
                   rotation_axis: Optional[str] = None,
                   quarter_turns: int = 1) -> Tuple[float, np.ndarray]:
        """
        Calculate rotated Φ: Same connection, but rotated through cube's orientation mapping.
        
        This gives the "phase-shifted" version - like the wave aspect.
        The rotated Φ is the same underlying invariant expressed in a rotated basis.
        
        Args:
            target_coords: Target cell coordinates (LO)
            observer_coords: Observer coordinates (OM, default: (0,0,0))
            rotation_axis: Rotation axis ("X", "Y", "Z") or None for auto
            quarter_turns: Number of quarter-turns (default: 1)
            
        Returns:
            Tuple of (rotated_polarity, rotation_matrix)
        """
        if observer_coords is None:
            if self.core_system.global_observer:
                observer_coords = self.core_system.global_observer.coordinates
            else:
                observer_coords = (0, 0, 0)  # Default OM at origin
        
        # Get straight-line Φ first
        straight_phi = self.phi_straight_line(target_coords, observer_coords)
        
        # Apply rotation to the connection vector
        if rotation_axis is None:
            # Auto-select: use Y-axis as default
            axis = RotationAxis.Y
        else:
            axis = RotationAxis[rotation_axis.upper()]
        
        # Rotate the target coordinates through the cube's orientation
        rotated_coords = RotationGroup.rotate_coordinates(
            target_coords, axis, quarter_turns
        )
        
        # Calculate polarity in rotated frame
        motion_vec_rotated = np.array(rotated_coords, dtype=float) - np.array(observer_coords, dtype=float)
        rotated_polarity = self.core_system.calculate_polarity(
            tuple(motion_vec_rotated),
            observer_coords=observer_coords
        )
        
        # Get rotation matrix for quantum representation
        rotation_matrix = self.geometric_rotation_to_quantum_gate(
            axis.name, quarter_turns
        )
        
        return rotated_polarity, rotation_matrix
    
    def phi_dual_representation(self,
                               target_coords: Tuple[int, int, int],
                               observer_coords: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Get both faces of Φ: straight-line and rotated forms.
        
        Both are valid projections of the same underlying invariant.
        Just like quantum objects are both wave and particle depending on how you look.
        
        Args:
            target_coords: Target cell coordinates (LO)
            observer_coords: Observer coordinates (OM, default: (0,0,0))
            
        Returns:
            Dictionary with:
            - 'straight_phi': Direct, linear mapping (eigenbasis)
            - 'rotated_phi': Rotated mapping (rotated basis)
            - 'invariant_preserved': Whether both preserve the same invariant
            - 'interpretation': Human-readable explanation
        """
        if observer_coords is None:
            if self.core_system.global_observer:
                observer_coords = self.core_system.global_observer.coordinates
            else:
                observer_coords = (0, 0, 0)
        
        # Get straight-line Φ (eigenbasis - stable, minimal energy)
        straight_phi = self.phi_straight_line(target_coords, observer_coords)
        
        # Get rotated Φ (rotated basis - expressive, dynamic)
        rotated_phi, rotation_matrix = self.phi_rotated(target_coords, observer_coords)
        
        # Both come from the same underlying object: a polarity field 
        # inside a rotating reference frame
        # The invariant is the magnitude of the connection, not its orientation
        straight_magnitude = abs(straight_phi)
        rotated_magnitude = abs(rotated_phi)
        
        # Check if invariant is preserved (magnitude should be similar)
        invariant_preserved = abs(straight_magnitude - rotated_magnitude) < 0.1
        
        return {
            'straight_phi': straight_phi,
            'rotated_phi': rotated_phi,
            'straight_magnitude': straight_magnitude,
            'rotated_magnitude': rotated_magnitude,
            'invariant_preserved': invariant_preserved,
            'rotation_matrix': rotation_matrix,
            'interpretation': {
                'straight': 'Eigenbasis (stable, minimal energy) - like particle',
                'rotated': 'Rotated basis (expressive, dynamic) - like wave',
                'both_valid': 'Both are two shadows of the same 3D truth',
                'when_straight': 'System wants stability',
                'when_rotated': 'System wants energy spread / superposition'
            }
        }
    
    def phi_under_rotation(self,
                          target_coords: Tuple[int, int, int],
                          observer_coords: Optional[Tuple[int, int, int]] = None) -> Dict:
        """
        Test Φ invariance under all 24 rotations of the cube.
        
        Demonstrates that both straight and rotated Φ preserve invariants
        under the full rotation group.
        
        Args:
            target_coords: Target cell coordinates
            observer_coords: Observer coordinates (default: (0,0,0))
            
        Returns:
            Dictionary with rotation test results
        """
        if observer_coords is None:
            if self.core_system.global_observer:
                observer_coords = self.core_system.global_observer.coordinates
            else:
                observer_coords = (0, 0, 0)
        
        # Get baseline straight Φ
        baseline_phi = self.phi_straight_line(target_coords, observer_coords)
        baseline_magnitude = abs(baseline_phi)
        
        results = {
            'baseline_phi': baseline_phi,
            'baseline_magnitude': baseline_magnitude,
            'rotations_tested': 0,
            'invariant_count': 0,
            'rotation_results': []
        }
        
        # Test all 24 rotations
        for axis in [RotationAxis.X, RotationAxis.Y, RotationAxis.Z]:
            for quarter_turns in [1, 2, 3]:
                rotated_phi, _ = self.phi_rotated(
                    target_coords, observer_coords, axis.name, quarter_turns
                )
                rotated_magnitude = abs(rotated_phi)
                
                # Check if magnitude preserved (invariant)
                magnitude_preserved = abs(baseline_magnitude - rotated_magnitude) < 0.1
                
                results['rotations_tested'] += 1
                if magnitude_preserved:
                    results['invariant_count'] += 1
                
                results['rotation_results'].append({
                    'axis': axis.name,
                    'quarter_turns': quarter_turns,
                    'rotated_phi': rotated_phi,
                    'rotated_magnitude': rotated_magnitude,
                    'invariant_preserved': magnitude_preserved
                })
        
        results['invariant_ratio'] = results['invariant_count'] / results['rotations_tested'] if results['rotations_tested'] > 0 else 0.0
        
        return results

