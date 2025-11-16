"""
Quantum Lattice: Quantum State Management for Livnium Core

Integrates quantum layer with geometric lattice.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .quantum_cell import QuantumCell
from .quantum_gates import QuantumGates, GateType
from .entanglement_manager import EntanglementManager
from .measurement_engine import MeasurementEngine
from .geometry_quantum_coupling import GeometryQuantumCoupling
from ..classical.livnium_core_system import LivniumCoreSystem
from ..config import LivniumCoreConfig


class QuantumLattice:
    """
    Quantum layer for Livnium Core System.
    
    Manages quantum states, gates, entanglement, and measurement.
    """
    
    def __init__(self, core_system: LivniumCoreSystem):
        """
        Initialize quantum lattice.
        
        Args:
            core_system: Livnium Core System instance
        """
        self.core_system = core_system
        self.config = core_system.config
        
        if not self.config.enable_quantum:
            raise ValueError("Quantum features not enabled in config")
        
        # Quantum cells: one per geometric cell
        self.quantum_cells: Dict[Tuple[int, int, int], QuantumCell] = {}
        self._initialize_quantum_cells()
        
        # Entanglement manager
        if self.config.enable_entanglement:
            self.entanglement_manager = EntanglementManager(self.core_system.lattice_size)
        else:
            self.entanglement_manager = None
        
        # Measurement engine
        if self.config.enable_measurement:
            self.measurement_engine = MeasurementEngine()
        else:
            self.measurement_engine = None
        
        # Geometry-quantum coupling
        if self.config.enable_geometry_quantum_coupling:
            self.coupling = GeometryQuantumCoupling(core_system)
            self._sync_with_geometry()
        else:
            self.coupling = None
    
    def _initialize_quantum_cells(self):
        """Initialize quantum cells for all geometric cells."""
        for coords, geometric_cell in self.core_system.lattice.items():
            # Default: |0⟩ state
            quantum_cell = QuantumCell(
                coordinates=coords,
                amplitudes=np.array([1.0, 0.0], dtype=complex),
                num_levels=2
            )
            self.quantum_cells[coords] = quantum_cell
    
    def _sync_with_geometry(self):
        """Sync quantum states with geometric properties."""
        if self.coupling:
            self.coupling.update_quantum_from_geometry(self.quantum_cells)
    
    def apply_gate(self, coords: Tuple[int, int, int],
                   gate_type: GateType, **params):
        """
        Apply quantum gate to a cell.
        
        Args:
            coords: Cell coordinates
            gate_type: Type of gate
            **params: Gate parameters
        """
        if not self.config.enable_quantum_gates:
            raise ValueError("Quantum gates not enabled")
        
        if coords not in self.quantum_cells:
            raise ValueError(f"Cell not found: {coords}")
        
        cell = self.quantum_cells[coords]
        gate = QuantumGates.get_gate(gate_type, **params)
        cell.apply_unitary(gate)
    
    def apply_gate_to_all(self, gate_type: GateType, **params):
        """Apply gate to all cells."""
        for coords in self.quantum_cells.keys():
            self.apply_gate(coords, gate_type, **params)
    
    def entangle_cells(self, cell1: Tuple[int, int, int],
                      cell2: Tuple[int, int, int],
                      bell_type: str = "phi_plus"):
        """
        Entangle two cells.
        
        Args:
            cell1: First cell coordinates
            cell2: Second cell coordinates
            bell_type: Type of Bell state
        """
        if not self.config.enable_entanglement:
            raise ValueError("Entanglement not enabled")
        
        if self.entanglement_manager:
            self.entanglement_manager.create_bell_pair(cell1, cell2, bell_type)
    
    def entangle_by_face_exposure(self, coords: Tuple[int, int, int]):
        """
        Entangle cell based on face exposure (Livnium-specific).
        
        Args:
            coords: Cell coordinates
        """
        if not self.config.enable_entanglement:
            raise ValueError("Entanglement not enabled")
        
        geometric_cell = self.core_system.get_cell(coords)
        if geometric_cell and self.entanglement_manager:
            self.entanglement_manager.entangle_by_face_exposure(
                coords, geometric_cell.face_exposure
            )
    
    def measure_cell(self, coords: Tuple[int, int, int],
                    collapse: bool = True):
        """
        Measure a quantum cell.
        
        Args:
            coords: Cell coordinates
            collapse: Whether to collapse state
            
        Returns:
            Measurement result
        """
        if not self.config.enable_measurement:
            raise ValueError("Measurement not enabled")
        
        if coords not in self.quantum_cells:
            raise ValueError(f"Cell not found: {coords}")
        
        cell = self.quantum_cells[coords]
        return self.measurement_engine.measure_cell(cell, collapse=collapse)
    
    def measure_all(self, collapse: bool = True):
        """Measure all cells."""
        if not self.config.enable_measurement:
            raise ValueError("Measurement not enabled")
        
        return self.measurement_engine.measure_all_cells(self.quantum_cells, collapse=collapse)
    
    def get_quantum_state_summary(self) -> Dict:
        """Get summary of quantum state."""
        summary = {
            'total_quantum_cells': len(self.quantum_cells),
            'features_enabled': {
                'superposition': self.config.enable_superposition,
                'quantum_gates': self.config.enable_quantum_gates,
                'entanglement': self.config.enable_entanglement,
                'measurement': self.config.enable_measurement,
                'geometry_coupling': self.config.enable_geometry_quantum_coupling,
            }
        }
        
        if self.entanglement_manager:
            summary['entanglement'] = self.entanglement_manager.get_entanglement_statistics()
        
        if self.measurement_engine:
            summary['measurements'] = self.measurement_engine.get_measurement_statistics()
        
        return summary

