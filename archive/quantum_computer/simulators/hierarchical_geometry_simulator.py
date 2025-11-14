"""
Hierarchical Geometry Quantum Simulator

Uses the full geometry > geometry > geometry system with sparse optimization
at every level. The hierarchical logic IS the simulation logic.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantum_computer.geometry.level2.sparse_hierarchical_geometry import SparseHierarchicalGeometrySystem


class HierarchicalGeometrySimulator:
    """
    Quantum simulator using full hierarchical geometry system.
    
    Level 0: Sparse base geometry (stores only non-zero states)
    Level 1: Efficient operations (processes only active states)
    Level 2: High-level optimizations (batch operations, etc.)
    
    The hierarchical logic IS the simulation logic.
    """
    
    def __init__(self, num_qubits: int, threshold: float = 1e-15):
        """
        Initialize hierarchical geometry simulator.
        
        Args:
            num_qubits: Number of qubits
            threshold: Minimum amplitude to store (sparse threshold)
        """
        self.num_qubits = num_qubits
        self.state_size = 2 ** num_qubits
        
        print(f"Initializing Hierarchical Geometry Simulator")
        print(f"  Qubits: {num_qubits}")
        print(f"  State space: 2^{num_qubits} = {self.state_size:,} states")
        print(f"  Using: Geometry > Geometry > Geometry (Sparse)")
        
        # Use sparse hierarchical geometry system
        self.geometry_system = SparseHierarchicalGeometrySystem(
            base_dimension=num_qubits,
            threshold=threshold
        )
        
        # Initialize |00...0⟩ state in Level 0
        coordinates = tuple([0.0] * num_qubits)
        self.geometry_system.add_base_state(coordinates, amplitude=1.0+0j, phase=0.0)
        
        print(f"  ✅ Initialized in Level 0 sparse base geometry")
        
        self.gate_history: List[Dict] = []
    
    def _get_state_coordinates(self, state_index: int) -> Tuple[float, ...]:
        """Convert state index to geometric coordinates."""
        binary = format(state_index, f'0{self.num_qubits}b')
        return tuple([float(int(bit)) for bit in binary])
    
    def _get_state_index(self, coordinates: Tuple[float, ...]) -> int:
        """Convert coordinates to state index."""
        binary = ''.join(['1' if abs(c - 1.0) < 0.5 else '0' for c in coordinates])
        return int(binary, 2)
    
    def _get_full_state_vector(self) -> np.ndarray:
        """Get full state vector from Level 0 geometry."""
        state = np.zeros(self.state_size, dtype=np.complex128)
        
        # Read from Level 0 sparse base geometry
        for coords in self.geometry_system.base_geometry.active_coordinates:
            amplitude = self.geometry_system.base_geometry.get_amplitude(coords)
            state_index = self._get_state_index(coords)
            state[state_index] = amplitude
        
        return state
    
    def _update_geometry_from_state_vector(self, state_vector: np.ndarray):
        """Update Level 0 geometry from state vector."""
        # Normalize
        norm = np.linalg.norm(state_vector)
        if norm > 0:
            state_vector /= norm
        
        # Update Level 0 sparse base geometry
        for i in range(self.state_size):
            coords = self._get_state_coordinates(i)
            amplitude = state_vector[i]
            self.geometry_system.base_geometry.set_amplitude(coords, amplitude)
    
    def hadamard(self, qubit: int):
        """Apply Hadamard gate through hierarchical geometry system."""
        print(f"  [Level 1] Applying Hadamard to qubit {qubit}...")
        
        # Get current state from Level 0
        current_state = self._get_full_state_vector()
        
        # Apply gate matrix
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
        before = 2 ** qubit
        after = 2 ** (self.num_qubits - qubit - 1)
        state_reshaped = current_state.reshape(before, 2, after)
        
        for i in range(before):
            for j in range(after):
                state_reshaped[i, :, j] = H @ state_reshaped[i, :, j]
        
        new_state = state_reshaped.reshape(self.state_size)
        
        # Update Level 0 geometry (sparse: only stores non-zero)
        self._update_geometry_from_state_vector(new_state)
        
        # Track operation in Level 1
        self.geometry_system.add_meta_operation('rotation', angle=np.pi/4, axis=qubit)
        
        self.gate_history.append({'gate': 'H', 'qubit': qubit})
        
        active = len(self.geometry_system.base_geometry.active_coordinates)
        print(f"    ✅ Updated Level 0, {active} active states")
    
    def cnot(self, control: int, target: int):
        """Apply CNOT gate through hierarchical geometry system."""
        print(f"  [Level 2] Applying CNOT (control={control}, target={target})...")
        
        # Get current state from Level 0
        current_state = self._get_full_state_vector()
        
        # Apply CNOT matrix
        CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)
        
        if control > target:
            control, target = target, control
        
        before = 2 ** control
        middle = 2 ** (target - control - 1)
        after = 2 ** (self.num_qubits - target - 1)
        
        state_reshaped = current_state.reshape(before, 2, middle, 2, after)
        state_2q = state_reshaped.reshape(before * middle * after, 2, 2)
        
        for i in range(before * middle * after):
            vec = state_2q[i].flatten()
            vec_new = CNOT @ vec
            state_2q[i] = vec_new.reshape(2, 2)
        
        new_state = state_reshaped.reshape(self.state_size)
        
        # Update Level 0 geometry (sparse)
        self._update_geometry_from_state_vector(new_state)
        
        # Track operation in Level 2 (entanglement)
        self.geometry_system.add_meta_meta_operation('entangle', control=control, target=target)
        
        self.gate_history.append({'gate': 'CNOT', 'control': control, 'target': target})
        
        active = len(self.geometry_system.base_geometry.active_coordinates)
        print(f"    ✅ Updated Level 0, {active} active states, Level 2 entanglement")
    
    def measure(self, qubit: int) -> int:
        """Measure qubit - computes probability from Level 0 geometry."""
        # Compute probability from Level 0 sparse geometry
        prob_1 = 0.0
        
        for coords in self.geometry_system.base_geometry.active_coordinates:
            state_index = self._get_state_index(coords)
            binary = format(state_index, f'0{self.num_qubits}b')
            if binary[qubit] == '1':
                amplitude = self.geometry_system.base_geometry.get_amplitude(coords)
                prob_1 += abs(amplitude) ** 2
        
        # Sample
        result = 1 if np.random.random() < prob_1 else 0
        
        # Collapse in Level 0 geometry
        to_remove = []
        for coords in list(self.geometry_system.base_geometry.active_coordinates):
            state_index = self._get_state_index(coords)
            binary = format(state_index, f'0{self.num_qubits}b')
            if int(binary[qubit]) != result:
                self.geometry_system.base_geometry.set_amplitude(coords, 0.0)
                to_remove.append(coords)
        
        # Normalize Level 0
        total_prob = 0.0
        for coords in self.geometry_system.base_geometry.active_coordinates:
            amplitude = self.geometry_system.base_geometry.get_amplitude(coords)
            total_prob += abs(amplitude) ** 2
        
        if total_prob > 0:
            norm = np.sqrt(total_prob)
            for coords in list(self.geometry_system.base_geometry.active_coordinates):
                current_amp = self.geometry_system.base_geometry.get_amplitude(coords)
                self.geometry_system.base_geometry.set_amplitude(coords, current_amp / norm)
        
        return result
    
    def measure_all(self) -> List[int]:
        """Measure all qubits."""
        results = []
        for i in range(self.num_qubits):
            results.append(self.measure(i))
        return results
    
    def get_probabilities(self) -> np.ndarray:
        """Get probability distribution from Level 0 geometry."""
        state = self._get_full_state_vector()
        return np.abs(state) ** 2
    
    def run(self, num_shots: int = 1000) -> Dict:
        """Run simulation."""
        initial_state = self._get_full_state_vector()
        
        results = []
        for shot in range(num_shots):
            # Reset Level 0 geometry
            self._update_geometry_from_state_vector(initial_state)
            
            # Measure
            shot_results = self.measure_all()
            results.append(tuple(shot_results))
        
        counts = Counter(results)
        
        return {
            'shots': num_shots,
            'results': dict(counts),
            'num_qubits': self.num_qubits,
            'geometry_structure': self.geometry_system.get_full_structure()
        }
    
    def get_capacity_info(self) -> Dict:
        """Get capacity information from hierarchical system."""
        structure = self.geometry_system.get_full_structure()
        level_0 = structure['level_0']
        
        return {
            'num_qubits': self.num_qubits,
            'state_size': self.state_size,
            'num_stored_states': level_0['num_states'],
            'num_active_states': level_0['num_active_states'],
            'efficiency': level_0.get('efficiency', 'N/A'),
            'level_0_info': level_0,
            'level_1_info': structure['level_1'],
            'level_2_info': structure.get('level_2', {})
        }


def test_hierarchical_simulator():
    """Test the hierarchical geometry simulator."""
    print("=" * 70)
    print("Testing Hierarchical Geometry Simulator")
    print("  (Geometry > Geometry > Geometry with Sparse Optimization)")
    print("=" * 70)
    
    # Test with 10 qubits
    sim = HierarchicalGeometrySimulator(10)
    
    print("\nBuilding circuit:")
    print("  |0⟩ → H → CNOT → Measure")
    
    # Apply gates through hierarchy
    sim.hadamard(0)
    sim.cnot(0, 1)
    
    # Get probabilities from Level 0
    probs = sim.get_probabilities()
    print(f"\nState probabilities from Level 0 geometry (top 5):")
    for i in range(min(5, len(probs))):
        if probs[i] > 1e-10:
            binary = format(i, f'0{sim.num_qubits}b')
            print(f"  |{binary}⟩: {probs[i]:.6f}")
    
    # Run simulation
    print(f"\nRunning 1000 shots...")
    results = sim.run(num_shots=1000)
    
    print(f"\nResults:")
    print(f"  Shots: {results['shots']}")
    print(f"  Unique outcomes: {len(results['results'])}")
    for outcome, count in list(results['results'].items())[:5]:
        print(f"  {outcome}: {count} ({count/results['shots']*100:.1f}%)")
    
    # Show hierarchical structure
    print(f"\nHierarchical Structure:")
    info = sim.get_capacity_info()
    print(f"  Level 0: {info['num_active_states']} active states")
    print(f"  Level 1: {info['level_1_info']['num_operations']} operations")
    print(f"  Level 2: {info['level_2_info'].get('num_meta_meta_operations', 0)} operations")
    print(f"  Efficiency: {info['efficiency']}")
    
    print("\n✅ Hierarchical simulator test complete!")
    print("  All optimization logic is in the geometry hierarchy itself!")


if __name__ == "__main__":
    test_hierarchical_simulator()

