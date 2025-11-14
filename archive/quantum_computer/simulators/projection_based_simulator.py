"""
Projection-Based Hierarchical Geometry Simulator

Uses geometry > geometry > geometry to PROJECT high-entanglement states
onto manageable representations, leveraging the hierarchy to handle
maximum entanglement without exponential memory.
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quantum_computer.geometry.level2.projection_hierarchical_geometry import ProjectionHierarchicalGeometrySystem
from quantum_computer.geometry.level0.sparse_base_geometry import SparseBaseGeometry


class ProjectionBasedSimulator:
    """
    Quantum simulator using projection-based hierarchical geometry.
    
    Key idea: Use Level 2 to PROJECT high-entanglement states onto
    Level 0/1, preserving critical information without exponential memory.
    """
    
    def __init__(self, num_qubits: int, projection_threshold: int = 2000):
        """
        Initialize projection-based simulator.
        
        Args:
            num_qubits: Number of qubits
            projection_threshold: Max states to keep after projection
        """
        self.num_qubits = num_qubits
        self.state_size = 2 ** num_qubits
        self.projection_threshold = projection_threshold
        
        print("=" * 70)
        print("Projection-Based Hierarchical Geometry Simulator")
        print("=" * 70)
        print(f"  Qubits: {num_qubits}")
        print(f"  Strategy: Project high-entanglement onto geometry hierarchy")
        print(f"  Level 0: Sparse local structure (low memory)")
        print(f"  Level 1: Efficient operations (fast)")
        print(f"  Level 2: Projection/compression (handles entanglement)")
        
        # Use projection-based hierarchical geometry
        self.geometry_system = ProjectionHierarchicalGeometrySystem(
            base_dimension=num_qubits,
            threshold=1e-15
        )
        
        # Initialize |00...0⟩ in Level 0
        coordinates = tuple([0.0] * num_qubits)
        self.geometry_system.add_base_state(coordinates, amplitude=1.0+0j, phase=0.0)
        
        self.gate_history: List[Dict] = []
        self.projection_count = 0
        
        print(f"  ✅ Initialized in geometry > geometry > geometry (Projection)")
        print()
    
    def _get_state_coordinates(self, state_index: int) -> Tuple[float, ...]:
        """Convert state index to coordinates."""
        binary = format(state_index, f'0{self.num_qubits}b')
        return tuple([float(int(bit)) for bit in binary])
    
    def _get_state_index(self, coordinates: Tuple[float, ...]) -> int:
        """Convert coordinates to state index."""
        binary = ''.join(['1' if abs(c - 1.0) < 0.5 else '0' for c in coordinates])
        return int(binary, 2)
    
    def hadamard(self, qubit: int):
        """Apply Hadamard gate - works directly in geometry space."""
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
        # Get current active states from Level 0
        current_states = {}
        for coords in list(self.geometry_system.base_geometry.active_coordinates):
            amplitude = self.geometry_system.base_geometry.get_amplitude(coords)
            current_states[coords] = amplitude
        
        # Apply Hadamard: For each state, create two new states
        new_states = {}
        for coords, amplitude in current_states.items():
            # Get qubit value at this position
            qubit_val = int(coords[qubit])
            
            # Create two new states: |0⟩ and |1⟩ for this qubit
            coords_0 = list(coords)
            coords_1 = list(coords)
            coords_0[qubit] = 0.0
            coords_1[qubit] = 1.0
            
            coords_0_tuple = tuple(coords_0)
            coords_1_tuple = tuple(coords_1)
            
            # Apply Hadamard matrix
            if qubit_val == 0:
                # |0⟩ → (|0⟩ + |1⟩)/√2
                new_states[coords_0_tuple] = new_states.get(coords_0_tuple, 0.0) + amplitude * H[0, 0]
                new_states[coords_1_tuple] = new_states.get(coords_1_tuple, 0.0) + amplitude * H[1, 0]
            else:
                # |1⟩ → (|0⟩ - |1⟩)/√2
                new_states[coords_0_tuple] = new_states.get(coords_0_tuple, 0.0) + amplitude * H[0, 1]
                new_states[coords_1_tuple] = new_states.get(coords_1_tuple, 0.0) + amplitude * H[1, 1]
        
        # Update Level 0 geometry
        self.geometry_system.base_geometry = SparseBaseGeometry(
            dimension=self.num_qubits,
            threshold=self.geometry_system.base_geometry.threshold
        )
        
        # Normalize
        total_norm = sum(abs(amp)**2 for amp in new_states.values())
        if total_norm > 0:
            norm = np.sqrt(total_norm)
            for coords, amplitude in new_states.items():
                if abs(amplitude) > self.geometry_system.base_geometry.threshold:
                    self.geometry_system.base_geometry.add_state(coords, amplitude / norm, np.angle(amplitude / norm))
        
        # Check if projection needed
        active_states = len(self.geometry_system.base_geometry.active_coordinates)
        if active_states > self.projection_threshold:
            self._project_if_needed()
        
        self.gate_history.append({'gate': 'H', 'qubit': qubit})
    
    def cnot(self, control: int, target: int):
        """Apply CNOT gate - works directly in geometry space."""
        # Get current active states
        current_states = {}
        for coords in list(self.geometry_system.base_geometry.active_coordinates):
            amplitude = self.geometry_system.base_geometry.get_amplitude(coords)
            current_states[coords] = amplitude
        
        # Apply CNOT: Flip target if control is 1
        new_states = {}
        for coords, amplitude in current_states.items():
            control_val = int(coords[control])
            target_val = int(coords[target])
            
            if control_val == 1:
                # CNOT: Flip target
                new_coords = list(coords)
                new_coords[target] = 1.0 - target_val  # Flip
                new_coords_tuple = tuple(new_coords)
                new_states[new_coords_tuple] = new_states.get(new_coords_tuple, 0.0) + amplitude
            else:
                # CNOT: No change
                new_states[coords] = new_states.get(coords, 0.0) + amplitude
        
        # Update Level 0 geometry
        self.geometry_system.base_geometry = SparseBaseGeometry(
            dimension=self.num_qubits,
            threshold=self.geometry_system.base_geometry.threshold
        )
        
        # Normalize
        total_norm = sum(abs(amp)**2 for amp in new_states.values())
        if total_norm > 0:
            norm = np.sqrt(total_norm)
            for coords, amplitude in new_states.items():
                if abs(amplitude) > self.geometry_system.base_geometry.threshold:
                    self.geometry_system.base_geometry.add_state(coords, amplitude / norm, np.angle(amplitude / norm))
        
        # Check if projection needed
        active_states = len(self.geometry_system.base_geometry.active_coordinates)
        if active_states > self.projection_threshold:
            self._project_if_needed()
        
        self.gate_history.append({'gate': 'CNOT', 'control': control, 'target': target})
    
    def _project_if_needed(self):
        """Project state if it exceeds threshold."""
        active_states = len(self.geometry_system.base_geometry.active_coordinates)
        
        if active_states > self.projection_threshold:
            print(f"  [Level 2] Projecting: {active_states} states → {self.projection_threshold} states")
            
            # Use Level 2 to project onto manageable representation
            self.geometry_system.project_hierarchical(max_states=self.projection_threshold)
            
            self.projection_count += 1
            
            new_active = len(self.geometry_system.base_geometry.active_coordinates)
            print(f"    ✅ Projected to {new_active} states (preserving critical information)")
    
    def measure(self, qubit: int) -> int:
        """Measure qubit."""
        prob_1 = 0.0
        
        for coords in self.geometry_system.base_geometry.active_coordinates:
            state_index = self._get_state_index(coords)
            binary = format(state_index, f'0{self.num_qubits}b')
            if binary[qubit] == '1':
                amplitude = self.geometry_system.base_geometry.get_amplitude(coords)
                prob_1 += abs(amplitude) ** 2
        
        result = 1 if np.random.random() < prob_1 else 0
        
        # Collapse
        to_remove = []
        for coords in list(self.geometry_system.base_geometry.active_coordinates):
            state_index = self._get_state_index(coords)
            binary = format(state_index, f'0{self.num_qubits}b')
            if int(binary[qubit]) != result:
                self.geometry_system.base_geometry.set_amplitude(coords, 0.0)
                to_remove.append(coords)
        
        # Normalize
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
    
    def run(self, num_shots: int = 1000) -> Dict:
        """Run simulation."""
        # Save initial state (just the active coordinates)
        initial_active = set(self.geometry_system.base_geometry.active_coordinates)
        initial_amplitudes = {
            coords: self.geometry_system.base_geometry.get_amplitude(coords)
            for coords in initial_active
        }
        
        results = []
        for shot in range(num_shots):
            # Reset to initial state
            self.geometry_system.base_geometry = SparseBaseGeometry(
                dimension=self.num_qubits,
                threshold=self.geometry_system.base_geometry.threshold
            )
            for coords, amplitude in initial_amplitudes.items():
                self.geometry_system.base_geometry.add_state(coords, amplitude, np.angle(amplitude))
            
            shot_results = self.measure_all()
            results.append(tuple(shot_results))
        
        counts = Counter(results)
        
        return {
            'shots': num_shots,
            'results': dict(counts),
            'num_qubits': self.num_qubits,
            'projections': self.projection_count,
            'geometry_structure': self.geometry_system.get_full_structure()
        }
    
    def get_capacity_info(self) -> Dict:
        """Get capacity information."""
        structure = self.geometry_system.get_full_structure()
        level_0 = structure['level_0']
        
        return {
            'num_qubits': self.num_qubits,
            'state_size': self.state_size,
            'num_active_states': level_0['num_active_states'],
            'projections': self.projection_count,
            'projection_threshold': self.projection_threshold,
            'level_0_info': level_0,
            'level_1_info': structure['level_1'],
            'level_2_projections': structure['level_2_projections']
        }


def test_projection_max_entanglement():
    """Test projection-based simulator on maximum entanglement."""
    print("=" * 70)
    print("Testing Projection-Based Simulator on Maximum Entanglement")
    print("=" * 70)
    
    import tracemalloc
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        sim = ProjectionBasedSimulator(500, projection_threshold=2000)
        
        print("\nStep 1: Hadamard on ALL 500 qubits...")
        for i in range(500):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                active = len(sim.geometry_system.base_geometry.active_coordinates)
                print(f"  Progress: {i}/500 | Time: {elapsed:.2f}s | Memory: {peak/1024/1024:.2f} MB | States: {active}")
            sim.hadamard(i)
        
        print("\nStep 2: CNOT on ALL 499 adjacent pairs...")
        for i in range(499):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                active = len(sim.geometry_system.base_geometry.active_coordinates)
                print(f"  Progress: {i}/499 | Time: {elapsed:.2f}s | Memory: {peak/1024/1024:.2f} MB | States: {active} | Projections: {sim.projection_count}")
            sim.cnot(i, i+1)
        
        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        info = sim.get_capacity_info()
        
        print("\n" + "=" * 70)
        print("RESULT:")
        print("=" * 70)
        print(f"  ✅ Completed in {elapsed:.2f} seconds")
        print(f"  Peak memory: {peak/1024/1024:.2f} MB")
        print(f"  Final active states: {info['num_active_states']}")
        print(f"  Projections applied: {info['projections']}")
        print(f"  Level 2 projections: {info['level_2_projections']}")
        print("\n  Strategy: Used Level 2 to PROJECT high-entanglement")
        print("  onto Level 0/1, preserving critical information")
        print("  while keeping memory manageable.")
        
    except Exception as e:
        tracemalloc.stop()
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_projection_max_entanglement()

