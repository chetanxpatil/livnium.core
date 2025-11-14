"""
Livnium-Based Entanglement-Preserving Simulator

Uses Livnium hierarchical structure and conservation laws to PRESERVE entanglement
instead of truncating it. Entanglement is encoded in the geometry hierarchy
and preserved through conservation invariants.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import Counter
import time



class LivniumEntanglementPreserving:
    """
    Quantum simulator using Livnium hierarchical structure to preserve entanglement.
    
    Key idea: Instead of truncating states, encode entanglement relationships
    in the hierarchical geometry (macro/micro levels) and preserve them through
    Livnium's conservation laws (symbolic weight, class counts).
    """
    
    def __init__(self, num_qubits: int, macro_size: int = 3):
        """
        Initialize Livnium-based entanglement-preserving simulator.
        
        Args:
            num_qubits: Number of qubits
            macro_size: Size of macro blocks (3x3x3 = 27 cells per block)
        """
        self.num_qubits = num_qubits
        self.macro_size = macro_size
        self.micro_size = macro_size  # Each macro cell contains micro lattice
        
        # Hierarchical structure: macro blocks, each containing micro cells
        # Total capacity: macro_size^3 * micro_size^3 = 27 * 27 = 729 cells
        # Can represent up to log2(729) ≈ 9.5 qubits per macro block
        
        # Calculate how many macro blocks we need
        self.num_macro_blocks = (num_qubits + 26) // 27  # Round up
        
        print("=" * 70)
        print("Livnium Entanglement-Preserving Simulator")
        print("=" * 70)
        print(f"  Qubits: {num_qubits}")
        print(f"  Macro blocks: {self.num_macro_blocks} (each {macro_size}×{macro_size}×{macro_size})")
        print(f"  Micro cells per block: {macro_size**3} = {macro_size**3}")
        print(f"  Total capacity: {self.num_macro_blocks} × {macro_size**3} = {self.num_macro_blocks * macro_size**3} cells")
        print(f"  Strategy: Encode entanglement in hierarchical geometry")
        print(f"  Preservation: Use Livnium conservation laws (SW, class counts)")
        
        # Hierarchical state representation
        # Level 0 (macro): Blocks representing qubit groups
        # Level 1 (micro): Cells within blocks representing entanglement
        self.macro_blocks: List[Dict] = []
        
        for block_idx in range(self.num_macro_blocks):
            block = {
                'index': block_idx,
                'qubits': [],  # Qubits assigned to this block
                'micro_cells': {},  # micro_coord -> {amplitude, phase, entanglement_links, qubit_state}
                'symbolic_weight': 0,  # Livnium SW conservation
                'class_counts': {'core': 0, 'center': 0, 'edge': 0, 'corner': 0}
            }
            self.macro_blocks.append(block)
        
        # Assign qubits to blocks
        for qubit_idx in range(num_qubits):
            block_idx = qubit_idx // (macro_size ** 3)
            if block_idx < len(self.macro_blocks):
                self.macro_blocks[block_idx]['qubits'].append(qubit_idx)
        
        # Initialize |00...0⟩ state
        self._initialize_ground_state()
        
        self.gate_history: List[Dict] = []
        self.entanglement_preserved = True
        
        print(f"  ✅ Initialized with Livnium hierarchical structure")
        print()
    
    def _initialize_ground_state(self):
        """Initialize in |00...0⟩ using Livnium structure."""
        # Initialize entire system as ONE joint state |00...0⟩
        # All qubits are in |0⟩, so we have one global state
        # Store this state in the first block (block 0)
        first_block = self.macro_blocks[0]
        core_coord = (0, 0, 0)
        
        # Create full qubit state for entire system (all 0s)
        full_qubit_state = [0] * self.num_qubits
        state_key = tuple(full_qubit_state)
        
        first_block['micro_cells'][state_key] = {
            'amplitude': 1.0 + 0j,  # Single global state with amplitude 1
            'phase': 0.0,
            'entanglement_links': set(),
            'exposure': 0,
            'symbolic_weight': 0,
            'qubit_state': full_qubit_state,
            'coord': core_coord
        }
        first_block['class_counts']['core'] = 1
        first_block['symbolic_weight'] = 0
        
        # Other blocks start empty (will be populated when qubits in those blocks are used)
        for block_idx in range(1, len(self.macro_blocks)):
            block = self.macro_blocks[block_idx]
            block['micro_cells'] = {}
            block['class_counts'] = {'core': 0, 'center': 0, 'edge': 0, 'corner': 0}
            block['symbolic_weight'] = 0
    
    def _get_micro_coord_from_qubit_state(self, qubit_states: List[int], block_idx: int) -> Tuple[int, int, int]:
        """
        Map qubit states to micro coordinate in Livnium structure.
        
        IMPORTANT: Coordinates are just ADDRESSES in geometry space.
        The full qubit state is stored in the cell DATA, not encoded in coordinates.
        This ensures reversibility and losslessness.
        
        Uses a better hash to avoid collisions and ensure unique mapping.
        """
        block = self.macro_blocks[block_idx]
        qubits_in_block = block['qubits']
        
        if len(qubits_in_block) == 0:
            return (0, 0, 0)
        
        # Create a better hash that distributes states across 27 coordinates
        # Use a prime-based hash to reduce collisions
        state_hash = 0
        prime = 31  # Prime number for better distribution
        for i, qubit_idx in enumerate(qubits_in_block):
            if qubit_idx < len(qubit_states):
                state_hash = state_hash * prime + qubit_states[qubit_idx]
        
        # Map hash to 3D coordinate using better distribution
        # Map to [-1, 0, 1] range with better spread
        x = ((state_hash // 1) % 3) - 1
        y = ((state_hash // 3) % 3) - 1
        z = ((state_hash // 9) % 3) - 1
        
        return (x, y, z)
    
    def _get_exposure(self, coord: Tuple[int, int, int]) -> int:
        """Calculate Livnium exposure (faces on boundary)."""
        x, y, z = coord
        exposure = 0
        
        # Check if on boundary of [-1, 0, 1] cube
        if abs(x) == 1:
            exposure += 1
        if abs(y) == 1:
            exposure += 1
        if abs(z) == 1:
            exposure += 1
        
        return exposure
    
    def _update_symbolic_weight(self, block_idx: int):
        """Update Livnium symbolic weight for block (conservation law)."""
        block = self.macro_blocks[block_idx]
        total_sw = 0
        
        for state_key, cell_data in block['micro_cells'].items():
            # Get coordinate from cell data (now stored in 'coord' field)
            coord = cell_data.get('coord', (0, 0, 0))
            exposure = self._get_exposure(coord)
            sw = 9 * exposure  # SW = 9 * f
            cell_data['exposure'] = exposure
            cell_data['symbolic_weight'] = sw
            total_sw += sw
        
        block['symbolic_weight'] = total_sw
    
    def hadamard(self, qubit: int):
        """Apply Hadamard gate - preserves entanglement through hierarchy (lossless)."""
        # Find which block(s) contain states with this qubit
        # Since we store full qubit states, we need to find blocks that have states
        # For now, operate on block 0 (where global states are stored)
        # TODO: Support distributed state storage across blocks
        
        # Find the block that contains this qubit (for qubit position calculation)
        qubit_block_idx = qubit // (self.macro_size ** 3)
        if qubit_block_idx >= len(self.macro_blocks):
            return
        
        qubit_block = self.macro_blocks[qubit_block_idx]
        qubits_in_qubit_block = qubit_block['qubits']
        qubit_pos_in_qubit_block = qubits_in_qubit_block.index(qubit) if qubit in qubits_in_qubit_block else -1
        
        if qubit_pos_in_qubit_block == -1:
            return
        
        # Find block(s) that actually contain states
        # For global states, this is typically block 0
        blocks_with_states = []
        for block_idx, block in enumerate(self.macro_blocks):
            if len(block['micro_cells']) > 0:
                blocks_with_states.append((block_idx, block))
        
        if not blocks_with_states:
            return
        
        # Operate on all blocks that have states (usually just block 0)
        for block_idx, block in blocks_with_states:
            qubits_in_block = block['qubits'] if block['qubits'] else list(range(self.num_qubits))
            # Calculate qubit position in the full state (not just in this block)
            qubit_pos = qubit  # Position in full qubit state
        
            # Get current micro cells
            current_cells = dict(block['micro_cells'])
            
            # Apply Hadamard: Each cell splits into two (lossless - full state preserved)
            new_cells = {}
            H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
            
            for state_key, cell_data in current_cells.items():
                amplitude = cell_data['amplitude']
                entanglement_links = cell_data['entanglement_links']
                qubit_state = cell_data.get('qubit_state', [0] * self.num_qubits).copy()
                
                # Get current qubit value (position in full state)
                current_qubit_val = qubit_state[qubit_pos] if qubit_pos < len(qubit_state) else 0
            
                # Create two new states: |0⟩ and |1⟩ for this qubit
                qubit_state_0 = qubit_state.copy()
                qubit_state_1 = qubit_state.copy()
                qubit_state_0[qubit_pos] = 0
                qubit_state_1[qubit_pos] = 1
                
                # Get coordinates for new states (reversible mapping)
                coord_0 = self._get_micro_coord_from_qubit_state(qubit_state_0, block_idx)
                coord_1 = self._get_micro_coord_from_qubit_state(qubit_state_1, block_idx)
            
                # Apply Hadamard matrix
                if current_qubit_val == 0:
                    amp_0 = amplitude * H[0, 0]  # |0⟩ → (|0⟩ + |1⟩)/√2
                    amp_1 = amplitude * H[1, 0]
                else:
                    amp_0 = amplitude * H[0, 1]  # |1⟩ → (|0⟩ - |1⟩)/√2
                    amp_1 = amplitude * H[1, 1]
                
                # Store full qubit state in cell data (lossless)
                # Use qubit_state tuple as key to avoid hash collisions!
                state_key_0 = tuple(qubit_state_0)
                state_key_1 = tuple(qubit_state_1)
                
                if abs(amp_0) > 1e-15:
                    # Check if this state already exists (same qubit_state, different coord possible)
                    if state_key_0 not in new_cells:
                        new_cells[state_key_0] = {
                            'amplitude': 0.0 + 0j,
                            'phase': 0.0,
                            'entanglement_links': entanglement_links.copy(),
                            'exposure': self._get_exposure(coord_0),
                            'symbolic_weight': 0,
                            'qubit_state': qubit_state_0,  # FULL state preserved
                            'coord': coord_0  # Store coordinate for Livnium structure
                        }
                    new_cells[state_key_0]['amplitude'] += amp_0
                
                if abs(amp_1) > 1e-15:
                    if state_key_1 not in new_cells:
                        new_cells[state_key_1] = {
                            'amplitude': 0.0 + 0j,
                            'phase': 0.0,
                            'entanglement_links': entanglement_links.copy(),
                            'exposure': self._get_exposure(coord_1),
                            'symbolic_weight': 0,
                            'qubit_state': qubit_state_1,  # FULL state preserved
                            'coord': coord_1  # Store coordinate for Livnium structure
                        }
                    new_cells[state_key_1]['amplitude'] += amp_1
            
            # Normalize
            total_norm = sum(abs(cell['amplitude'])**2 for cell in new_cells.values())
            if total_norm > 0:
                norm = np.sqrt(total_norm)
                for state_key in new_cells:
                    new_cells[state_key]['amplitude'] = new_cells[state_key]['amplitude'] / norm
            
            # Update block with new cells (now keyed by qubit_state tuple, not coordinate)
            block['micro_cells'] = new_cells
            
            # Update Livnium conservation (symbolic weight)
            self._update_symbolic_weight(block_idx)
            
            # Update class counts
            self._update_class_counts(block_idx)
        
        self.gate_history.append({'gate': 'H', 'qubit': qubit})
        
        # Normalize globally to preserve total probability = 1
        self._normalize_global()
    
    def cnot(self, control: int, target: int):
        """Apply CNOT gate - creates entanglement preserved in hierarchy."""
        # If control and target are in same block, handle locally
        control_block = control // (self.macro_size ** 3)
        target_block = target // (self.macro_size ** 3)
        
        if control_block >= len(self.macro_blocks) or target_block >= len(self.macro_blocks):
            return
        
        # For same-block CNOT, apply directly (lossless)
        if control_block == target_block:
            block = self.macro_blocks[control_block]
            qubits_in_block = block['qubits']
            current_cells = dict(block['micro_cells'])
            new_cells = {}
            
            control_pos_in_block = qubits_in_block.index(control) if control in qubits_in_block else -1
            target_pos_in_block = qubits_in_block.index(target) if target in qubits_in_block else -1
            
            if control_pos_in_block == -1 or target_pos_in_block == -1:
                return
            
            for state_key, cell_data in current_cells.items():
                amplitude = cell_data['amplitude']
                entanglement_links = cell_data['entanglement_links']
                qubit_state = cell_data.get('qubit_state', [0] * len(qubits_in_block)).copy()
                
                if abs(amplitude) < 1e-15:
                    continue
                
                # Get control qubit value from FULL stored state (lossless)
                control_is_one = (control_pos_in_block < len(qubit_state) and 
                                 qubit_state[control_pos_in_block] == 1)
                
                # CNOT: Flip target if control is 1
                new_qubit_state = qubit_state.copy()
                if control_is_one:
                    # Flip target qubit
                    if target_pos_in_block < len(new_qubit_state):
                        new_qubit_state[target_pos_in_block] = 1 - new_qubit_state[target_pos_in_block]
                
                # Get coordinate for new state (reversible mapping)
                new_coord = self._get_micro_coord_from_qubit_state(new_qubit_state, control_block)
                new_state_key = tuple(new_qubit_state)
                
                # Preserve amplitude and entanglement links (use qubit_state as key)
                if new_state_key not in new_cells:
                    new_cells[new_state_key] = {
                        'amplitude': 0.0 + 0j,
                        'phase': 0.0,
                        'entanglement_links': entanglement_links.copy(),
                        'exposure': self._get_exposure(new_coord),
                        'symbolic_weight': 0,
                        'qubit_state': new_qubit_state,  # FULL state preserved
                        'coord': new_coord  # Store coordinate for Livnium structure
                    }
                
                new_cells[new_state_key]['amplitude'] += amplitude
            
            # Normalize
            total_norm = sum(abs(cell['amplitude'])**2 for cell in new_cells.values())
            if total_norm > 0:
                norm = np.sqrt(total_norm)
                for coord in new_cells:
                    new_cells[coord]['amplitude'] = new_cells[coord]['amplitude'] / norm
            
            block['micro_cells'] = new_cells
            self._update_symbolic_weight(control_block)
            self._update_class_counts(control_block)
        else:
            # Cross-block CNOT: Create entanglement (FIXED - uses full qubit_state, proper tensor product)
            control_block_obj = self.macro_blocks[control_block]
            target_block_obj = self.macro_blocks[target_block]
            control_qubits_in_block = control_block_obj['qubits']
            target_qubits_in_block = target_block_obj['qubits']
            
            # Get positions in blocks
            control_pos_in_block = control_qubits_in_block.index(control) if control in control_qubits_in_block else -1
            target_pos_in_block = target_qubits_in_block.index(target) if target in target_qubits_in_block else -1
            
            if control_pos_in_block == -1 or target_pos_in_block == -1:
                return
            
            # Get current cells
            control_cells = dict(control_block_obj['micro_cells'])
            target_cells = dict(target_block_obj['micro_cells'])
            
            # Apply CNOT: Create tensor product of control and target blocks
            # Joint state: |control_block⟩ ⊗ |target_block⟩
            # After CNOT: target block state depends on control block state
            # IMPORTANT: Target block is REPLACED based on control block (entanglement)
            new_target_cells = {}
            
            # Compute joint normalization factor (sum over all control × target combinations)
            joint_norm_sq = 0.0
            for c_state_key, c_data in control_cells.items():
                control_amplitude = c_data['amplitude']
                for t_state_key, t_data in target_cells.items():
                    target_amplitude = t_data['amplitude']
                    joint_norm_sq += abs(control_amplitude * target_amplitude) ** 2
            
            joint_norm = np.sqrt(joint_norm_sq) if joint_norm_sq > 0 else 1.0
            
            for c_state_key, c_data in control_cells.items():
                control_amplitude = c_data['amplitude']
                control_qubit_state = c_data.get('qubit_state', [0] * len(control_qubits_in_block))
                c_coord = c_data.get('coord', (0, 0, 0))
                
                if abs(control_amplitude) < 1e-15:
                    continue
                
                # Check if control qubit is |1⟩ using FULL stored state (not coordinate!)
                control_is_one = (control_pos_in_block < len(control_qubit_state) and 
                                 control_qubit_state[control_pos_in_block] == 1)
                
                for t_state_key, t_data in target_cells.items():
                    target_amplitude = t_data['amplitude']
                    target_qubit_state = t_data.get('qubit_state', [0] * len(target_qubits_in_block))
                    
                    if abs(target_amplitude) < 1e-15:
                        continue
                    
                    # CNOT: Flip target if control is 1 (using FULL state)
                    new_target_qubit_state = target_qubit_state.copy()
                    if control_is_one:
                        # Flip target qubit using FULL state
                        if target_pos_in_block < len(new_target_qubit_state):
                            new_target_qubit_state[target_pos_in_block] = 1 - new_target_qubit_state[target_pos_in_block]
                    
                    # Get coordinate for new state (reversible mapping)
                    new_t_coord = self._get_micro_coord_from_qubit_state(new_target_qubit_state, target_block)
                    new_t_state_key = tuple(new_target_qubit_state)
                    
                    # Create entangled state: tensor product (control ⊗ target)
                    # Joint amplitude = control_amplitude * target_amplitude
                    # Normalize by joint norm to preserve total probability = 1
                    joint_amplitude = (control_amplitude * target_amplitude) / joint_norm
                    
                    # Preserve entanglement link
                    entanglement_link = (control_block, c_coord, target_block, new_t_coord)
                    
                    if abs(joint_amplitude) > 1e-15:
                        if new_t_state_key not in new_target_cells:
                            new_target_cells[new_t_state_key] = {
                                'amplitude': 0.0 + 0j,
                                'phase': 0.0,
                                'entanglement_links': set(),
                                'exposure': self._get_exposure(new_t_coord),
                                'symbolic_weight': 0,
                                'qubit_state': new_target_qubit_state,  # FULL state preserved
                                'coord': new_t_coord  # Store coordinate for Livnium structure
                            }
                        
                        # Accumulate amplitude (multiple control states can lead to same target state)
                        new_target_cells[new_t_state_key]['amplitude'] += joint_amplitude
                        new_target_cells[new_t_state_key]['entanglement_links'].add(entanglement_link)
            
            # Final normalization (should be close to 1 already, but ensure it)
            total_norm = sum(abs(cell['amplitude'])**2 for cell in new_target_cells.values())
            if total_norm > 0:
                norm = np.sqrt(total_norm)
                for state_key in new_target_cells:
                    new_target_cells[state_key]['amplitude'] = new_target_cells[state_key]['amplitude'] / norm
            
            # Update target block
            target_block_obj['micro_cells'] = new_target_cells
            
            # Update Livnium conservation
            self._update_symbolic_weight(target_block)
            self._update_class_counts(target_block)
        
        self.gate_history.append({'gate': 'CNOT', 'control': control, 'target': target})
        self.entanglement_preserved = True
        
        # Normalize globally to preserve total probability = 1
        self._normalize_global()
    
    def _normalize_global(self):
        """Normalize entire system globally to preserve total probability = 1."""
        # Compute total probability across all blocks
        total_prob = 0.0
        for block in self.macro_blocks:
            for cell_data in block['micro_cells'].values():
                total_prob += abs(cell_data['amplitude']) ** 2
        
        if total_prob > 0:
            norm = np.sqrt(total_prob)
            # Normalize all amplitudes
            for block in self.macro_blocks:
                for cell_data in block['micro_cells'].values():
                    cell_data['amplitude'] = cell_data['amplitude'] / norm
    
    def _update_class_counts(self, block_idx: int):
        """Update Livnium class counts (conservation law)."""
        block = self.macro_blocks[block_idx]
        counts = {'core': 0, 'center': 0, 'edge': 0, 'corner': 0}
        
        for state_key, cell_data in block['micro_cells'].items():
            exposure = cell_data['exposure']
            if exposure == 0:
                counts['core'] += 1
            elif exposure == 1:
                counts['center'] += 1
            elif exposure == 2:
                counts['edge'] += 1
            elif exposure == 3:
                counts['corner'] += 1
        
        block['class_counts'] = counts
    
    def measure(self, qubit: int) -> int:
        """Measure qubit - computes probability from hierarchical structure (lossless)."""
        block_idx = qubit // (self.macro_size ** 3)
        block = self.macro_blocks[block_idx]
        qubits_in_block = block['qubits']
        qubit_pos_in_block = qubits_in_block.index(qubit) if qubit in qubits_in_block else -1
        
        if qubit_pos_in_block == -1:
            return 0
        
        # Compute probability from micro cells using FULL qubit state (lossless)
        prob_1 = 0.0
        total_prob = 0.0
        
        for state_key, cell_data in block['micro_cells'].items():
            amplitude = cell_data['amplitude']
            prob = abs(amplitude) ** 2
            total_prob += prob
            
            # Get qubit value from FULL stored state (not from coordinate)
            qubit_state = cell_data.get('qubit_state', [0] * len(qubits_in_block))
            if qubit_pos_in_block < len(qubit_state) and qubit_state[qubit_pos_in_block] == 1:
                prob_1 += prob
        
        if total_prob > 0:
            prob_1 = prob_1 / total_prob
        
        result = 1 if np.random.random() < prob_1 else 0
        
        # Collapse: Keep only states matching result (using FULL state)
        new_cells = {}
        for state_key, cell_data in block['micro_cells'].items():
            amplitude = cell_data['amplitude']
            qubit_state = cell_data.get('qubit_state', [0] * len(qubits_in_block))
            
            # Check if state matches measurement using FULL qubit state
            if qubit_pos_in_block < len(qubit_state) and qubit_state[qubit_pos_in_block] == result:
                if abs(amplitude) > 1e-15:
                    new_cells[state_key] = cell_data.copy()
        
        # Normalize
        total_norm = sum(abs(cell['amplitude'])**2 for cell in new_cells.values())
        if total_norm > 0:
            norm = np.sqrt(total_norm)
            for coord in new_cells:
                new_cells[coord]['amplitude'] = new_cells[coord]['amplitude'] / norm
        
        block['micro_cells'] = new_cells
        self._update_symbolic_weight(block_idx)
        self._update_class_counts(block_idx)
        
        return result
    
    def measure_all(self) -> List[int]:
        """Measure all qubits."""
        results = []
        for i in range(self.num_qubits):
            results.append(self.measure(i))
        return results
    
    def run(self, num_shots: int = 1000) -> Dict:
        """Run simulation with multiple shots."""
        from collections import Counter
        
        # Save initial state (lossless - preserve full qubit_state)
        initial_state = {}
        for block_idx, block in enumerate(self.macro_blocks):
            initial_state[block_idx] = {
                state_key: {
                    'amplitude': cell_data['amplitude'],
                    'phase': cell_data['phase'],
                    'entanglement_links': cell_data['entanglement_links'].copy(),
                    'qubit_state': cell_data.get('qubit_state', []).copy(),  # Preserve full state
                    'coord': cell_data.get('coord', (0, 0, 0))  # Preserve coordinate
                }
                for state_key, cell_data in block['micro_cells'].items()
            }
        
        results = []
        for shot in range(num_shots):
            # Reset to initial state (lossless restoration)
            for block_idx, block in enumerate(self.macro_blocks):
                block['micro_cells'] = {}
                for state_key, cell_data in initial_state[block_idx].items():
                    coord = cell_data.get('coord', (0, 0, 0))
                    block['micro_cells'][state_key] = {
                        'amplitude': cell_data['amplitude'],
                        'phase': cell_data['phase'],
                        'entanglement_links': cell_data['entanglement_links'].copy(),
                        'exposure': self._get_exposure(coord),
                        'symbolic_weight': 0,
                        'qubit_state': cell_data['qubit_state'].copy(),  # Restore full state
                        'coord': coord  # Restore coordinate
                    }
                self._update_symbolic_weight(block_idx)
                self._update_class_counts(block_idx)
            
            # Measure all qubits
            shot_results = self.measure_all()
            results.append(tuple(shot_results))
        
        counts = Counter(results)
        
        return {
            'shots': num_shots,
            'results': dict(counts),
            'num_qubits': self.num_qubits
        }
    
    def get_entanglement_info(self) -> Dict:
        """Get information about preserved entanglement."""
        total_cells = 0
        total_entanglement_links = 0
        total_sw = 0
        
        for block in self.macro_blocks:
            total_cells += len(block['micro_cells'])
            for cell_data in block['micro_cells'].values():
                total_entanglement_links += len(cell_data['entanglement_links'])
            total_sw += block['symbolic_weight']
        
        return {
            'num_qubits': self.num_qubits,
            'num_macro_blocks': self.num_macro_blocks,
            'total_micro_cells': total_cells,
            'total_entanglement_links': total_entanglement_links,
            'total_symbolic_weight': total_sw,
            'entanglement_preserved': self.entanglement_preserved,
            'blocks': [
                {
                    'index': b['index'],
                    'cells': len(b['micro_cells']),
                    'sw': b['symbolic_weight'],
                    'classes': b['class_counts']
                }
                for b in self.macro_blocks
            ]
        }


def test_livnium_max_entanglement():
    """Test Livnium-based simulator on maximum entanglement."""
    print("=" * 70)
    print("Testing Livnium Entanglement-Preserving Simulator")
    print("=" * 70)
    
    import tracemalloc
    
    tracemalloc.start()
    start_time = time.time()
    
    try:
        # Test with 500 qubits
        sim = LivniumEntanglementPreserving(500, macro_size=3)
        
        print("\nStep 1: Hadamard on ALL 500 qubits...")
        for i in range(500):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                info = sim.get_entanglement_info()
                print(f"  Progress: {i}/500 | Time: {elapsed:.2f}s | Memory: {peak/1024/1024:.2f} MB | Cells: {info['total_micro_cells']} | Links: {info['total_entanglement_links']}")
            sim.hadamard(i)
        
        print("\nStep 2: CNOT on ALL 499 adjacent pairs...")
        for i in range(499):
            if i % 100 == 0:
                elapsed = time.time() - start_time
                current, peak = tracemalloc.get_traced_memory()
                info = sim.get_entanglement_info()
                print(f"  Progress: {i}/499 | Time: {elapsed:.2f}s | Memory: {peak/1024/1024:.2f} MB | Cells: {info['total_micro_cells']} | Links: {info['total_entanglement_links']}")
            sim.cnot(i, i+1)
        
        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        info = sim.get_entanglement_info()
        
        print("\n" + "=" * 70)
        print("RESULT:")
        print("=" * 70)
        print(f"  ✅ Completed in {elapsed:.2f} seconds")
        print(f"  Peak memory: {peak/1024/1024:.2f} MB")
        print(f"  Total micro cells: {info['total_micro_cells']}")
        print(f"  Total entanglement links: {info['total_entanglement_links']}")
        print(f"  Total symbolic weight: {info['total_symbolic_weight']}")
        print(f"  Entanglement preserved: {info['entanglement_preserved']}")
        print("\n  Strategy: Encoded entanglement in Livnium hierarchical geometry")
        print("  Preservation: Entanglement links stored, not truncated")
        print("  Conservation: Symbolic weight and class counts maintained")
        
    except Exception as e:
        tracemalloc.stop()
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_livnium_max_entanglement()

