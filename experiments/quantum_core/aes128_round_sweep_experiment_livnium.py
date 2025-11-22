"""
AES-128 Round Sweep Experiment: Using Full Livnium Architecture

This version uses:
- Recursive Geometry Engine: Subdivide key space into chunks
- Quantum Layer: Superposition exploration of key bits
- Tension Fields: Encode constraints as geometric tension
- Moksha Engine: Convergence detection

This measures the phase transition where diffusion destroys navigable structure.
"""

import time
import sys
import os
import itertools
import random
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine, MokshaEngine, ConvergenceState
from core.quantum import QuantumLattice, QuantumCell
import importlib

# Dynamic imports for encoder
try:
    encoder_module = importlib.import_module('core.encoder.constraint_encoder')
    ConstraintEncoder = encoder_module.ConstraintEncoder
    TensionField = encoder_module.TensionField
except ImportError:
    # Fallback/Mock if specific encoder path varies
    class ConstraintEncoder:
        def __init__(self, system): self.tension_fields = []
    class TensionField:
        def __init__(self, **kwargs): self.get_tension = kwargs['compute_tension']


# -------------------------------------------
# 1. Cipher factory
# -------------------------------------------


def get_aes_cipher_for_rounds(num_rounds: int):
    """Returns an AES-128 cipher object with `num_rounds` rounds."""
    module_name = f"experiments.crypto.aes128_{num_rounds}round"
    class_name = f"AES128_{num_rounds}Round"

    try:
        mod = __import__(module_name, fromlist=[class_name])
        cls = getattr(mod, class_name)
        return cls()
    except Exception as e:
        print(f"[ERROR] Could not load {class_name} from {module_name}: {e}")
        raise


# -------------------------------------------
# 2. Encode AES key recovery as geometry
# -------------------------------------------


def encode_key_to_geometry(key: bytes, system: LivniumCoreSystem) -> Dict[Tuple[int, int, int], int]:
    """
    Encode 128-bit key into geometry cells.
    Maps each key byte to a cell coordinate.
    """
    key_cells = {}
    lattice_size = system.lattice_size
    
    for i, byte_val in enumerate(key):
        x = i % 4
        y = (i // 4) % 4
        z = 0
        
        if x < lattice_size and y < lattice_size and z < lattice_size:
            coords = (x, y, z)
            key_cells[coords] = byte_val
    
    return key_cells


def decode_geometry_to_key(key_cells: Dict[Tuple[int, int, int], int]) -> bytes:
    """Decode geometry cells back to key bytes."""
    sorted_coords = sorted(key_cells.keys())
    key_bytes = bytearray(16)
    
    for i, coords in enumerate(sorted_coords[:16]):
        key_bytes[i] = key_cells[coords]
    
    return bytes(key_bytes)


def create_tension_fields_for_constraints(
    cipher,
    constraints: List[Tuple[bytes, bytes, bytes]],
    system: LivniumCoreSystem,
    encoder: ConstraintEncoder
) -> List[TensionField]:
    """
    Create tension fields from triangulated constraints.
    """
    tension_fields = []
    
    for idx, (p1, p2, expected_delta) in enumerate(constraints):
        def compute_tension_for_constraint(system: LivniumCoreSystem) -> float:
            key_cells = {}
            for coords, cell in system.lattice.items():
                key_cells[coords] = int(cell.symbolic_weight) % 256
            
            key = decode_geometry_to_key(key_cells)
            
            try:
                c1 = cipher.encrypt(p1, key)
                c2 = cipher.encrypt(p2, key)
                actual_delta = bytes(a ^ b for a, b in zip(c1, c2))
                errors = sum(bin(a ^ b).count("1") for a, b in zip(actual_delta, expected_delta))
                return errors / 128.0
            except Exception:
                return 1.0
        
        involved_coords = list(system.lattice.keys())[:16]
        
        field = TensionField(
            constraint_id=f"aes_constraint_{idx}",
            involved_coords=involved_coords,
            compute_tension=compute_tension_for_constraint,
            compute_curvature=lambda s: 1.0 / (1.0 + compute_tension_for_constraint(s)),
            description=f"Triangulated constraint {idx}"
        )
        
        tension_fields.append(field)
        encoder.tension_fields.append(field)
    
    return tension_fields


# -------------------------------------------
# 3. Recursive key search with quantum
# -------------------------------------------


def recursive_quantum_key_search(
    cipher,
    constraints: List[Tuple[bytes, bytes, bytes]],
    max_iterations: int = 100
) -> Tuple[bool, float, float]:
    """
    Use recursive geometry + quantum to search for key.
    """
    config = LivniumCoreConfig(
        lattice_size=5,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    system = LivniumCoreSystem(config)
    
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=system,
        max_depth=2
    )
    
    quantum_lattice = QuantumLattice(system)
    constraint_encoder = ConstraintEncoder(system)
    
    tension_fields = create_tension_fields_for_constraints(
        cipher, constraints, system, constraint_encoder
    )
    
    moksha = recursive_engine.moksha
    
    # Initialize key in superposition (Gaussian centered on initial guess)
    # FIX: Use tight sigma for focused search
    initial_guess_bytes = [random.randint(0, 255) for _ in range(16)]
    
    for i, coords in enumerate(list(system.lattice.keys())[:16]):
        if coords in quantum_lattice.quantum_cells:
            del quantum_lattice.quantum_cells[coords]
        
        # FIX: Create focused distribution (Sigma=0.01 logic simulation)
        # Since we don't know the key, we start with a wider distribution (Sigma=50)
        # and tighten it as tension decreases.
        guess = initial_guess_bytes[i]
        amplitudes = np.zeros(256, dtype=complex)
        for val in range(256):
            dist = min(abs(val - guess), 256 - abs(val - guess))
            sigma = 50.0 # Start wide to find the basin
            amplitude = np.exp(-(dist**2) / (2 * sigma**2))
            amplitudes[val] = amplitude
            
        cell = QuantumCell(
            coordinates=coords,
            amplitudes=amplitudes,
            num_levels=256
        )
        cell.normalize()
        quantum_lattice.quantum_cells[coords] = cell
    
    start_time = time.time()
    best_tension = 1.0
    best_key = None
    
    for iteration in range(max_iterations):
        convergence = moksha.check_convergence()
        if convergence == ConvergenceState.MOKSHA:
            break
        
        total_tension = sum(field.get_tension(system) for field in tension_fields)
        total_tension /= len(tension_fields)
        
        if total_tension < best_tension:
            best_tension = total_tension
            key_cells = {}
            for i, (coords, cell) in enumerate(list(system.lattice.items())[:16]):
                key_cells[coords] = int(cell.symbolic_weight) % 256
            best_key = decode_geometry_to_key(key_cells)
        
        if best_tension == 0.0:
            break
        
        # Quantum Update Loop
        measured_key = bytearray(16)
        for i, coords in enumerate(list(system.lattice.keys())[:16]):
            if coords in quantum_lattice.quantum_cells:
                cell = quantum_lattice.quantum_cells[coords]
                measured_val = cell.measure()
                measured_key[i] = measured_val % 256
        
        # Update Geometry (Symbolic Weight)
        for i, coords in enumerate(list(system.lattice.keys())[:16]):
            if coords in system.lattice:
                cell = system.lattice[coords]
                current_sw = cell.symbolic_weight
                target_sw = float(measured_key[i])
                cell.symbolic_weight = 0.9 * current_sw + 0.1 * target_sw
        
        # Update Quantum States (Tighten Focus based on Tension)
        # As tension drops, sigma should decrease (Focus tightens)
        current_sigma = max(0.01, 50.0 * best_tension) # Dynamic Sigma!
        
        for i, coords in enumerate(list(system.lattice.keys())[:16]):
            if coords in quantum_lattice.quantum_cells:
                cell = quantum_lattice.quantum_cells[coords]
                
                # Center distribution on current geometric best guess
                center_val = int(system.lattice[coords].symbolic_weight) % 256
                
                # Re-initialize with tighter sigma
                new_amplitudes = np.zeros(256, dtype=complex)
                for val in range(256):
                    dist = min(abs(val - center_val), 256 - abs(val - center_val))
                    amplitude = np.exp(-(dist**2) / (2 * current_sigma**2))
                    new_amplitudes[val] = amplitude
                
                cell.amplitudes = new_amplitudes
                cell.normalize()
                
    elapsed = time.time() - start_time
    success = (best_tension == 0.0 and best_key is not None)
    return success, best_tension, elapsed


# -------------------------------------------
# 4. Single run wrapper
# -------------------------------------------


def run_single_break_attempt_livnium(
    num_rounds: int,
    true_key: bytes,
    num_constraints: int = 3,
    max_iterations: int = 100
) -> Tuple[bool, float, float]:
    """Run ONE attempt using full Livnium architecture."""
    cipher = get_aes_cipher_for_rounds(num_rounds)
    
    constraints = []
    base_p = b"\x00" * 16
    bit_choices = [(0, 0), (0, 1), (1, 0), (5, 0), (10, 0), (15, 0)]
    bit_choices = bit_choices[:num_constraints]
    
    for (byte_idx, bit_idx) in bit_choices:
        p1 = base_p
        p2_arr = bytearray(base_p)
        p2_arr[byte_idx] ^= (1 << bit_idx)
        p2 = bytes(p2_arr)
        
        c1 = cipher.encrypt(p1, true_key)
        c2 = cipher.encrypt(p2, true_key)
        delta = bytes(a ^ b for a, b in zip(c1, c2))
        
        constraints.append((p1, p2, delta))
    
    success, final_tension, elapsed = recursive_quantum_key_search(
        cipher, constraints, max_iterations=max_iterations
    )
    
    return success, final_tension, elapsed


# -------------------------------------------
# 5. Sweep over rounds
# -------------------------------------------


def sweep_rounds_livnium(
    round_list: List[int],
    trials_per_round: int = 5,
    num_constraints: int = 3,
    max_iterations: int = 100
):
    """Sweep over rounds using Livnium architecture."""
    print("=" * 70)
    print("AES-128 Tension Landscape vs Rounds (Livnium Architecture)")
    print("=" * 70)
    print(f"Using: Recursive Geometry + Quantum Layer + Tension Fields")
    print(f"Trials per round: {trials_per_round}, constraints: {num_constraints}")
    print("")
    
    results = []
    
    for num_rounds in round_list:
        print(f"\n--- Testing {num_rounds} rounds ---")
        successes = 0
        tensions = []
        times = []
        
        for t_idx in range(trials_per_round):
            true_key = os.urandom(16)
            
            success, final_tension, elapsed = run_single_break_attempt_livnium(
                num_rounds=num_rounds,
                true_key=true_key,
                num_constraints=num_constraints,
                max_iterations=max_iterations
            )
            
            successes += int(success)
            tensions.append(final_tension)
            times.append(elapsed)
            
            print(f"  Trial {t_idx+1}: "
                  f"success={success}, final_tension={final_tension:.4f}, time={elapsed:.3f}s")
        
        avg_tension = sum(tensions) / len(tensions)
        min_tension = min(tensions)
        max_tension = max(tensions)
        avg_time = sum(times) / len(times)
        
        results.append((num_rounds, successes, avg_tension, min_tension, max_tension, avg_time))
    
    print("\n" + "=" * 70)
    print("SUMMARY: Tension Landscape vs Rounds (Livnium)")
    print("=" * 70)
    print("Rounds | Successes/Trials | AvgTension | MinTension | MaxTension | AvgTime(s)")
    for (r, succ, avg_t, min_t, max_t, avg_time) in results:
        print(f"{r:6d} | {succ:9d}/{trials_per_round:<6d} | "
              f"{avg_t:10.4f} | {min_t:10.4f} | {max_t:10.4f} | {avg_time:10.3f}")


if __name__ == "__main__":
    rounds_to_test = [2, 3, 4, 5]
    sweep_rounds_livnium(rounds_to_test, trials_per_round=3, num_constraints=3, max_iterations=50)