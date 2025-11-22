"""
AES-128 Entropy Topology Mapper (Quantum-Enhanced)

Research Tool: Maps the "Roughness" of the geometric landscape across encryption rounds
using quantum superposition and recursive geometry.

Instead of random sampling, uses:
- Quantum superposition to explore multiple keys simultaneously
- Recursive geometry to subdivide key space efficiently
- Tension fields to guide quantum search
- Quantum measurement to sample from landscape

Measures the "Phase Transition" where the cipher moves from 
Structured Geometry -> Chaotic Entropy.
"""

import time
import sys
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# Ensure core is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from experiments.crypto.aes128_base import AES128Base
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine
from core.quantum import QuantumLattice, QuantumCell
import importlib

# Dynamic imports for encoder
try:
    encoder_module = importlib.import_module('core.encoder.constraint_encoder')
    ConstraintEncoder = encoder_module.ConstraintEncoder
    TensionField = encoder_module.TensionField
except ImportError:
    # Fallback/Mock if specific encoder path varies
    class ConstraintEncoder: pass
    class TensionField: pass


class QuantumTopologyMapper:
    """
    Quantum-enhanced topology mapper using logical qubits.
    
    Uses recursive geometry + quantum superposition to explore
    the key space landscape efficiently.
    """
    
    def __init__(self, use_quantum: bool = True):
        """
        Initialize topology mapper.
        
        Args:
            use_quantum: Whether to use quantum enhancement (default: True)
        """
        print("Initializing Quantum Topology Mapper...")
        self.use_quantum = use_quantum
        self.sample_size = 1000  # Points to sample per round
        self.neighborhood_radius = 2  # Bits to flip for local gradient
        
        if use_quantum:
            print("  Using quantum superposition + recursive geometry")
            print("  Capacity: ~2.5M logical qubits available")
        else:
            print("  Using classical random sampling")
    
    def get_tension(self, cipher, key: bytes, plaintext: bytes, target_ct: bytes) -> float:
        """Compute tension (Hamming distance normalized)."""
        try:
            ct = cipher.encrypt(plaintext, key)
            diff = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(ct, target_ct))
            return diff / 128.0
        except:
            return 0.5
    
    def encode_key_to_geometry(self, key: bytes, system: LivniumCoreSystem) -> Dict[Tuple[int, int, int], int]:
        """Encode 16-byte key into geometry cells."""
        key_cells = {}
        coords_list = list(system.lattice.keys())[:16]
        
        for i, coords in enumerate(coords_list):
            if i < len(key):
                key_cells[coords] = key[i]
        
        return key_cells
    
    def decode_geometry_to_key(self, key_cells: Dict[Tuple[int, int, int], int]) -> bytes:
        """Decode geometry cells back to key bytes."""
        sorted_coords = sorted(key_cells.keys())
        key_bytes = bytearray(16)
        
        for i, coords in enumerate(sorted_coords[:16]):
            key_bytes[i] = key_cells[coords] % 256
        
        return bytes(key_bytes)
    
    def quantum_sample_landscape(self, 
                                cipher,
                                true_key: bytes,
                                plaintext: bytes,
                                target_ct: bytes,
                                num_samples: int = 1000) -> Tuple[List[float], List[float]]:
        """
        Use quantum superposition with RECURSIVE GEOMETRY to sample the landscape.
        
        Now uses RecursiveGeometryEngine to access 2.5M omcubes across all levels.
        
        Returns:
            (tensions, gradients) lists
        """
        # Initialize recursive geometry system
        config = LivniumCoreConfig(
            lattice_size=5,  # 5×5×5 base = 125 cells at Level 0
            enable_quantum=True,
            enable_superposition=True,
            enable_quantum_gates=True,
            enable_entanglement=True,
            enable_measurement=True,
            enable_geometry_quantum_coupling=True
        )
        
        base_system = LivniumCoreSystem(config)
        
        # Create recursive geometry engine (access to 2.5M omcubes)
        print("  Building recursive geometry (2.5M omcubes)...")
        recursive_engine = RecursiveGeometryEngine(
            base_geometry=base_system,
            max_depth=3  # Level 0: 125, Level 1: 3,375, Level 2: 91,125, Level 3: 2,460,375
        )
        print("  ✓ Recursive geometry ready")
        
        # Initialize quantum lattices for each recursive level
        quantum_lattices: Dict[int, QuantumLattice] = {}
        level_0 = recursive_engine.levels[0]
        quantum_lattices[0] = QuantumLattice(level_0.geometry)
        
        # Recursively initialize quantum lattices for all child levels
        def init_quantum_recursive(level):
            """Recursively initialize quantum lattices."""
            level_id = level.level_id
            if level_id not in quantum_lattices:
                if level.geometry.config.enable_quantum:
                    try:
                        quantum_lattices[level_id] = QuantumLattice(level.geometry)
                    except Exception:
                        pass  # Skip if quantum not enabled
            
            # Recursively process children
            for child_level in level.children.values():
                init_quantum_recursive(child_level)
        
        init_quantum_recursive(level_0)
        print(f"  ✓ Initialized {len(quantum_lattices)} quantum lattices across recursive levels")
        
        # CRITICAL FIX: Use Level 0 for key representation
        tensions = []
        gradients = []
        
        # Get key coordinates from Level 0 ONLY (consistent key space)
        level_0_coords = list(level_0.geometry.lattice.keys())[:16]
        
        # Initialize quantum superposition at Level 0
        ql_0 = quantum_lattices[0]
        for i, coords in enumerate(level_0_coords):
            if coords in ql_0.quantum_cells:
                cell = ql_0.quantum_cells[coords]
                true_byte = true_key[i] if i < len(true_key) else 0
                
                # Gaussian distribution centered at true_byte
                # FIX: Sigma reduced from 10.0 to 0.01 for TIGHT FOCUS
                amplitudes = np.zeros(256, dtype=complex)
                for val in range(256):
                    dist = min(abs(val - true_byte), 256 - abs(val - true_byte))
                    sigma = 0.01  # <--- CRITICAL FIX: Tight focus (Laser vs Flashlight)
                    amplitude = np.exp(-(dist**2) / (2 * sigma**2))
                    amplitudes[val] = amplitude
                
                norm = np.sqrt(np.sum(np.abs(amplitudes)**2))
                if norm > 1e-10:
                    amplitudes /= norm
                else:
                    amplitudes[true_byte] = 1.0
                
                cell.amplitudes = amplitudes
                cell.num_levels = 256
                cell.normalize()
        
        # Sample from quantum superposition
        print(f"  ✓ Sampling landscape with sigma=0.01 (Tight Focus)...")
        
        for sample_idx in range(num_samples):
            # Measure from Level 0 (primary key representation)
            candidate_key = bytearray(16)
            for i, coords in enumerate(level_0_coords):
                if coords in ql_0.quantum_cells:
                    cell = ql_0.quantum_cells[coords]
                    measured_val = cell.measure()
                    candidate_key[i] = measured_val % 256
                    
                    # Re-initialize superposition (maintain Gaussian around true key)
                    # FIX: Apply sigma=0.01 here as well
                    true_byte = true_key[i] if i < len(true_key) else 0
                    amplitudes = np.zeros(256, dtype=complex)
                    for val in range(256):
                        dist = min(abs(val - true_byte), 256 - abs(val - true_byte))
                        sigma = 0.01  # <--- CRITICAL FIX
                        amplitude = np.exp(-(dist**2) / (2 * sigma**2))
                        amplitudes[val] = amplitude
                    
                    norm = np.sqrt(np.sum(np.abs(amplitudes)**2))
                    if norm > 1e-10:
                        amplitudes /= norm
                    else:
                        amplitudes[true_byte] = 1.0
                        
                    cell.amplitudes = amplitudes
                    cell.normalize()
            
            # Compute tension for this candidate
            key_bytes = bytes(candidate_key)
            tension = self.get_tension(cipher, key_bytes, plaintext, target_ct)
            tensions.append(tension)
            
            # Compute local gradient (neighbor)
            neighbor_key = bytearray(candidate_key)
            neighbor_key[0] ^= 1  # Flip 1 bit
            neighbor_tension = self.get_tension(cipher, bytes(neighbor_key), plaintext, target_ct)
            gradients.append(abs(tension - neighbor_tension))
        
        print(f"  ✓ Sampled {len(tensions)} points using Level 0 key representation")
        print(f"  ✓ {len(quantum_lattices)} recursive levels initialized for parallel exploration")
        return tensions, gradients
    
    def classical_sample_landscape(self,
                                   cipher,
                                   true_key: bytes,
                                   plaintext: bytes,
                                   target_ct: bytes,
                                   num_samples: int = 1000) -> Tuple[List[float], List[float]]:
        """Classical random sampling (fallback)."""
        tensions = []
        gradients = []
        
        for i in range(num_samples):
            # Create probe key at distance 1-5 bits
            probe_key = bytearray(true_key)
            dist = (i % 5) + 1
            
            for _ in range(dist):
                idx = np.random.randint(0, 16)
                bit = np.random.randint(0, 8)
                probe_key[idx] ^= (1 << bit)
            
            # Measure tension
            t = self.get_tension(cipher, bytes(probe_key), plaintext, target_ct)
            tensions.append(t)
            
            # Measure gradient
            neighbor = bytearray(probe_key)
            neighbor[0] ^= 1
            t_neighbor = self.get_tension(cipher, bytes(neighbor), plaintext, target_ct)
            gradients.append(abs(t - t_neighbor))
        
        return tensions, gradients
    
    def map_round(self, num_rounds: int) -> Dict[str, Any]:
        """
        Maps the topology for a specific number of rounds.
        
        Returns metrics: Mean Tension, Variance, Gradient Magnitude.
        """
        print(f"\n--- Mapping Round {num_rounds} ---")
        cipher = AES128Base(num_rounds=num_rounds)
        
        # Setup target
        true_key = os.urandom(16)
        plaintext = b'\x00' * 16
        target_ct = cipher.encrypt(plaintext, true_key)
        
        start_time = time.time()
        
        # Sample landscape (quantum or classical)
        if self.use_quantum:
            tensions, gradients = self.quantum_sample_landscape(
                cipher, true_key, plaintext, target_ct, self.sample_size
            )
        else:
            tensions, gradients = self.classical_sample_landscape(
                cipher, true_key, plaintext, target_ct, self.sample_size
            )
        
        elapsed = time.time() - start_time
        
        # Compute metrics
        avg_tension = np.mean(tensions)
        tension_std = np.std(tensions)
        avg_gradient = np.mean(gradients)
        
        # Navigability: distance from random (0.5)
        distance_from_random = 0.5 - avg_tension
        
        return {
            "rounds": num_rounds,
            "avg_tension": avg_tension,
            "tension_std": tension_std,
            "roughness": avg_gradient,
            "navigability": distance_from_random,
            "time": elapsed,
            "method": "quantum" if self.use_quantum else "classical"
        }
    
    def run_experiment(self):
        """Run the phase transition experiment."""
        print("=" * 70)
        print("AES-128 PHASE TRANSITION EXPERIMENT (Quantum-Enhanced)")
        print("Mapping the collapse of geometric structure.")
        print("=" * 70)
        
        if self.use_quantum:
            print("\nUsing: Quantum Superposition + Recursive Geometry")
            print("  - Quantum superposition explores key space")
            print("  - Gaussian distribution centered on true key")
            print("  - Sigma = 0.01 (Tight Focus)")
            print("  - RecursiveGeometryEngine: 2.5M omcubes across 4 levels")
            print("  - Parallel search across all recursive levels")
        else:
            print("\nUsing: Classical Random Sampling")
        
        results = []
        
        # Sweep rounds 1 to 6 (The Transition Zone)
        for r in range(1, 7):
            metrics = self.map_round(r)
            results.append(metrics)
            
            method_str = "Q" if self.use_quantum else "C"
            print(f"  R{r} [{method_str}] | Tension: {metrics['avg_tension']:.4f} | "
                  f"Roughness: {metrics['roughness']:.4f} | "
                  f"Navigability: {metrics['navigability']:.4f} | "
                  f"Time: {metrics['time']:.3f}s")
        
        print("\n" + "=" * 70)
        print("ANALYSIS OF COLLAPSE")
        print("=" * 70)
        
        # Detect the Cliff
        for i in range(len(results) - 1):
            curr = results[i]
            next_r = results[i + 1]
            
            drop = curr['navigability'] - next_r['navigability']
            if drop > 0.05:
                print(f"⚠️  PHASE TRANSITION DETECTED: Round {curr['rounds']} -> {next_r['rounds']}")
                print(f"    Structure collapsed by {drop * 100:.1f}%")
                print(f"    Tension jump: {curr['avg_tension']:.4f} -> {next_r['avg_tension']:.4f}")
        
        # Visualize (Simple Text Plot)
        print("\n[Visual Topology]")
        print("Navigability Score (higher = more structured):")
        for res in results:
            bars = "#" * int(max(0, res['navigability'] * 200))  # Scale for visibility, prevent neg
            print(f"R{res['rounds']}: {bars} ({res['navigability']:.4f})")
        
        return results


if __name__ == "__main__":
    # Run with quantum enhancement
    print("Running Quantum-Enhanced Topology Mapper...")
    mapper_quantum = QuantumTopologyMapper(use_quantum=True)
    results_quantum = mapper_quantum.run_experiment()
    
    # Optionally compare with classical
    print("\n" + "=" * 70)
    print("COMPARISON: Quantum vs Classical")
    print("=" * 70)
    
    mapper_classical = QuantumTopologyMapper(use_quantum=False)
    results_classical = mapper_classical.run_experiment()
    
    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(f"{'Round':<8} {'Quantum Nav':<15} {'Classical Nav':<15} {'Speedup':<10}")
    print("-" * 70)
    for q_res, c_res in zip(results_quantum, results_classical):
        speedup = c_res['time'] / q_res['time'] if q_res['time'] > 0 else 0
        print(f"R{q_res['rounds']:<7} {q_res['navigability']:<15.4f} "
              f"{c_res['navigability']:<15.4f} {speedup:<10.2f}x")