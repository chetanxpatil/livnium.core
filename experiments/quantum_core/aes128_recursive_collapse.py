"""
AES-128 Recursive Collapse (The Sculptor's Approach)
Phase 3: Geometric Manifold Search

This version uses the 'GeometricKeyEmbedding' (Gray Codes) to map
the 3D lattice onto the 128-bit key space.

Algorithm:
1. Initialize Level 0 with random entropy (Seeds).
2. Prune sectors based on 3D-to-128bit mapped keys.
3. Subdivide survivors (Zoom in on the manifold).
4. Smart Finisher: Walk the 1-bit gradient using the embedding.
"""

import time
import sys
import os
import numpy as np
import random
from typing import List, Tuple, Dict, Set

# Ensure core is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.embedding.geometric_key_embedding import GeometricKeyEmbedding
import importlib

# -------------------------------------------
# 1. Setup & Helpers
# -------------------------------------------

def get_aes_cipher(num_rounds: int):
    """Load AES cipher."""
    module_name = f"experiments.crypto.aes128_{num_rounds}round"
    class_name = f"AES128_{num_rounds}Round"
    try:
        mod = __import__(module_name, fromlist=[class_name])
        return getattr(mod, class_name)()
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return None

def generate_constraints(cipher, true_key: bytes, num_constraints: int = 3):
    """Generate differential constraints."""
    constraints = []
    base_p = b"\x00" * 16
    targets = [(0,0), (0,1), (1,0), (2,0), (3,0), (5,0), (10,0)][:num_constraints]
    
    for (byte_idx, bit_idx) in targets:
        p1 = base_p
        p2_arr = bytearray(base_p)
        p2_arr[byte_idx] ^= (1 << bit_idx)
        p2 = bytes(p2_arr)
        c1 = cipher.encrypt(p1, true_key)
        c2 = cipher.encrypt(p2, true_key)
        delta = bytes(a ^ b for a, b in zip(c1, c2))
        constraints.append((p1, p2, delta))
    return constraints

# -------------------------------------------
# 2. The Recursive Collapse Logic
# -------------------------------------------

class RecursiveCollapseSolver:
    def __init__(self, cipher, system):
        self.cipher = cipher
        self.system = system
        self.lattice = system.lattice
        self.embedding = GeometricKeyEmbedding()
        
    def measure_tension_for_key(self, key: bytes, constraints: List) -> float:
        """Exact tension measurement for a specific key."""
        total_errors = 0
        total_bits = 0
        try:
            for p1, p2, expected_delta in constraints:
                c1 = self.cipher.encrypt(p1, key)
                c2 = self.cipher.encrypt(p2, key)
                actual_delta = bytes(a ^ b for a, b in zip(c1, c2))
                errors = sum(bin(a ^ b).count("1") for a, b in zip(actual_delta, expected_delta))
                total_errors += errors
                total_bits += 128
            return total_errors / total_bits
        except:
            return 1.0

    def measure_sector_tension(self, coords: Tuple[int, int, int], constraints: List) -> float:
        """Measure tension using the GEOMETRIC EMBEDDING."""
        if coords in self.lattice:
            weight = int(self.lattice[coords].symbolic_weight)
        else:
            return 1.0 # Prune phantom sectors
            
        # CRITICAL UPDATE: Use Embedding instead of repeated bytes
        # This maps (x,y,z, weight) -> Unique 128-bit Key
        x, y, z = coords
        candidate_key = self.embedding.coords_to_key(x, y, z, entropy_seed=weight)
        
        return self.measure_tension_for_key(candidate_key, constraints)

    def get_children(self, coords: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Geometric Zoom with Bounds Checking."""
        x, y, z = coords
        children = []
        offsets = [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        for dx, dy, dz in offsets:
            child = (x+dx, y+dy, z+dz)
            if child in self.lattice:
                children.append(child)
        return children

    def run_collapse(self, constraints, max_depth=3):
        """Execute the Top-Down Pruning Search."""
        current_candidates = list(self.lattice.keys())
        print(f"--- Starting Global Collapse (Universe: {len(current_candidates)} sectors) ---")
        
        for depth in range(max_depth):
            print(f"\n[Depth {depth}] Scanning & Pruning...")
            survivors = []
            # Threshold slightly below random noise (0.5) to find the basin
            threshold = 0.48 - (depth * 0.02) 
            
            # Parallel Scan
            for coords in current_candidates:
                tension = self.measure_sector_tension(coords, constraints)
                if tension < threshold:
                    survivors.append(coords)
            
            print(f"  Stats: {len(current_candidates)} -> {len(survivors)} survivors (Threshold: {threshold:.2f})")
            
            if not survivors:
                print("  ❌ Extinction Event! (No keys found in this basin)")
                return None
                
            if depth < max_depth - 1:
                next_gen = set()
                for parent in survivors:
                    children = self.get_children(parent)
                    next_gen.update(children)
                current_candidates = list(next_gen)
                print(f"  ↳ Spawning {len(current_candidates)} children")
            else:
                current_candidates = survivors

        print(f"\n--- Collapse Complete ---")
        print(f"Final Candidate Sectors: {len(current_candidates)}")
        return current_candidates

    def refine_candidate_geometric(self, coords: Tuple[int, int, int], constraints: List) -> Tuple[bytes, float]:
        """
        The GEOMETRIC FINISHER:
        Instead of randomly mutating bytes, we walk the 3D lattice neighbors.
        This exploits the 1-bit locality we just proved.
        """
        weight = int(self.lattice[coords].symbolic_weight)
        x, y, z = coords
        
        current_key = self.embedding.coords_to_key(x, y, z, weight)
        best_key = current_key
        best_tension = self.measure_tension_for_key(best_key, constraints)
        
        # 1. Geometric Walk (Lattice Neighbors)
        # Since we proved neighbors = 1 bit flip, this is efficient gradient descent.
        improved = True
        while improved:
            improved = False
            neighbors = self.embedding.get_neighbors(x, y, z)
            
            for nx, ny, nz in neighbors:
                # Keep same weight, move in space
                cand_key = self.embedding.coords_to_key(nx, ny, nz, weight)
                t = self.measure_tension_for_key(cand_key, constraints)
                
                if t < best_tension:
                    best_tension = t
                    best_key = cand_key
                    x, y, z = nx, ny, nz # Move the walker
                    improved = True
        
        # 2. Entropy Walk (Weight refinement)
        # Once we find the best spot in space, fine-tune the entropy seed
        for w_offset in range(-10, 11):
            cand_key = self.embedding.coords_to_key(x, y, z, weight + w_offset)
            t = self.measure_tension_for_key(cand_key, constraints)
            if t < best_tension:
                best_tension = t
                best_key = cand_key
                
        return best_key, best_tension

# -------------------------------------------
# 3. Execution Wrapper
# -------------------------------------------

def run_experiment():
    print("="*70)
    print("AES-128 RECURSIVE COLLAPSE (Geometric Manifold Edition)")
    print("Strategy: 3D Gray Code Mapping -> Lattice Descent")
    print("="*70)
    
    # 1. Initialize System
    config = LivniumCoreConfig(lattice_size=7, enable_quantum=True)
    system = LivniumCoreSystem(config)
    print("  Entropy Injection: Randomizing geometric sectors...")
    import random
    for coords, cell in system.lattice.items():
        cell.symbolic_weight = float(random.randint(0, 255))
    print(f"  ✓ Initialized {len(system.lattice)} sectors")
    
    # 2. Setup AES (Round 2)
    cipher = get_aes_cipher(num_rounds=2)
    true_key = os.urandom(16)
    print(f"Target Key: {true_key.hex()}")
    
    # 3. Generate Constraints
    constraints = generate_constraints(cipher, true_key, num_constraints=4)
    
    # 4. Run Solver
    solver = RecursiveCollapseSolver(cipher, system)
    start_time = time.time()
    
    # Global Pruning
    result = solver.run_collapse(constraints, max_depth=3)
    
    if result:
        print(f"\n--- Running 'The Geometric Finisher' ---")
        best_key = None
        best_tension = 1.0
        
        # Optimize survivors
        print(f"Refining {len(result)} candidates via Lattice Walk...")
        for coords in result:
            refined_key, t = solver.refine_candidate_geometric(coords, constraints)
            
            if t < best_tension:
                best_tension = t
                best_key = refined_key
                print(f"  New Best: Tension {t:.4f} | Key: {refined_key.hex()[:8]}...")
                if t == 0.0: break
        
        elapsed = time.time() - start_time
        print("-" * 40)
        print(f"Final Result in {elapsed:.3f}s:")
        print(f"Found:  {best_key.hex() if best_key else 'None'}")
        print(f"Tension: {best_tension:.4f}")
        
        if best_tension < 0.01:
            print("🏆 PERFECT MATCH / EXTREME PROXIMITY")
        elif best_tension < 0.12:
            print("✅ Significant Gradient Found (Better than random)")
        else:
            print("⚠️  Stuck in Noise Floor")

if __name__ == "__main__":
    run_experiment()