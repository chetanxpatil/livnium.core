"""
AES-128 Recursive Collapse (The Sculptor's Approach)

This experiment implements the "Big to Small" search strategy.
Instead of building a key, it starts with the whole lattice (Universe)
and aggressively prunes sectors that have High Tension.

Algorithm:
1. Initialize Level 0 (Satellite View).
2. Apply constraints to entire sectors.
3. Prune sectors with high tension (Kill the bad branches).
4. Subdivide survivors to Level 1.
5. Repeat until the Key is isolated.
6. FINISHER: Run local refinement on survivors to snap to exact key.
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
from core.recursive import RecursiveGeometryEngine
from core.quantum import QuantumLattice, TrueQuantumRegister
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

def decode_sector_to_key(center_weight: int) -> bytes:
    """
    Hypothesis: The 'Center' of a sector represents a specific key byte value.
    """
    val = int(center_weight) % 256
    return bytes([val] * 16)

def generate_constraints(cipher, true_key: bytes, num_constraints: int = 3):
    """Generate geometric constraints (triangulation)."""
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
        """Measure tension for a specific geometric sector."""
        if coords in self.lattice:
            weight = int(self.lattice[coords].symbolic_weight)
        else:
            # Phantom sector (outside lattice) - treat as high tension to prune it
            return 1.0 
            
        # Simplified heuristic: Use the weight as a repeated byte
        candidate_key = bytes([weight % 256] * 16)
        return self.measure_tension_for_key(candidate_key, constraints)

    def get_children(self, coords: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """Geometric Zoom."""
        x, y, z = coords
        children = []
        offsets = [(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        for dx, dy, dz in offsets:
            child = (x+dx, y+dy, z+dz)
            # --- CRITICAL FIX: Bounds Checking ---
            # Only return children that actually exist in our initialized universe.
            if child in self.lattice:
                children.append(child)
        return children

    def run_collapse(self, constraints, max_depth=3):
        """Execute the Top-Down Pruning Search."""
        # Start with all Level 0 cells
        current_candidates = list(self.lattice.keys()) # Use full lattice
        
        print(f"--- Starting Global Collapse (Universe: {len(current_candidates)} sectors) ---")
        
        for depth in range(max_depth):
            print(f"\n[Depth {depth}] Scanning & Pruning...")
            survivors = []
            threshold = 0.35 - (depth * 0.08) 
            
            for coords in current_candidates:
                tension = self.measure_sector_tension(coords, constraints)
                if tension < threshold:
                    survivors.append(coords)
            
            print(f"  Stats: {len(current_candidates)} -> {len(survivors)} survivors (Threshold: {threshold:.2f})")
            
            if not survivors:
                print("  ❌ Extinction Event! Relaxing threshold...")
                return None
                
            if depth < max_depth - 1:
                next_gen = set()
                for parent in survivors:
                    children = self.get_children(parent)
                    next_gen.update(children)
                current_candidates = list(next_gen)
                print(f"  ↳ Spawning {len(current_candidates)} children for Depth {depth+1}")
            else:
                current_candidates = survivors

        print(f"\n--- Collapse Complete ---")
        print(f"Final Candidate Sectors: {len(current_candidates)}")
        return current_candidates

    def refine_candidate(self, seed_weight: int, constraints: List) -> Tuple[bytes, float]:
        """
        The FINISHER: Takes a seed value and runs byte-wise optimization.
        This snaps the 'Neighborhood' guess to the 'Exact Key'.
        """
        # Start with the repeated byte guess
        base_val = seed_weight % 256
        current_key = bytearray([base_val] * 16)
        
        best_tension = self.measure_tension_for_key(bytes(current_key), constraints)
        
        # Iterative refinement (Hill Climbing)
        improved = True
        while improved:
            improved = False
            for i in range(16): # For each byte
                original_byte = current_key[i]
                best_local_byte = original_byte
                
                # Try probing local neighbors (+/- 1, 10, etc) and random jumps
                candidates = [original_byte]
                candidates.extend([(original_byte + x) % 256 for x in range(-5, 6)]) # Local
                candidates.extend([random.randint(0, 255) for _ in range(5)]) # Random jumps
                
                for cand in candidates:
                    current_key[i] = cand
                    t = self.measure_tension_for_key(bytes(current_key), constraints)
                    if t < best_tension:
                        best_tension = t
                        best_local_byte = cand
                        improved = True
                    else:
                        current_key[i] = original_byte # Revert
                
                current_key[i] = best_local_byte # Commit best
        
        return bytes(current_key), best_tension

# -------------------------------------------
# 3. Execution Wrapper
# -------------------------------------------

def run_experiment():
    print("="*70)
    print("AES-128 RECURSIVE COLLAPSE (The Sculptor's Approach)")
    print("Strategy: Global Pruning -> Local Refinement (Moksha)")
    print("="*70)
    
    # 1. Initialize System with Entropy
    config = LivniumCoreConfig(lattice_size=7, enable_quantum=True)
    system = LivniumCoreSystem(config)
    print("  Entropy Injection: Randomizing geometric sectors...")
    
    # Ensure weights are properly randomized
    import random
    for coords, cell in system.lattice.items():
        cell.symbolic_weight = float(random.randint(0, 255))
    print(f"  ✓ Randomized {len(system.lattice)} sectors")
    
    # 2. Setup AES (Round 2)
    cipher = get_aes_cipher(num_rounds=2)
    true_key = os.urandom(16)
    print(f"Target Key: {true_key.hex()}")
    
    # 3. Generate Constraints
    constraints = generate_constraints(cipher, true_key, num_constraints=4)
    
    # 4. Run Solver
    solver = RecursiveCollapseSolver(cipher, system)
    start_time = time.time()
    result = solver.run_collapse(constraints, max_depth=3)
    
    if result:
        print(f"\n--- Running 'The Finisher' (Local Refinement) ---")
        
        best_key = None
        best_tension = 1.0
        
        # Safe extraction of weights
        surviving_weights = set()
        for c in result:
            if c in system.lattice:
                surviving_weights.add(int(system.lattice[c].symbolic_weight))
        
        surviving_weights = list(surviving_weights)
        print(f"Optimizing {len(surviving_weights)} unique seed values...")
        
        for weight in surviving_weights:
            refined_key, t = solver.refine_candidate(weight, constraints)
            
            if t < best_tension:
                best_tension = t
                best_key = refined_key
                print(f"  New Best: Tension {t:.4f} | Key: {refined_key.hex()}")
                if t == 0.0: break # Perfect match
        
        elapsed = time.time() - start_time
        print("-" * 40)
        print(f"Final Result in {elapsed:.3f}s:")
        print(f"Target: {true_key.hex()}")
        print(f"Found:  {best_key.hex() if best_key else 'None'}")
        
        if best_key and best_key == true_key:
            print("🏆 PERFECT MATCH! KEY RECOVERED.")
        elif best_key and best_tension < 0.05:
            print(f"✅ Extremely Close Match (Tension {best_tension:.4f})")
        else:
            print(f"⚠️  Convergence Incomplete (Tension {best_tension:.4f})")

if __name__ == "__main__":
    run_experiment()