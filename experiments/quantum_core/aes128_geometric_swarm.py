"""
AES-128 GEOMETRIC SWARM ATTACK (Thermal Edition)
Strategy: Massive Parallel Lattice Walking with Quantum Tunneling.

Improvement: Walkers are no longer greedy. They use Simulated Annealing (Metropolis)
to escape local minima (Ghost Keys).
"""

import time
import sys
import os
import random
import statistics
import math
import numpy as np
from typing import List, Tuple

# Ensure core is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.embedding.geometric_key_embedding import GeometricKeyEmbedding

# -------------------------------------------
# 1. Setup Cipher & Constraints
# -------------------------------------------

def get_aes_cipher(num_rounds: int):
    module_name = f"experiments.crypto.aes128_{num_rounds}round"
    class_name = f"AES128_{num_rounds}Round"
    try:
        mod = __import__(module_name, fromlist=[class_name])
        return getattr(mod, class_name)()
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return None

def generate_constraints(cipher, true_key: bytes, num_constraints: int = 4):
    constraints = []
    base_p = b"\x00" * 16
    # Spread targets across the block to catch more diffusion
    targets = [(0,0), (1,0), (2,0), (3,0), (4,0), (8,0), (12,0)][:num_constraints]
    
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

def measure_tension(cipher, key: bytes, constraints: List) -> float:
    total_errors = 0
    total_bits = 0
    try:
        for p1, p2, expected_delta in constraints:
            c1 = cipher.encrypt(p1, key)
            c2 = cipher.encrypt(p2, key)
            actual_delta = bytes(a ^ b for a, b in zip(c1, c2))
            errors = sum(bin(a ^ b).count("1") for a, b in zip(actual_delta, expected_delta))
            total_errors += errors
            total_bits += 128
        return total_errors / total_bits
    except:
        return 1.0

# -------------------------------------------
# 2. The Thermal Walker Agent
# -------------------------------------------

class ThermalWalker:
    def __init__(self, id, embedding, cipher, constraints, bounds=100):
        self.id = id
        self.embedding = embedding
        self.cipher = cipher
        self.constraints = constraints
        
        # Initialize random position
        self.x = random.randint(-bounds, bounds)
        self.y = random.randint(-bounds, bounds)
        self.z = random.randint(-bounds, bounds)
        self.weight = random.randint(0, 255)
        
        # State
        self.current_key = self.embedding.coords_to_key(self.x, self.y, self.z, self.weight)
        self.tension = measure_tension(self.cipher, self.current_key, self.constraints)
        
        # Physics Parameters
        self.temperature = 0.2  # Start hot enough to jump 0.05 barriers
        self.cooling_rate = 0.98
        self.stuck = False

    def step(self):
        """
        Metropolis-Hastings Step:
        1. Pick a random neighbor.
        2. If better, move.
        3. If worse, move with probability P = exp(-delta/T).
        """
        if self.temperature < 0.001: 
            self.stuck = True
            return False

        # 1. Propose a Move (Randomly select ONE neighbor to evaluate)
        # Efficient walking: don't scan all neighbors, just pick a direction
        move_type = "spatial" if random.random() > 0.2 else "entropy"
        
        if move_type == "spatial":
            # Pick random direction
            dims = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
            dx, dy, dz = random.choice(dims)
            nx, ny, nz = self.x + dx, self.y + dy, self.z + dz
            nw = self.weight
        else:
            # Pick entropy shift
            nx, ny, nz = self.x, self.y, self.z
            nw = (self.weight + random.randint(-10, 10)) % 256

        # 2. Measure
        cand_key = self.embedding.coords_to_key(nx, ny, nz, nw)
        new_tension = measure_tension(self.cipher, cand_key, self.constraints)
        
        # 3. Metropolis Decision
        delta = new_tension - self.tension
        
        accept = False
        if delta < 0:
            accept = True # Always accept improvement
        else:
            # Tunneling probability
            prob = math.exp(-delta / self.temperature)
            if random.random() < prob:
                accept = True
        
        if accept:
            self.x, self.y, self.z, self.weight = nx, ny, nz, nw
            self.current_key = cand_key
            self.tension = new_tension
        
        # 4. Cooldown
        self.temperature *= self.cooling_rate
        return True

# -------------------------------------------
# 3. The Swarm Manager
# -------------------------------------------

def run_swarm_attack():
    print("="*70)
    print("AES-128 GEOMETRIC SWARM ATTACK (Thermal Edition)")
    print("Strategy: 500 Thermal Walkers with Quantum Tunneling")
    print("="*70)

    # 1. Setup
    cipher = get_aes_cipher(num_rounds=2)
    true_key = os.urandom(16)
    print(f"Target Key: {true_key.hex()}")
    
    embedding = GeometricKeyEmbedding()
    constraints = generate_constraints(cipher, true_key, num_constraints=5)
    
    # 2. Spawn Swarm
    swarm_size = 500
    walkers = [ThermalWalker(i, embedding, cipher, constraints) for i in range(swarm_size)]
    
    print(f"  ✓ Spawned {swarm_size} thermal walkers.")
    initial_tensions = [w.tension for w in walkers]
    print(f"  Avg Start Tension: {statistics.mean(initial_tensions):.4f}")

    # 3. Run Evolution
    generations = 200 # More time for thermal equilibrium
    print(f"\n--- Starting Evolution ({generations} steps) ---")
    
    global_best_tension = 1.0
    global_best_key = None
    
    start_time = time.time()
    
    for gen in range(generations):
        active_walkers = 0
        
        # Batch processing
        tensions = []
        for w in walkers:
            if not w.stuck:
                active_walkers += 1
                w.step()
                
            tensions.append(w.tension)
            
            # Check Global Best
            if w.tension < global_best_tension:
                global_best_tension = w.tension
                global_best_key = w.current_key
                print(f"  [Gen {gen}] New Global Best! Tension: {w.tension:.4f} (Walker {w.id})")
                
                if global_best_tension == 0.0:
                    print(f"\n🏆 EUREKA! Key Found by Walker {w.id}")
                    print(f"Key: {w.current_key.hex()}")
                    return

        # Swarm Status
        if gen % 10 == 0:
            avg_t = statistics.mean(tensions)
            # Count how many are in the 'basin' (< 0.08)
            in_basin = sum(1 for t in tensions if t < 0.08)
            print(f"  Gen {gen}: Active {active_walkers} | Avg T {avg_t:.4f} | Best {global_best_tension:.4f} | In Basin: {in_basin}")
        
        if active_walkers == 0:
            print("\n  ⚠️  Swarm froze (Temperature reached 0).")
            break

    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"SWARM RESULT ({elapsed:.2f}s)")
    print(f"Target: {true_key.hex()}")
    print(f"Best:   {global_best_key.hex() if global_best_key else 'None'}")
    print(f"Tension: {global_best_tension:.4f}")
    
    if global_best_key == true_key:
        print("✅ SUCCESS: Exact key recovered.")
    elif global_best_tension < 0.05:
        print("✅ SUCCESS: Found Basin of Attraction (Very close).")
    else:
        print("❌ FAILURE: Swarm trapped in noise.")

if __name__ == "__main__":
    run_swarm_attack()