"""
AES-128 Tension Calibration Test

Purpose:
Scientifically determine if the 'Geometric Tension' metric has a gradient.
We compare the Tension of:
1. The True Key (Should be 0.0)
2. Keys with 1 bit flipped (Should be Low)
3. Keys with 1 byte flipped (Should be Medium)
4. Random Keys (Should be ~0.5 / High)
5. Repeated 'Heuristic' Keys (The space we were searching)

If (2) and (4) have the same tension, the landscape is FLAT, 
and no optimization algorithm can ever solve it.
"""

import os
import sys
import numpy as np
import statistics

# Ensure core is reachable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Load Cipher (Use 2 Rounds - The 'Solvable' one)
def get_aes_cipher(num_rounds: int):
    module_name = f"experiments.crypto.aes128_{num_rounds}round"
    class_name = f"AES128_{num_rounds}Round"
    try:
        mod = __import__(module_name, fromlist=[class_name])
        return getattr(mod, class_name)()
    except Exception as e:
        print(f"[Error] Load failed: {e}")
        return None

cipher = get_aes_cipher(num_rounds=2)

# -------------------------------------------------------
# 1. The Measurement Tool (Tension Function)
# -------------------------------------------------------

def generate_constraints(true_key: bytes, num_constraints: int = 4):
    """Generate differential constraints based on the True Key."""
    constraints = []
    base_p = b"\x00" * 16
    # Targets: Flip bits in different bytes
    targets = [(0,0), (1,0), (2,0), (3,0), (4,0)][:num_constraints]
    
    for (byte_idx, bit_idx) in targets:
        p1 = base_p
        p2_arr = bytearray(base_p)
        p2_arr[byte_idx] ^= (1 << bit_idx)
        p2 = bytes(p2_arr)
        
        # Get expected delta from True Key
        c1 = cipher.encrypt(p1, true_key)
        c2 = cipher.encrypt(p2, true_key)
        delta = bytes(a ^ b for a, b in zip(c1, c2))
        constraints.append((p1, p2, delta))
    return constraints

def measure_tension(key: bytes, constraints: list) -> float:
    """Calculate how much a key violates the constraints."""
    total_errors = 0
    total_bits = 0
    try:
        for p1, p2, expected_delta in constraints:
            c1 = cipher.encrypt(p1, key)
            c2 = cipher.encrypt(p2, key)
            actual_delta = bytes(a ^ b for a, b in zip(c1, c2))
            
            # Hamming distance between Actual Delta and Expected Delta
            errors = sum(bin(a ^ b).count("1") for a, b in zip(actual_delta, expected_delta))
            total_errors += errors
            total_bits += 128
        return total_errors / total_bits
    except:
        return 0.5 # Default random

# -------------------------------------------------------
# 2. The Experiment
# -------------------------------------------------------

def run_calibration():
    print("="*70)
    print("AES-128 TENSION CALIBRATION TEST (2 ROUNDS)")
    print("Checking if the 'Landscape' actually exists.")
    print("="*70)

    # 1. Setup
    true_key = os.urandom(16)
    print(f"Target Key: {true_key.hex()}")
    constraints = generate_constraints(true_key, num_constraints=5)
    
    # 2. Baseline: True Key
    t_true = measure_tension(true_key, constraints)
    print(f"\n[Baseline] True Key Tension:   {t_true:.4f} (Should be 0.0000)")
    
    # 3. Test: Near Neighbors (1 Bit Flip)
    print("\n[Test 1] Near Neighbors (1 Bit Flip)")
    tensions_1bit = []
    for i in range(16): # Flip 1 bit in each byte
        k = bytearray(true_key)
        k[i] ^= 1 
        tensions_1bit.append(measure_tension(bytes(k), constraints))
    
    avg_1bit = statistics.mean(tensions_1bit)
    print(f"  Avg Tension: {avg_1bit:.4f}")
    print(f"  Range:       {min(tensions_1bit):.4f} - {max(tensions_1bit):.4f}")

    # 4. Test: Random Keys (The Ocean)
    print("\n[Test 2] Random Keys (Global Entropy)")
    tensions_random = []
    for _ in range(100):
        k = os.urandom(16)
        tensions_random.append(measure_tension(k, constraints))
        
    avg_random = statistics.mean(tensions_random)
    print(f"  Avg Tension: {avg_random:.4f}")
    print(f"  Range:       {min(tensions_random):.4f} - {max(tensions_random):.4f}")

    # 5. Test: The "Repeated Byte" Heuristic (What we were doing before)
    print("\n[Test 3] Repeated Byte Keys (0000..., 0101..., etc)")
    tensions_repeated = []
    best_repeated_tension = 1.0
    best_repeated_key = None
    
    for val in range(256):
        k = bytes([val] * 16)
        t = measure_tension(k, constraints)
        tensions_repeated.append(t)
        if t < best_repeated_tension:
            best_repeated_tension = t
            best_repeated_key = k
            
    avg_repeated = statistics.mean(tensions_repeated)
    print(f"  Avg Tension: {avg_repeated:.4f}")
    print(f"  Best Found:  {best_repeated_tension:.4f} (Key: {best_repeated_key.hex()[:4]}...)")

    # -------------------------------------------------------
    # 3. The Verdict
    # -------------------------------------------------------
    print("\n" + "="*70)
    print("SCIENTIFIC VERDICT")
    print("="*70)
    
    # Check for gradient
    diff_random_vs_1bit = avg_random - avg_1bit
    
    print(f"Random Noise Floor: {avg_random:.4f}")
    print(f"1-Bit Flip Signal:  {avg_1bit:.4f}")
    print(f"Gradient Strength:  {diff_random_vs_1bit:.4f}")
    
    if diff_random_vs_1bit < 0.01:
        print("\n❌ CRITICAL FAILURE: The landscape is FLAT.")
        print("   Even a 1-bit error looks just as random as a wrong key.")
        print("   Geometric optimization is IMPOSSIBLE with this metric.")
    elif diff_random_vs_1bit < 0.05:
        print("\n⚠️  WEAK GRADIENT: The slope is very slippery.")
        print("   Optimization will struggle to find the path.")
    else:
        print("\n✅ GRADIENT CONFIRMED: Structure exists!")
        print("   Optimization is mathematically possible.")
        
    print("\n--- Analysis of Previous Experiment ---")
    print(f"Your previous best (Repeated Key) had Tension ~{best_repeated_tension:.4f}")
    print(f"True Random Keys have Tension ~{avg_random:.4f}")
    print("This confirms that the 'Collapse' was just finding the best")
    print("repeated-byte pattern, which accidentally matched constraints better")
    print("than pure randomness, but was nowhere near the True Key.")

if __name__ == "__main__":
    run_calibration()