"""
Phase 2 Verification: Testing Geometric Locality
Checks if moving in the lattice corresponds to small jumps in Key Space.
"""
import sys
import os
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from core.embedding.geometric_key_embedding import GeometricKeyEmbedding

def hamming_distance(b1: bytes, b2: bytes) -> int:
    diff = 0
    for x, y in zip(b1, b2):
        diff += bin(x ^ y).count('1')
    return diff

def run_test():
    embedding = GeometricKeyEmbedding()
    print("="*60)
    print("GEOMETRIC EMBEDDING LOCALITY TEST")
    print("="*60)
    
    distances = []
    
    # Sample random points in the lattice
    for _ in range(100):
        cx = int(np.random.randint(-50, 50))
        cy = int(np.random.randint(-50, 50))
        cz = int(np.random.randint(-50, 50))
        
        # Base Key
        base_key = embedding.coords_to_key(cx, cy, cz, entropy_seed=100)
        
        # Check neighbors
        neighbors = embedding.get_neighbors(cx, cy, cz)
        for nx, ny, nz in neighbors:
            n_key = embedding.coords_to_key(nx, ny, nz, entropy_seed=100)
            dist = hamming_distance(base_key, n_key)
            distances.append(dist)
            
    avg_dist = statistics.mean(distances)
    min_dist = min(distances)
    max_dist = max(distances)
    
    print(f"Neighbor Hamming Distance Stats:")
    print(f"  Average: {avg_dist:.2f} bits")
    print(f"  Min:     {min_dist} bits")
    print(f"  Max:     {max_dist} bits")
    
    # Benchmark vs Random
    rand_k1 = os.urandom(16)
    rand_k2 = os.urandom(16)
    rand_dist = hamming_distance(rand_k1, rand_k2)
    print(f"\nReference (Random Keys): ~{rand_dist} bits")
    
    if avg_dist < 10:
        print("\n✅ SUCCESS: Geometry is LOCAL. Neighbors are similar.")
    else:
        print("\n❌ FAILURE: Geometry is chaotic. Embedding needs fixing.")

if __name__ == "__main__":
    import numpy as np
    import os
    run_test()