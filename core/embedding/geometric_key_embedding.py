"""
Geometric Key Embedding: The Bridge between 3D Lattice and 128-bit Key Space.

Goal: Map (x,y,z) coordinates to a 128-bit integer such that geometric locality
preserves Hamming locality. Moving to a neighbor cell should only flip a few bits.

Technique: 3D Gray Code Interleaving.
"""

import numpy as np

class GeometricKeyEmbedding:
    def __init__(self):
        self.cache = {}

    def _int_to_gray(self, n: int) -> int:
        """Convert integer to Gray code."""
        return n ^ (n >> 1)

    def _gray_to_int(self, g: int) -> int:
        """Convert Gray code back to integer."""
        mask = g
        while mask:
            mask >>= 1
            g ^= mask
        return g

    def coords_to_key(self, x: int, y: int, z: int, entropy_seed: int = 0) -> bytes:
        """
        Maps 3D coordinates + a local 'weight' (entropy_seed) to a 128-bit key.
        
        Structure:
        - We treat (x,y,z) as high-order structure (The Neighborhood).
        - We treat 'entropy_seed' (symbolic weight) as low-order structure (Fine tuning).
        """
        # 1. Normalize coordinates to positive integers (shift origin)
        # Assuming lattice size ~100, shift by 1000 to be safe
        ux = abs(x + 1000)
        uy = abs(y + 1000)
        uz = abs(z + 1000)

        # 2. Convert spatial coords to Gray codes (Preserves locality)
        # We allocate ~32 bits per dimension
        gx = self._int_to_gray(ux)
        gy = self._int_to_gray(uy)
        gz = self._int_to_gray(uz)

        # 3. Interleave bits to mix dimensions (Z-order curve style)
        # This ensures that moving in X, Y, or Z changes bits distributed across the key
        spatial_part = 0
        for i in range(32):
            bit_x = (gx >> i) & 1
            bit_y = (gy >> i) & 1
            bit_z = (gz >> i) & 1
            
            spatial_part |= (bit_x << (3*i))
            spatial_part |= (bit_y << (3*i + 1))
            spatial_part |= (bit_z << (3*i + 2))

        # spatial_part is now roughly 96 bits.
        
        # 4. Add the Entropy Seed (The fine-grained local variation)
        # We use the symbolic weight (0-255) to fill the remaining bits and 
        # XOR against the spatial part to add local texture.
        # This allows the "Smart Finisher" to wiggle the key without changing coordinates.
        
        # We construct the 128-bit integer
        # Low 96 bits: Spatial geometry
        # High 32 bits: Entropy/Variation
        
        high_part = entropy_seed & 0xFFFFFFFF
        full_int = (high_part << 96) | spatial_part
        
        # XOR mixing to ensure entropy affects the whole key slightly
        # (Optional: keeps valid AES key format)
        mask = (entropy_seed * 0x123456789ABCDEF) % (2**128)
        final_int = full_int ^ mask
        
        return final_int.to_bytes(16, byteorder='big')

    def get_neighbors(self, x, y, z):
        """Return 6 geometric neighbors."""
        return [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1)
        ]