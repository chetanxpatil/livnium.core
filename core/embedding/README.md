# Geometric Key Embedding

Maps cryptographic keys to 3D geometric coordinates using Gray codes.

## Contents

- **`geometric_key_embedding.py`**: Implements Gray code mapping between 3D lattice coordinates and 128-bit keys.

## Purpose

This module provides:
- **`coords_to_key()`**: Maps (x, y, z, entropy_seed) → 128-bit key
- **`get_neighbors()`**: Returns 1-bit-flip neighbors in key space
- **Locality preservation**: Neighbors in 3D space = neighbors in key space

Used by AES cryptanalysis experiments to enable geometric search strategies.

