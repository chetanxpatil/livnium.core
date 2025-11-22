# Geometric Key Embedding

Maps cryptographic keys to 3D geometric coordinates using Gray codes.

## What is "Embedding"?

**"Embedding"** here means **mapping one space into another** while preserving structure:

- **From**: 3D geometric coordinates (x, y, z) in the Livnium lattice
- **To**: 128-bit cryptographic keys (AES-128 key space)
- **Key property**: **Locality preservation** - neighbors in 3D space become neighbors in key space

This allows us to use **geometric search** (walking through the 3D lattice) to explore the **key space** efficiently. Moving to an adjacent cell in 3D only flips a few bits in the resulting key, making it perfect for gradient-based search strategies.

## Contents

- **`geometric_key_embedding.py`**: Implements Gray code mapping between 3D lattice coordinates and 128-bit keys.

## Purpose

This module provides:
- **`coords_to_key()`**: Maps (x, y, z, entropy_seed) → 128-bit key
  - Uses Gray codes to preserve locality
  - Interleaves bits from x, y, z dimensions
  - Adds entropy_seed for fine-grained variation
- **`get_neighbors()`**: Returns 6 geometric neighbors (adjacent cells)
- **Locality preservation**: Neighbors in 3D space = neighbors in key space (minimal bit flips — typically 1–5 bits)

## How It Works

1. **Gray Code Conversion**: Converts coordinates to Gray codes (ensures adjacent numbers differ by 1 bit)
2. **Bit Interleaving**: Mixes x, y, z bits using Z-order curve style (distributes changes across the key)
3. **Entropy Mixing**: Adds symbolic weight as entropy seed for local fine-tuning

Used by AES cryptanalysis experiments to enable geometric search strategies - you can walk the lattice to search the key space!

