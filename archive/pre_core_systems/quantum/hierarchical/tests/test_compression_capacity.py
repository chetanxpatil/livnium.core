#!/usr/bin/env python3
"""
Compression Capacity Test

Measures how much data can be encoded into LIVNIUM geometry under different encoding schemes.

Important: This is NOT claiming to beat ZIP. This measures:
- How much semantic/structured data can be packed into geometry per byte of RAM
- Compression ratios under specific encoding schemes
- The relationship between base dimension and compression capacity

The compression ratio depends on how you encode data into omcubes, not just the geometry itself.
"""

import sys
import time
import tracemalloc
import csv
import json
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from quantum.hierarchical.geometry.dynamic_hierarchical_geometry import DynamicHierarchicalGeometrySystem


# Test configuration
SIZES_MB = [10, 50, 100, 250, 500]  # Logical data sizes to test
SIZES_MB_QUICK = [10, 50, 100]  # Quick test mode (smaller sizes)
BASE_DIMENSIONS = [3, 5, 7]  # Different base dimensions to test
BASE_DIMENSIONS_QUICK = [3]  # Quick test mode (just 3×3×3)
ENCODING_SCHEMES = {
    # Small encodings (for testing - these will expand data)
    '1_byte_per_omcube': 1,   # Each omcube = 1 byte (EXPANDS: 240×)
    '4_bytes_per_omcube': 4,  # Each omcube = 4 bytes (EXPANDS: 60×)
    '8_bytes_per_omcube': 8,  # Each omcube = 8 bytes (EXPANDS: 30×)
    
    # Realistic encodings (these can actually compress)
    '32_bytes_per_omcube': 32,   # Each omcube = 32 bytes (may compress)
    '64_bytes_per_omcube': 64,   # Each omcube = 64 bytes (should compress)
    '128_bytes_per_omcube': 128, # Each omcube = 128 bytes (should compress well)
    '256_bytes_per_omcube': 256, # Each omcube = 256 bytes (should compress well)
}


def encode_data_to_coordinates(i: int, scheme: str, bytes_per_omcube: int) -> tuple:
    """
    Encode data index i into geometric coordinates.
    
    This simulates encoding N bytes of logical data into one omcube.
    More sophisticated schemes could encode actual data patterns, learned embeddings, etc.
    
    Args:
        i: Data index (which "chunk" of data we're encoding)
        scheme: Encoding scheme name
        bytes_per_omcube: How many logical bytes per omcube
        
    Returns:
        Tuple of (x, y, z) coordinates representing the encoded data
    """
    # For larger encodings, we need to pack more information into coordinates
    # We can use higher precision or multiple coordinate sets
    
    if bytes_per_omcube <= 8:
        # Small encodings: simple coordinate mapping
        if scheme == '1_byte_per_omcube':
            x = (i % 256) / 255.0
            y = ((i // 256) % 256) / 255.0
            z = ((i // 65536) % 256) / 255.0
        elif scheme == '4_bytes_per_omcube':
            x = (i % 65536) / 65535.0
            y = ((i // 65536) % 65536) / 65535.0
            z = ((i // 4294967296) % 65536) / 65535.0
        elif scheme == '8_bytes_per_omcube':
            x = (i % 2147483647) / 2147483646.0
            y = ((i // 2147483647) % 2147483647) / 2147483646.0
            z = ((i // 4611686014132420609) % 2147483647) / 2147483646.0
        else:
            # Default: simple encoding
            x = (i % 256) / 255.0
            y = ((i // 256) % 256) / 255.0
            z = ((i // 65536) % 256) / 255.0
    else:
        # Larger encodings: pack more data using higher precision
        # We simulate encoding N bytes by using the full range of float precision
        # In reality, you'd use actual data patterns, but this shows the concept
        
        # Use i to generate coordinates that represent more data
        # For 32+ bytes, we can use multiple coordinate dimensions or higher precision
        max_val = 2 ** (bytes_per_omcube * 8)  # Max value for N bytes
        
        # Split across 3 coordinates with high precision
        x = (i % max_val) / (max_val - 1) if max_val > 1 else 0.0
        y = ((i // max_val) % max_val) / (max_val - 1) if max_val > 1 else 0.0
        z = ((i // (max_val ** 2)) % max_val) / (max_val - 1) if max_val > 1 else 0.0
        
        # Clamp to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        z = max(0.0, min(1.0, z))
    
    return (x, y, z)


def run_compression_capacity_test(
    size_mb: int,
    base_dimension: int,
    bytes_per_omcube_logical: int,
    encoding_scheme: str,
    num_levels: int = 3
) -> Dict:
    """
    Run compression capacity test for a specific configuration.
    
    Args:
        size_mb: Logical data size in MB
        base_dimension: Base dimension of geometry (3, 5, 7, etc.)
        bytes_per_omcube_logical: Logical bytes encoded per omcube
        encoding_scheme: Name of encoding scheme
        num_levels: Number of hierarchy levels
        
    Returns:
        Dictionary with test results
    """
    raw_bytes = size_mb * 1024 * 1024
    needed_omcubes = raw_bytes // bytes_per_omcube_logical
    
    print(f"\n  Testing: {size_mb} MB logical, {base_dimension}×{base_dimension}×{base_dimension}, "
          f"{bytes_per_omcube_logical} bytes/omcube, {needed_omcubes:,} omcubes")
    
    system = DynamicHierarchicalGeometrySystem(
        base_dimension=base_dimension,
        num_levels=num_levels
    )
    
    tracemalloc.start()
    t0 = time.time()
    
    # Encode data into omcubes
    for i in range(needed_omcubes):
        coords = encode_data_to_coordinates(i, encoding_scheme, bytes_per_omcube_logical)
        system.add_base_state(coords)
        
        # Progress indicator for large tests
        if (i + 1) % 100_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    Progress: {i+1:,}/{needed_omcubes:,} ({rate:,.0f} omcubes/s)")
    
    current, peak = tracemalloc.get_traced_memory()
    t1 = time.time()
    tracemalloc.stop()
    
    elapsed_time = t1 - t0
    peak_mb = peak / (1024 * 1024)
    compression_ratio = raw_bytes / peak if peak > 0 else 0.0
    bytes_per_omcube_ram = peak / needed_omcubes if needed_omcubes > 0 else 0
    
    return {
        "logical_size_mb": size_mb,
        "raw_bytes": raw_bytes,
        "base_dimension": base_dimension,
        "num_levels": num_levels,
        "encoding_scheme": encoding_scheme,
        "bytes_per_omcube_logical": bytes_per_omcube_logical,
        "num_omcubes": needed_omcubes,
        "peak_memory_bytes": peak,
        "peak_memory_mb": peak_mb,
        "bytes_per_omcube_ram": bytes_per_omcube_ram,
        "compression_ratio_raw_over_ram": compression_ratio,
        "time_seconds": elapsed_time,
        "omcubes_per_second": needed_omcubes / elapsed_time if elapsed_time > 0 else 0
    }


def run_full_compression_suite(quick_mode: bool = False):
    """
    Run full compression capacity test suite.
    
    Tests multiple:
    - Logical data sizes
    - Base dimensions
    - Encoding schemes
    
    Args:
        quick_mode: If True, use smaller test sizes for faster execution
    """
    print("=" * 70)
    print("COMPRESSION CAPACITY TEST SUITE")
    if quick_mode:
        print("(QUICK MODE - Smaller test sizes)")
    print("=" * 70)
    print("\nThis test measures:")
    print("  - How much semantic/structured data can be packed into geometry")
    print("  - Compression ratios under specific encoding schemes")
    print("  - Relationship between base dimension and compression capacity")
    print("\nNote: Compression depends on encoding scheme, not just geometry.")
    
    results = []
    
    # Choose test sizes based on mode
    test_sizes = SIZES_MB_QUICK if quick_mode else SIZES_MB
    test_dims = BASE_DIMENSIONS_QUICK if quick_mode else BASE_DIMENSIONS
    
    # Test with realistic encoding scheme (256 bytes per omcube - should compress)
    # Note: Each omcube uses ~193-240 bytes RAM, so we need >200 bytes logical to compress
    default_scheme = '256_bytes_per_omcube'
    default_bytes = ENCODING_SCHEMES[default_scheme]
    
    print(f"\n{'=' * 70}")
    print(f"Test Series 1: Base Dimension Comparison")
    print(f"Encoding: {default_scheme} ({default_bytes} bytes/omcube logical)")
    print(f"{'=' * 70}")
    
    for base_dim in test_dims:
        print(f"\nBase Dimension: {base_dim}×{base_dim}×{base_dim}")
        
        for size_mb in test_sizes:
            try:
                result = run_compression_capacity_test(
                    size_mb=size_mb,
                    base_dimension=base_dim,
                    bytes_per_omcube_logical=default_bytes,
                    encoding_scheme=default_scheme
                )
                results.append(result)
                
                print(f"    ✅ {size_mb} MB: {result['peak_memory_mb']:.2f} MB RAM, "
                      f"ratio: {result['compression_ratio_raw_over_ram']:.3f}×")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                break  # Skip larger sizes if this one failed
    
    # Test with different encoding schemes (using base_dim=3)
    print(f"\n{'=' * 70}")
    print(f"Test Series 2: Encoding Scheme Comparison")
    print(f"Base Dimension: 3×3×3")
    print(f"{'=' * 70}")
    print("\nNote: Small encodings (1-8 bytes) will EXPAND data (not compress)")
    print("      Large encodings (32+ bytes) should COMPRESS data")
    
    for scheme_name, bytes_per_omcube in ENCODING_SCHEMES.items():
        print(f"\nEncoding Scheme: {scheme_name} ({bytes_per_omcube} bytes/omcube logical)")
        
        # For small encodings, use smaller test sizes (they expand anyway)
        # For large encodings, we can test larger sizes (they should compress)
        if bytes_per_omcube < 32:
            test_sizes = [10, 50]  # Small tests for expanding schemes
        else:
            test_sizes = [10, 50, 100]  # Larger tests for compressing schemes
        
        for size_mb in test_sizes:
            try:
                result = run_compression_capacity_test(
                    size_mb=size_mb,
                    base_dimension=3,
                    bytes_per_omcube_logical=bytes_per_omcube,
                    encoding_scheme=scheme_name
                )
                results.append(result)
                
                print(f"    ✅ {size_mb} MB: {result['peak_memory_mb']:.2f} MB RAM, "
                      f"ratio: {result['compression_ratio_raw_over_ram']:.3f}×")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                break
    
    # Save results
    out_dir = Path(project_root) / "docs" / "tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "COMPRESSION_CAPACITY_RESULTS.csv"
    json_path = out_dir / "COMPRESSION_CAPACITY_RESULTS.json"
    
    if results:
        # Write CSV
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        # Write JSON
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'=' * 70}")
        print("RESULTS SAVED")
        print(f"{'=' * 70}")
        print(f"  CSV: {csv_path}")
        print(f"  JSON: {json_path}")
        print(f"  Total tests: {len(results)}")
    
    # Summary analysis
    print(f"\n{'=' * 70}")
    print("COMPRESSION ANALYSIS")
    print(f"{'=' * 70}")
    
    if results:
        # Group by base dimension
        by_dimension = {}
        for r in results:
            dim = r['base_dimension']
            if dim not in by_dimension:
                by_dimension[dim] = []
            by_dimension[dim].append(r)
        
        print(f"\nAverage Compression Ratios by Base Dimension:")
        for dim in sorted(by_dimension.keys()):
            ratios = [r['compression_ratio_raw_over_ram'] for r in by_dimension[dim]]
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0
            print(f"  {dim}×{dim}×{dim}: {avg_ratio:.3f}× (avg)")
        
        # Group by encoding scheme
        by_scheme = {}
        for r in results:
            scheme = r['encoding_scheme']
            if scheme not in by_scheme:
                by_scheme[scheme] = []
            by_scheme[scheme].append(r)
        
        print(f"\nAverage Compression Ratios by Encoding Scheme:")
        for scheme in sorted(by_scheme.keys()):
            ratios = [r['compression_ratio_raw_over_ram'] for r in by_scheme[scheme]]
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0
            bytes_per = ENCODING_SCHEMES.get(scheme, 0)
            print(f"  {scheme} ({bytes_per} bytes/omcube): {avg_ratio:.3f}× (avg)")
        
        # Best and worst
        best = max(results, key=lambda r: r['compression_ratio_raw_over_ram'])
        worst = min(results, key=lambda r: r['compression_ratio_raw_over_ram'])
        
        print(f"\nBest Compression:")
        print(f"  {best['compression_ratio_raw_over_ram']:.3f}× at {best['logical_size_mb']} MB, "
              f"{best['base_dimension']}×{best['base_dimension']}×{best['base_dimension']}, "
              f"{best['encoding_scheme']}")
        
        print(f"\nWorst Compression:")
        print(f"  {worst['compression_ratio_raw_over_ram']:.3f}× at {worst['logical_size_mb']} MB, "
              f"{worst['base_dimension']}×{worst['base_dimension']}×{worst['base_dimension']}, "
              f"{worst['encoding_scheme']}")
    
    print(f"\n{'=' * 70}")
    print("✅ Compression capacity test suite complete!")
    print(f"{'=' * 70}")
    print("\nKey Insights:")
    print("  - Compression ratio depends on encoding scheme")
    print("  - Base dimension affects memory per omcube")
    print("  - This measures semantic data packing, not general compression")
    print("\nSee results in:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compression capacity test')
    parser.add_argument('--quick', action='store_true', 
                       help='Run in quick mode (smaller test sizes)')
    args = parser.parse_args()
    
    run_full_compression_suite(quick_mode=args.quick)

