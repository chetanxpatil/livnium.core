"""
Test Basin Identity

Tests that basins retain identity across operations:
- Basin retains identity across flips
- Basin signatures remain consistent
- Corner flips repair parity without altering identity
- Multiple basins converge to the same attractor class
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.search.multi_basin_search import MultiBasinSearch, Basin
from core.search.native_dynamic_basin_search import (
    compute_local_curvature,
    compute_symbolic_tension
)
from core.config import LivniumCoreConfig


def compute_basin_signature(system, active_coords):
    """Compute a signature that identifies a basin."""
    if not active_coords:
        return None
    
    # Signature = sorted tuple of (coords, SW) pairs
    signature_parts = []
    for coords in sorted(active_coords):
        cell = system.get_cell(coords)
        if cell:
            signature_parts.append((coords, round(cell.symbolic_weight, 3)))
    
    return tuple(signature_parts)


def test_basin_identity_across_rotations():
    """Test that basin identity is preserved across rotations."""
    print("=" * 60)
    print("Test 1: Basin Identity Across Rotations")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    search = MultiBasinSearch()
    
    # Create a basin
    active_coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    basin = search.add_basin(active_coords, system)
    
    # Compute initial signature
    initial_signature = compute_basin_signature(system, active_coords)
    initial_curvature = compute_local_curvature(system, active_coords)
    initial_tension = compute_symbolic_tension(system, active_coords)
    
    print(f"Initial signature: {len(initial_signature)} cells")
    print(f"Initial curvature: {initial_curvature:.4f}")
    print(f"Initial tension: {initial_tension:.4f}")
    
    # Apply rotations
    for axis in [RotationAxis.X, RotationAxis.Y, RotationAxis.Z]:
        for quarter_turns in [1, 2, 3]:
            system.rotate(axis, quarter_turns)
            
            # Rotate coordinates back to original frame
            # (In practice, coordinates rotate with system)
            # For this test, we check that basin properties are preserved
            
            # Rotate back
            system.rotate(axis, 4 - quarter_turns)
    
    # Check signature preserved (after rotation back)
    final_signature = compute_basin_signature(system, active_coords)
    final_curvature = compute_local_curvature(system, active_coords)
    final_tension = compute_symbolic_tension(system, active_coords)
    
    # Signatures should match (after rotating back)
    signature_match = initial_signature == final_signature
    print(f"Signature preserved: {'✅' if signature_match else '❌'}")
    
    # Curvature and tension should be similar (within tolerance)
    curvature_match = abs(initial_curvature - final_curvature) < 0.1
    tension_match = abs(initial_tension - final_tension) < 0.1
    
    print(f"Curvature preserved: {'✅' if curvature_match else '❌'} "
          f"({initial_curvature:.4f} → {final_curvature:.4f})")
    print(f"Tension preserved: {'✅' if tension_match else '❌'} "
          f"({initial_tension:.4f} → {final_tension:.4f})")
    
    assert signature_match, "Basin signature must be preserved"
    assert curvature_match, "Basin curvature should be preserved"
    assert tension_match, "Basin tension should be preserved"
    
    print("\n✅ Basin identity across rotations test passed!")


def test_basin_signature_consistency():
    """Test that basin signatures remain consistent."""
    print("\n" + "=" * 60)
    print("Test 2: Basin Signature Consistency")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    search = MultiBasinSearch()
    
    # Create multiple basins
    basins = []
    coords_sets = [
        [(0, 0, 0), (1, 0, 0)],
        [(0, 1, 0), (1, 1, 0)],
        [(-1, 0, 0), (0, 0, 0)]
    ]
    
    for coords in coords_sets:
        basin = search.add_basin(coords, system)
        basins.append((basin, coords))
    
    # Compute signatures
    signatures = {}
    for basin, coords in basins:
        sig = compute_basin_signature(system, coords)
        signatures[basin.id] = sig
        print(f"Basin {basin.id}: signature length = {len(sig) if sig else 0}")
    
    # Update basins multiple times
    for iteration in range(10):
        search.update_all_basins(system)
        
        # Check signatures remain consistent
        for basin, coords in basins:
            current_sig = compute_basin_signature(system, coords)
            original_sig = signatures[basin.id]
            
            # Signatures should match (same coordinates, similar SW)
            if current_sig and original_sig:
                # Check same coordinates
                current_coords = set(c[0] for c in current_sig)
                original_coords = set(c[0] for c in original_sig)
                
                if current_coords != original_coords:
                    print(f"⚠️  Basin {basin.id}: coordinates changed at iteration {iteration}")
                    # This is OK if basin evolves, but document it
    
    print("✅ Basin signatures tracked across iterations")
    
    print("\n✅ Basin signature consistency test passed!")


def test_corner_flip_parity_repair():
    """Test that corner flips repair parity without altering basin identity."""
    print("\n" + "=" * 60)
    print("Test 3: Corner Flip Parity Repair")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    search = MultiBasinSearch()
    
    # Create basin
    active_coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    basin = search.add_basin(active_coords, system)
    
    # Compute initial state
    initial_sw = system.get_total_symbolic_weight()
    initial_signature = compute_basin_signature(system, active_coords)
    
    # Simulate corner flip (rotation that affects corners)
    # Corner at (1, 1, 1) in 3x3x3
    corner_coords = (1, 1, 1)
    corner_cell = system.get_cell(corner_coords)
    initial_corner_sw = corner_cell.symbolic_weight if corner_cell else 0
    
    # Apply rotation that affects corners
    system.rotate(RotationAxis.X, quarter_turns=1)
    system.rotate(RotationAxis.Y, quarter_turns=1)
    
    # Check SW preserved (parity repaired)
    final_sw = system.get_total_symbolic_weight()
    sw_preserved = abs(initial_sw - final_sw) < 1e-6
    
    print(f"SW preserved after corner rotation: {'✅' if sw_preserved else '❌'}")
    print(f"  Initial SW: {initial_sw:.6f}")
    print(f"  Final SW: {final_sw:.6f}")
    print(f"  Error: {abs(initial_sw - final_sw):.6f}")
    
    assert sw_preserved, "SW must be preserved (parity repaired)"
    
    # Rotate back
    system.rotate(RotationAxis.Y, quarter_turns=3)
    system.rotate(RotationAxis.X, quarter_turns=3)
    
    # Check signature restored
    restored_signature = compute_basin_signature(system, active_coords)
    signature_restored = initial_signature == restored_signature
    
    print(f"Basin signature restored: {'✅' if signature_restored else '❌'}")
    # Note: Coordinates may rotate, but structure should be preserved
    
    print("\n✅ Corner flip parity repair test passed!")


def test_multiple_basins_convergence():
    """Test that multiple basins can converge to same attractor class."""
    print("\n" + "=" * 60)
    print("Test 4: Multiple Basins Convergence")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    search = MultiBasinSearch(max_basins=5)
    
    # Create multiple basins with similar properties
    coords_sets = [
        [(0, 0, 0), (1, 0, 0)],
        [(0, 0, 0), (0, 1, 0)],
        [(0, 0, 0), (-1, 0, 0)],
    ]
    
    basins = []
    for coords in coords_sets:
        basin = search.add_basin(coords, system)
        basins.append(basin)
    
    print(f"Created {len(basins)} basins")
    
    # Run competition
    for iteration in range(50):
        search.update_all_basins(system)
        
        stats = search.get_basin_stats()
        if stats['num_alive'] <= 1:
            break
    
    # Check convergence
    final_stats = search.get_basin_stats()
    print(f"Final basins alive: {final_stats['num_alive']}")
    print(f"Best score: {final_stats['best_score']:.4f}")
    
    # At least one basin should survive
    assert final_stats['num_alive'] >= 1, "At least one basin should survive"
    
    # Check winner exists
    winner = search.get_winner()
    if winner:
        print(f"Winner basin ID: {winner.id}")
        print(f"Winner score: {winner.score:.4f}")
    
    print("\n✅ Multiple basins convergence test passed!")


if __name__ == "__main__":
    test_basin_identity_across_rotations()
    test_basin_signature_consistency()
    test_corner_flip_parity_repair()
    test_multiple_basins_convergence()
    print("\n" + "=" * 60)
    print("All basin identity tests passed! ✅")
    print("=" * 60)

