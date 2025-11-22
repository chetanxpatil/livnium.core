"""
Test Dynamic Basin Physics

Tests the physics of dynamic basin search:
- Drift stays bounded
- Noise does not accumulate
- Reinforcement actually deepens basins
- Decay does not collapse everything
- Tension never goes negative
- Entropy stays finite
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from core.classical.livnium_core_system import LivniumCoreSystem
from core.search.native_dynamic_basin_search import (
    compute_local_curvature,
    compute_symbolic_tension,
    compute_noise_entropy,
    update_basin_dynamic
)
from core.config import LivniumCoreConfig


def test_drift_bounded():
    """Test that drift stays bounded."""
    print("=" * 60)
    print("Test 1: Drift Boundedness")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    active_coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    
    # Track SW over iterations
    initial_sw = system.get_total_symbolic_weight()
    sw_history = [initial_sw]
    
    # Run dynamic basin updates
    for iteration in range(100):
        # Simulate correct answer (reinforcement)
        update_basin_dynamic(system, active_coords, is_correct=True)
        
        current_sw = system.get_total_symbolic_weight()
        sw_history.append(current_sw)
    
    # Check drift
    sw_drift = max(sw_history) - min(sw_history)
    expected_sw = system.get_expected_total_sw()
    relative_drift = sw_drift / expected_sw if expected_sw > 0 else 0
    
    print(f"SW drift: {sw_drift:.6f}")
    print(f"Relative drift: {relative_drift * 100:.2f}%")
    print(f"Drift Bounded: {'✅ PASS' if relative_drift < 0.1 else '❌ FAIL'}")
    
    # Drift should be small relative to total SW
    assert relative_drift < 0.1, "Drift must stay bounded (< 10% of total SW)"
    
    print("\n✅ Drift boundedness test passed!")


def test_noise_no_accumulation():
    """Test that noise does not accumulate."""
    print("\n" + "=" * 60)
    print("Test 2: Noise Non-Accumulation")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    active_coords = [(0, 0, 0), (1, 0, 0)]
    
    # Track entropy (noise proxy)
    entropy_history = []
    
    for iteration in range(50):
        # Mix correct and incorrect updates
        is_correct = (iteration % 3 == 0)  # 1/3 correct
        update_basin_dynamic(system, active_coords, is_correct=is_correct)
        
        entropy = compute_noise_entropy(system, active_coords)
        entropy_history.append(entropy)
    
    # Check entropy doesn't explode
    max_entropy = max(entropy_history)
    mean_entropy = np.mean(entropy_history)
    
    print(f"Max entropy: {max_entropy:.6f}")
    print(f"Mean entropy: {mean_entropy:.6f}")
    print(f"Noise Bounded: {'✅ PASS' if max_entropy < 10.0 else '❌ FAIL'}")
    
    assert max_entropy < 10.0, "Entropy (noise) must not accumulate unbounded"
    
    # Check entropy variance (should stabilize)
    early_entropy = entropy_history[:10]
    late_entropy = entropy_history[-10:]
    early_var = np.var(early_entropy)
    late_var = np.var(late_entropy)
    
    print(f"Early variance: {early_var:.6f}")
    print(f"Late variance: {late_var:.6f}")
    print(f"Variance Stable: {'✅ PASS' if late_var <= early_var * 2 else '❌ FAIL'}")
    
    print("\n✅ Noise non-accumulation test passed!")


def test_reinforcement_deepens():
    """Test that reinforcement actually deepens basins."""
    print("\n" + "=" * 60)
    print("Test 3: Reinforcement Deepens Basins")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    active_coords = [(0, 0, 0), (1, 0, 0)]
    
    # Initial curvature
    initial_curvature = compute_local_curvature(system, active_coords)
    initial_sw = [system.get_cell(c).symbolic_weight for c in active_coords]
    
    print(f"Initial curvature: {initial_curvature:.6f}")
    print(f"Initial SW: {[f'{sw:.2f}' for sw in initial_sw]}")
    
    # Apply reinforcement
    for iteration in range(20):
        update_basin_dynamic(system, active_coords, is_correct=True)
    
    # Final curvature
    final_curvature = compute_local_curvature(system, active_coords)
    final_sw = [system.get_cell(c).symbolic_weight for c in active_coords]
    
    print(f"Final curvature: {final_curvature:.6f}")
    print(f"Final SW: {[f'{sw:.2f}' for sw in final_sw]}")
    
    # Curvature should increase (basin deepens)
    curvature_increased = final_curvature > initial_curvature
    sw_increased = all(f > i for f, i in zip(final_sw, initial_sw))
    
    print(f"Curvature increased: {'✅' if curvature_increased else '❌'}")
    print(f"SW increased: {'✅' if sw_increased else '❌'}")
    
    assert curvature_increased or sw_increased, "Reinforcement should deepen basin"
    
    print("\n✅ Reinforcement deepens basins test passed!")


def test_decay_no_collapse():
    """Test that decay does not collapse everything."""
    print("\n" + "=" * 60)
    print("Test 4: Decay Does Not Collapse")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    active_coords = [(0, 0, 0), (1, 0, 0)]
    
    # Set initial SW
    for coords in active_coords:
        cell = system.get_cell(coords)
        if cell:
            cell.symbolic_weight = 50.0  # High initial SW
    
    initial_sw = [system.get_cell(c).symbolic_weight for c in active_coords]
    
    # Apply decay (incorrect updates)
    for iteration in range(100):
        update_basin_dynamic(system, active_coords, is_correct=False)
    
    final_sw = [system.get_cell(c).symbolic_weight for c in active_coords]
    
    print(f"Initial SW: {[f'{sw:.2f}' for sw in initial_sw]}")
    print(f"Final SW: {[f'{sw:.2f}' for sw in final_sw]}")
    
    # SW should not go to zero (collapse)
    all_positive = all(sw > 0 for sw in final_sw)
    not_collapsed = any(sw > 1.0 for sw in final_sw)  # At least one cell retains SW
    
    print(f"All SW positive: {'✅' if all_positive else '❌'}")
    print(f"Not collapsed: {'✅' if not_collapsed else '❌'}")
    
    assert all_positive, "SW must stay positive"
    assert not_collapsed, "Decay should not collapse all SW"
    
    print("\n✅ Decay no collapse test passed!")


def test_tension_non_negative():
    """Test that tension never goes negative."""
    print("\n" + "=" * 60)
    print("Test 5: Tension Non-Negative")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    active_coords = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    
    tension_history = []
    
    # Run many updates
    for iteration in range(100):
        is_correct = (iteration % 2 == 0)
        update_basin_dynamic(system, active_coords, is_correct=is_correct)
        
        tension = compute_symbolic_tension(system, active_coords)
        tension_history.append(tension)
        
        if tension < 0:
            print(f"⚠️  Negative tension at iteration {iteration}: {tension:.6f}")
    
    min_tension = min(tension_history)
    mean_tension = np.mean(tension_history)
    
    print(f"Min tension: {min_tension:.6f}")
    print(f"Mean tension: {mean_tension:.6f}")
    print(f"Tension Non-Negative: {'✅ PASS' if min_tension >= 0 else '❌ FAIL'}")
    
    assert min_tension >= 0, "Tension must never be negative"
    
    print("\n✅ Tension non-negative test passed!")


def test_entropy_finite():
    """Test that entropy stays finite."""
    print("\n" + "=" * 60)
    print("Test 6: Entropy Finiteness")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    active_coords = [(0, 0, 0), (1, 0, 0)]
    
    entropy_history = []
    
    for iteration in range(100):
        update_basin_dynamic(system, active_coords, is_correct=(iteration % 3 == 0))
        
        entropy = compute_noise_entropy(system, active_coords)
        entropy_history.append(entropy)
        
        if not np.isfinite(entropy):
            print(f"⚠️  Non-finite entropy at iteration {iteration}: {entropy}")
    
    max_entropy = max(entropy_history)
    all_finite = all(np.isfinite(e) for e in entropy_history)
    
    print(f"Max entropy: {max_entropy:.6f}")
    print(f"All finite: {'✅' if all_finite else '❌'}")
    print(f"Entropy Finite: {'✅ PASS' if all_finite and max_entropy < 1e6 else '❌ FAIL'}")
    
    assert all_finite, "Entropy must be finite"
    assert max_entropy < 1e6, "Entropy must not explode"
    
    print("\n✅ Entropy finiteness test passed!")


if __name__ == "__main__":
    test_drift_bounded()
    test_noise_no_accumulation()
    test_reinforcement_deepens()
    test_decay_no_collapse()
    test_tension_non_negative()
    test_entropy_finite()
    print("\n" + "=" * 60)
    print("All dynamic basin physics tests passed! ✅")
    print("=" * 60)

