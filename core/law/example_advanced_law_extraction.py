#!/usr/bin/env python3
"""
Example: Advanced Law Extraction (v2-v6)

Demonstrates all advanced law extraction features:
- v2: Nonlinear function discovery
- v3: Symbolic regression
- v4: Law stability + confidence scoring
- v5: Multi-layer law fusion
- v6: Basin-based law extraction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import random

from core.runtime.orchestrator import Orchestrator
from core.classical.livnium_core_system import LivniumCoreSystem, RotationAxis
from core.config import LivniumCoreConfig
from core.search.multi_basin_search import MultiBasinSearch, create_candidate_basins
from core.search.native_dynamic_basin_search import update_basin_dynamic
from core.law.advanced_law_extractor import AdvancedLawExtractor


class SimpleTask:
    """Simple task that changes system state."""
    def __init__(self, system, rng):
        self.system = system
        self.rng = rng
        coords_list = list(system.lattice.keys())
        self.input_coords = [coords_list[i] for i in rng.choice(len(coords_list), size=3, replace=False)]
        remaining = [c for c in coords_list if c not in self.input_coords]
        self.output_coord = remaining[rng.integers(0, len(remaining))] if remaining else coords_list[0]
        self.target_input = [rng.integers(0, 2) for _ in range(3)]
        self.target_output = sum(self.target_input) % 2
        
        # Encode input
        for coords, bit in zip(self.input_coords, self.target_input):
            cell = system.get_cell(coords)
            if cell:
                cell.symbolic_weight = 20.0 if bit == 1 else 10.0
    
    def is_correct(self):
        output_cell = self.system.get_cell(self.output_coord)
        if output_cell:
            answer = 1 if output_cell.symbolic_weight > 15.0 else 0
            return answer == self.target_output
        return False


def main():
    """Run advanced law extraction example."""
    print("=" * 70)
    print("Advanced Law Extractor - v2-v6 Features")
    print("=" * 70)
    print()
    
    # Step 1: Create system
    print("Step 1: Creating Livnium Core System...")
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_symbolic_weight=True,
        enable_90_degree_rotations=True,
        enable_semantic_polarity=True
    )
    system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(system)
    
    # Replace with advanced extractor
    orchestrator.law_extractor = AdvancedLawExtractor(
        min_confidence=0.6,
        stability_window=20
    )
    
    print("✓ System created with AdvancedLawExtractor")
    print()
    
    # Step 2: Evolve system
    print("Step 2: Evolving system for 150 timesteps...")
    print("(Applying rotations, basin updates, dynamic forces)")
    num_steps = 150
    rng = np.random.Generator(np.random.PCG64(42))
    
    # Create multi-basin search
    search = MultiBasinSearch(max_basins=5)
    candidates = create_candidate_basins(system, n_candidates=3, basin_size=4)
    basins = {}
    for i, coords in enumerate(candidates):
        basin = search.add_basin(coords, system)
        basins[f"basin_{basin.id}"] = []
    
    for i in range(num_steps):
        # Apply rotations
        if i % 5 == 0 and config.enable_90_degree_rotations:
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        
        # Apply basin dynamics
        if i % 3 == 0:
            search.update_all_basins(system)
            
            # Record basin-specific states (v6)
            for basin in search.basins:
                if basin.is_alive:
                    basin_id = f"basin_{basin.id}"
                    # Get state for this basin's coordinates
                    basin_state = {}
                    for coords in basin.active_coords:
                        cell = system.get_cell(coords)
                        if cell:
                            basin_state[f"SW_{coords}"] = cell.symbolic_weight
                    if basin_state:
                        basins[basin_id].append(basin_state)
        
        # Apply dynamic basin updates
        if i % 7 == 0:
            task = SimpleTask(system, rng)
            is_correct = task.is_correct()
            update_basin_dynamic(system, task, is_correct)
        
        # Record state (v5: with layer info)
        layer = 0  # Base layer
        orchestrator.step()
        state = system.export_physics_state()
        orchestrator.law_extractor.record_state(state, layer=layer)
        
        if (i + 1) % 30 == 0:
            print(f"  Step {i+1}: SW_sum={state['SW_sum']:.2f}, "
                  f"curvature={state['curvature']:.2f}, tension={state['tension']:.2f}")
    
    print("✓ System evolution complete")
    print()
    
    # Step 3: Extract all laws
    print("Step 3: Extracting all types of laws (v1-v6)...")
    all_laws = orchestrator.law_extractor.extract_all()
    print("✓ Laws extracted")
    print()
    
    # Step 4: Display results
    print("=" * 70)
    print("DISCOVERED LAWS (All Versions)")
    print("=" * 70)
    print()
    
    # v1: Invariants
    print("v1: INVARIANTS (Conserved Quantities)")
    print("-" * 70)
    invariants = all_laws["invariants"]
    for name, is_invariant in invariants.items():
        if is_invariant:
            values = [h[name] for h in orchestrator.law_extractor.history if name in h]
            if values:
                print(f"  ✓ {name}: {values[0]:.6f} (constant)")
    print()
    
    # v1: Linear relationships
    print("v1: LINEAR RELATIONSHIPS")
    print("-" * 70)
    linear = all_laws["linear_relationships"]
    for name, (a, b) in list(linear.items())[:5]:  # Show first 5
        y_name, x_name = name.split("_vs_")
        if abs(b) < 1e-6:
            print(f"  {y_name} = {a:.6f} * {x_name}")
        else:
            sign = "+" if b >= 0 else ""
            print(f"  {y_name} = {a:.6f} * {x_name} {sign}{b:.6f}")
    if len(linear) > 5:
        print(f"  ... and {len(linear) - 5} more")
    print()
    
    # v2: Nonlinear relationships
    print("v2: NONLINEAR RELATIONSHIPS")
    print("-" * 70)
    nonlinear = all_laws["nonlinear_relationships"]
    if nonlinear:
        for name, (formula, func, error) in list(nonlinear.items())[:5]:
            print(f"  {formula} (error: {error:.6f})")
    else:
        print("  (No strong nonlinear relationships detected)")
    print()
    
    # v3: Symbolic expressions
    print("v3: SYMBOLIC EXPRESSIONS")
    print("-" * 70)
    symbolic = all_laws["symbolic_expressions"]
    if symbolic:
        for name, (formula, func, error) in list(symbolic.items())[:5]:
            print(f"  {formula} (error: {error:.6f})")
    else:
        print("  (No symbolic expressions detected)")
    print()
    
    # v4: Discovered laws with confidence/stability
    print("v4: DISCOVERED LAWS (with Confidence & Stability)")
    print("-" * 70)
    discovered = all_laws["discovered_laws"]
    if discovered:
        # Sort by confidence
        sorted_laws = sorted(discovered.items(), key=lambda x: x[1]["confidence"], reverse=True)
        for name, law_info in sorted_laws[:10]:
            print(f"  {law_info['formula']}")
            print(f"    Confidence: {law_info['confidence']:.4f}, "
                  f"Stability: {law_info['stability']:.4f}")
    else:
        print("  (No high-confidence laws yet - need more evolution)")
    print()
    
    # v5: Fused laws
    print("v5: FUSED LAWS (Multi-Layer)")
    print("-" * 70)
    fused = all_laws["fused_laws"]
    if fused:
        for name, law_info in fused.items():
            print(f"  {law_info['formula']}")
            print(f"    Confidence: {law_info['confidence']:.4f}, "
                  f"Stability: {law_info['stability']:.4f}")
    else:
        print("  (No multi-layer laws - single layer system)")
    print()
    
    # v6: Basin laws
    print("v6: BASIN-SPECIFIC LAWS")
    print("-" * 70)
    # Extract basin laws
    basin_laws = orchestrator.law_extractor.extract_basin_laws(basins)
    if basin_laws:
        for basin_id, laws in basin_laws.items():
            print(f"  Basin {basin_id}:")
            for name, law in list(laws.items())[:3]:
                print(f"    {law.formula} (confidence: {law.confidence:.4f})")
    else:
        print("  (No basin-specific laws detected)")
    print()
    
    print("=" * 70)
    print("Summary:")
    print(f"  - Invariants: {sum(1 for v in invariants.values() if v)}")
    print(f"  - Linear relationships: {len(linear)}")
    print(f"  - Nonlinear relationships: {len(nonlinear)}")
    print(f"  - Symbolic expressions: {len(symbolic)}")
    print(f"  - High-confidence laws: {len(discovered)}")
    print(f"  - Fused laws: {len(fused)}")
    print(f"  - Basin laws: {sum(len(laws) for laws in basin_laws.values())}")
    print("=" * 70)


if __name__ == "__main__":
    main()

