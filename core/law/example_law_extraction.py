#!/usr/bin/env python3
"""
Example: How to Run Law Extraction on an Evolving System

This script demonstrates how to use the law extractor to discover
physical laws from Livnium system behavior.

IMPORTANT: The system must actually EVOLVE for laws to emerge.
A static system will only show invariants, not relationships.
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
    """Run law extraction example with evolving system."""
    print("=" * 60)
    print("Livnium Law Extractor - Evolving System Example")
    print("=" * 60)
    print()
    print("This example evolves the system so real laws can emerge.")
    print("A static system only shows invariants, not relationships.")
    print()
    
    # Step 1: Create system with features enabled
    print("Step 1: Creating Livnium Core System...")
    config = LivniumCoreConfig(
        lattice_size=3,
        enable_symbolic_weight=True,
        enable_90_degree_rotations=True,
        enable_semantic_polarity=True
    )
    system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(system)
    print("✓ System created")
    print()
    
    # Step 2: Evolve system with actual changes
    print("Step 2: Evolving system for 100 timesteps...")
    print("(Applying rotations, basin updates, and dynamic forces)")
    num_steps = 100
    rng = np.random.Generator(np.random.PCG64(42))
    
    # Create multi-basin search for competition
    search = MultiBasinSearch(max_basins=5)
    candidates = create_candidate_basins(system, n_candidates=3, basin_size=4)
    for coords in candidates:
        search.add_basin(coords, system)
    
    for i in range(num_steps):
        # Apply rotations (changes geometry)
        if i % 5 == 0 and config.enable_90_degree_rotations:
            axis = random.choice(list(RotationAxis))
            system.rotate(axis, quarter_turns=random.choice([1, 2, 3]))
        
        # Apply basin dynamics (changes SW, curvature, tension)
        if i % 3 == 0:
            search.update_all_basins(system)
        
        # Apply dynamic basin updates (changes SW based on tasks)
        if i % 7 == 0:
            task = SimpleTask(system, rng)
            is_correct = task.is_correct()
            update_basin_dynamic(system, task, is_correct)
        
        # Record state via orchestrator
        orchestrator.step()
        
        if (i + 1) % 20 == 0:
            state = system.export_physics_state()
            print(f"  Step {i+1}: SW_sum={state['SW_sum']:.2f}, "
                  f"curvature={state['curvature']:.2f}, tension={state['tension']:.2f}")
    
    print("✓ System evolution complete")
    print()
    
    # Step 3: Extract discovered laws
    print("Step 3: Extracting discovered laws...")
    laws = orchestrator.extract_laws()
    print("✓ Laws extracted")
    print()
    
    # Step 4: Display results
    print("=" * 60)
    print("DISCOVERED LAWS")
    print("=" * 60)
    print()
    
    # Print summary
    summary = orchestrator.get_law_summary()
    print(summary)
    
    # Print raw data
    print("=" * 60)
    print("RAW DATA")
    print("=" * 60)
    print()
    
    print("Invariants (Conserved Quantities):")
    invariants_found = False
    for name, is_invariant in laws['invariants'].items():
        if is_invariant:
            status = "✓ CONSERVED"
            invariants_found = True
            # Get actual value
            values = [h[name] for h in orchestrator.law_extractor.history if name in h]
            if values:
                print(f"  {name}: {status} (value: {values[0]:.6f})")
    
    if not invariants_found:
        print("  (No strict invariants - system is evolving)")
    
    print()
    print("Functional Relationships:")
    if laws['relationships']:
        # Filter to show only strong relationships
        strong_rels = []
        for rel_name, (a, b) in laws['relationships'].items():
            y_name, x_name = rel_name.split("_vs_")
            # Check if relationship is meaningful (not just noise)
            if abs(a) > 0.01 or abs(b) > 0.01:
                strong_rels.append((rel_name, a, b, y_name, x_name))
        
        if strong_rels:
            for rel_name, a, b, y_name, x_name in strong_rels[:10]:  # Show top 10
                if abs(a) < 1e-6:
                    print(f"  {y_name} = {b:.6f}")
                elif abs(b) < 1e-6:
                    print(f"  {y_name} = {a:.6f} * {x_name}")
                else:
                    sign = "+" if b >= 0 else ""
                    print(f"  {y_name} = {a:.6f} * {x_name} {sign}{b:.6f}")
        else:
            print("  (Relationships detected but coefficients are very small)")
    else:
        print("  (No strong relationships detected)")
        print("  Note: This may mean the system needs more evolution or different forces.")
    
    print()
    print("=" * 60)
    print("Interpretation:")
    print("=" * 60)
    print()
    print("If all quantities are constant:")
    print("  → System is too static. Add more evolution (rotations, basin updates, etc.)")
    print()
    print("If relationships are detected:")
    print("  → These are the physical laws governing your system!")
    print()
    print("If SW_sum is conserved but other quantities vary:")
    print("  → This is correct! SW conservation is a fundamental law.")
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

