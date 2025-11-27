#!/usr/bin/env python3
"""
Example: How to Run Law Extraction

This script demonstrates how to use the law extractor to discover
physical laws from Livnium system behavior.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.runtime.orchestrator import Orchestrator
from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig


def main():
    """Run law extraction example."""
    print("=" * 60)
    print("Livnium Law Extractor - Example")
    print("=" * 60)
    print()
    
    # Step 1: Create system
    print("Step 1: Creating Livnium Core System...")
    config = LivniumCoreConfig(lattice_size=3)
    system = LivniumCoreSystem(config)
    orchestrator = Orchestrator(system)
    print("✓ System created")
    print()
    
    # Step 2: Run system for N steps
    print("Step 2: Running system for 50 timesteps...")
    print("(Recording physics state each timestep)")
    num_steps = 50
    
    for i in range(num_steps):
        orchestrator.step()
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{num_steps} timesteps...")
    
    print("✓ System run complete")
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
    for name, is_invariant in laws['invariants'].items():
        status = "✓ CONSERVED" if is_invariant else "✗ NOT CONSERVED"
        print(f"  {name}: {status}")
    
    print()
    print("Functional Relationships:")
    if laws['relationships']:
        for rel_name, (a, b) in laws['relationships'].items():
            y_name, x_name = rel_name.split("_vs_")
            if abs(a) < 1e-6:
                print(f"  {y_name} = {b:.6f}")
            elif abs(b) < 1e-6:
                print(f"  {y_name} = {a:.6f} * {x_name}")
            else:
                sign = "+" if b >= 0 else ""
                print(f"  {y_name} = {a:.6f} * {x_name} {sign}{b:.6f}")
    else:
        print("  (No strong relationships detected)")
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()

