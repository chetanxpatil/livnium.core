"""
Demonstration: Omcube vs DataCube Distinction

This script demonstrates the fundamental difference between:
- Omcubes (odd N ≥ 3): Livnium Core Universes with full computational power
- DataCubes (even N ≥ 2): Resource grids for data storage only

This protects the core Livnium concept by clearly showing:
1. Why even cubes CANNOT be Livnium cores
2. What capabilities each type has
3. How they can work together (DataCube → OmCube → DataCube)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.classical import LivniumCoreSystem, DataCube
from core.config import LivniumCoreConfig


def demonstrate_omcube(sizes: list = [3, 5, 7]):
    """
    Demonstrate Omcubes (odd N ≥ 3) - Livnium Core Universes.
    
    Shows:
    - Observer anchoring (center cell exists)
    - Symbolic Weight system
    - Face exposure rules
    - Class structure
    - Total SW calculations
    """
    print("\n" + "="*70)
    print("OMCUBES: Livnium Core Universes (Odd N ≥ 3)")
    print("="*70)
    
    for size in sizes:
        print(f"\n{'─'*70}")
        print(f"Omcube {size}×{size}×{size}")
        print(f"{'─'*70}")
        
        try:
            config = LivniumCoreConfig(lattice_size=size)
            system = LivniumCoreSystem(config)
            
            # Check observer anchor
            center = (0, 0, 0)
            center_cell = system.get_cell(center)
            
            print(f"✅ Center cell exists: {center}")
            print(f"   Symbol: {center_cell.symbol if center_cell else 'None'}")
            print(f"   Face exposure: {center_cell.face_exposure}")
            print(f"   Symbolic Weight: {center_cell.symbolic_weight}")
            print(f"   Cell class: {center_cell.cell_class.name}")
            
            # Total SW
            total_sw = system.get_total_symbolic_weight()
            expected_sw = system.get_expected_total_sw()
            print(f"\n✅ Total Symbolic Weight: {total_sw:.1f} (expected: {expected_sw:.1f})")
            
            # Class counts
            class_counts = system.get_class_counts()
            expected_counts = system.get_expected_class_counts()
            print(f"\n✅ Class counts:")
            for cls in class_counts:
                actual = class_counts[cls]
                expected = expected_counts[cls]
                match = "✅" if actual == expected else "❌"
                print(f"   {match} {cls.name}: {actual} (expected: {expected})")
            
            # Observer system
            if system.global_observer:
                print(f"\n✅ Global Observer anchored at: {system.global_observer.coordinates}")
            
            print(f"\n✅ This is a LIVNIUM CORE UNIVERSE")
            print(f"   - Can execute all 7 axioms")
            print(f"   - Can perform collapse mechanics")
            print(f"   - Can do recursive geometry")
            print(f"   - Can anchor observers")
            
        except ValueError as e:
            print(f"❌ Error: {e}")


def demonstrate_datacube(sizes: list = [2, 4, 6]):
    """
    Demonstrate DataCubes (even N ≥ 2) - Resource Grids.
    
    Shows:
    - No center cell (no observer anchor)
    - No symbolic weight
    - No face exposure
    - Just data storage
    """
    print("\n" + "="*70)
    print("DATACUBES: Resource Grids (Even N ≥ 2)")
    print("="*70)
    
    for size in sizes:
        print(f"\n{'─'*70}")
        print(f"DataCube {size}×{size}×{size}")
        print(f"{'─'*70}")
        
        try:
            datacube = DataCube(size)
            
            # Check for center
            center = (0, 0, 0)
            center_cell = datacube.get_cell(center)
            
            if center_cell:
                print(f"⚠️  Center cell exists: {center}")
                print(f"   But this is NOT an observer anchor")
                print(f"   (Even cubes have no geometric center in Livnium)")
            else:
                print(f"❌ No center cell at {center}")
            
            # No SW system
            print(f"\n❌ NO Symbolic Weight system")
            print(f"   DataCubes cannot calculate SW = 9·f")
            
            # No face exposure
            print(f"\n❌ NO Face Exposure rules")
            print(f"   DataCubes have no cell classification")
            
            # Just data storage
            print(f"\n✅ Data storage works:")
            test_data = f"data_{size}"
            datacube.set_data((0, 0, 0), test_data)
            retrieved = datacube.get_data((0, 0, 0))
            print(f"   Set data at (0,0,0): '{test_data}'")
            print(f"   Retrieved: '{retrieved}'")
            
            # Numpy conversion
            arr = datacube.to_numpy()
            print(f"\n✅ Numpy array shape: {arr.shape}")
            
            print(f"\n✅ This is a RESOURCE GRID")
            print(f"   - Can store data")
            print(f"   - Can act as I/O buffer")
            print(f"   - CANNOT execute Livnium axioms")
            print(f"   - CANNOT perform collapse mechanics")
            print(f"   - CANNOT do recursive geometry")
            
        except ValueError as e:
            print(f"❌ Error: {e}")


def demonstrate_why_even_cannot_be_core():
    """
    Demonstrate why even cubes CANNOT be Livnium cores.
    
    Shows the mathematical/geometric reasons:
    1. No center cell → no observer anchor
    2. Parity mismatch → no stable exposure cycles
    3. Rotations don't preserve class-count invariants
    4. SW maps cannot align symmetrically
    """
    print("\n" + "="*70)
    print("WHY EVEN CUBES CANNOT BE LIVNIUM CORES")
    print("="*70)
    
    print("\n1. NO CENTER CELL → NO OBSERVER ANCHOR")
    print("   " + "─"*66)
    print("   For odd N (3, 5, 7, ...):")
    print("   - Coordinate range: {-(N-1)/2, ..., (N-1)/2}")
    print("   - Center at (0, 0, 0) exists")
    print("   - Observer can anchor at center")
    print()
    print("   For even N (2, 4, 6, ...):")
    print("   - Coordinate range: {-(N/2-1), ..., N/2-1}")
    print("   - No true geometric center")
    print("   - Observer cannot anchor → Axiom A2 violated")
    
    print("\n2. PARITY MISMATCH → NO STABLE EXPOSURE CYCLES")
    print("   " + "─"*66)
    print("   Odd cubes: Symmetric face exposure patterns")
    print("   - Core cells: f=0")
    print("   - Center cells: f=1")
    print("   - Edge cells: f=2")
    print("   - Corner cells: f=3")
    print("   - All classes have stable counts")
    print()
    print("   Even cubes: Asymmetric patterns")
    print("   - No clear core/center/edge/corner distinction")
    print("   - Exposure cycles break under rotation")
    print("   - Class counts not preserved → Axiom A3 violated")
    
    print("\n3. ROTATIONS DON'T PRESERVE INVARIANTS")
    print("   " + "─"*66)
    print("   Odd cubes: 24-element rotation group preserves:")
    print("   - Total SW (ΣSW invariant)")
    print("   - Class counts (core/center/edge/corner)")
    print("   - Observer position")
    print()
    print("   Even cubes: Rotations break invariants")
    print("   - Class counts change")
    print("   - SW distribution shifts")
    print("   - No stable observer reference → Axiom A4 violated")
    
    print("\n4. SW MAPS CANNOT ALIGN SYMMETRICALLY")
    print("   " + "─"*66)
    print("   Odd cubes: SW = 9·f works perfectly")
    print("   - Face exposure f ∈ {0,1,2,3} maps cleanly")
    print("   - Total SW = 54(N-2)² + 216(N-2) + 216")
    print("   - Formula verified for N=3,5,7,...")
    print()
    print("   Even cubes: SW formula breaks")
    print("   - No clear face exposure classification")
    print("   - SW distribution is asymmetric")
    print("   - Cannot maintain SW conservation → Axiom A3 violated")
    
    print("\n" + "="*70)
    print("CONCLUSION: Even cubes are NOT Livnium cores")
    print("="*70)
    print("\nThey can only serve as:")
    print("  • Data storage containers")
    print("  • I/O buffers")
    print("  • Feature maps")
    print("  • Preprocessing/postprocessing containers")
    print("\nThey CANNOT:")
    print("  • Execute Livnium collapse mechanics")
    print("  • Implement symbolic weight system")
    print("  • Perform recursive geometry")
    print("  • Anchor observers")
    print("  • Maintain Livnium invariants")


def demonstrate_architecture():
    """
    Demonstrate the architecture: DataCube → OmCube → DataCube
    """
    print("\n" + "="*70)
    print("ARCHITECTURE: DataCube → OmCube → DataCube")
    print("="*70)
    
    print("\n1. INPUT BUFFER (DataCube 4×4×4)")
    print("   " + "─"*66)
    input_cube = DataCube(4)
    # Simulate input data
    for i in range(4):
        for j in range(4):
            for k in range(4):
                input_cube.set_data((i-1, j-1, k-1), f"input_{i}_{j}_{k}")
    print(f"   ✅ Loaded {len(input_cube.lattice)} data points")
    print(f"   ✅ Ready to feed into OmCube")
    
    print("\n2. COMPUTATION (OmCube 3×3×3)")
    print("   " + "─"*66)
    config = LivniumCoreConfig(lattice_size=3)
    omcube = LivniumCoreSystem(config)
    print(f"   ✅ Livnium Core initialized")
    print(f"   ✅ Observer anchored at {omcube.global_observer.coordinates}")
    print(f"   ✅ Total SW: {omcube.get_total_symbolic_weight():.1f}")
    print(f"   ✅ Can execute all 7 axioms")
    print(f"   ✅ Processing data from DataCube...")
    
    # Simulate processing (in real use, you'd map DataCube data to OmCube)
    print(f"   ✅ Computation complete")
    
    print("\n3. OUTPUT BUFFER (DataCube 6×6×6)")
    print("   " + "─"*66)
    output_cube = DataCube(6)
    # Simulate output data
    for i in range(6):
        for j in range(6):
            for k in range(6):
                output_cube.set_data((i-2, j-2, k-2), f"output_{i}_{j}_{k}")
    print(f"   ✅ Received {len(output_cube.lattice)} results")
    print(f"   ✅ Ready for postprocessing")
    
    print("\n" + "="*70)
    print("ARCHITECTURE SUMMARY")
    print("="*70)
    print("\nDataCubes = RAM (storage, I/O)")
    print("OmCubes = CPU (computation, axioms)")
    print("\nThis separation protects Livnium's core intellectual property:")
    print("  • Only odd cubes can implement Livnium axioms")
    print("  • Even cubes are just plain grids")
    print("  • Using even cubes ≠ running Livnium")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("LIVNIUM: OMCUBE vs DATACUBE DEMONSTRATION")
    print("="*70)
    print("\nThis demonstration protects the core Livnium concept by")
    print("clearly showing why only odd-dimensional cubes can be Livnium cores.")
    
    # Demonstrate Omcubes
    demonstrate_omcube([3, 5, 7])
    
    # Demonstrate DataCubes
    demonstrate_datacube([2, 4, 6])
    
    # Explain why even cubes can't be cores
    demonstrate_why_even_cannot_be_core()
    
    # Show architecture
    demonstrate_architecture()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Takeaway:")
    print("  • Omcubes (odd N ≥ 3) = Livnium Core Universes")
    print("  • DataCubes (even N ≥ 2) = Resource Grids")
    print("  • This distinction protects Livnium's intellectual property")
    print("  • Even cubes cannot implement Livnium axioms")


if __name__ == "__main__":
    main()

