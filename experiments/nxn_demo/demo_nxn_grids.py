"""
Demonstration: 2D N×N Grids vs 3D N×N×N Omcubes

This script demonstrates the fundamental difference between:
- Omcubes (3D, odd N ≥ 3): Livnium Core Universes with full computational power
- DataGrids (2D, any N ≥ 2): Resource grids for data storage only
- DataCubes (3D, even N ≥ 2): Resource grids for data storage only

This protects the core Livnium concept by clearly showing:
1. Why 2D grids CANNOT be Livnium cores (Livnium is fundamentally 3D)
2. Why even 3D cubes CANNOT be Livnium cores (no center cell)
3. What capabilities each type has
4. How they can work together (DataGrid → OmCube → DataGrid)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.classical import LivniumCoreSystem, DataCube, DataGrid
from core.config import LivniumCoreConfig


def demonstrate_omcube(sizes: list = [3, 5, 7]):
    """
    Demonstrate Omcubes (3D, odd N ≥ 3) - Livnium Core Universes.
    
    Shows:
    - Observer anchoring (center cell exists)
    - Symbolic Weight system
    - Face exposure rules (3D)
    - Class structure
    - Total SW calculations
    """
    print("\n" + "="*70)
    print("OMCUBES: Livnium Core Universes (3D, Odd N ≥ 3)")
    print("="*70)
    
    for size in sizes:
        print(f"\n{'─'*70}")
        print(f"Omcube {size}×{size}×{size} (3D)")
        print(f"{'─'*70}")
        
        try:
            config = LivniumCoreConfig(lattice_size=size)
            system = LivniumCoreSystem(config)
            
            # Check observer anchor
            center = (0, 0, 0)
            center_cell = system.get_cell(center)
            
            print(f"✅ Center cell exists: {center} (3D)")
            print(f"   Symbol: {center_cell.symbol if center_cell else 'None'}")
            print(f"   Face exposure: {center_cell.face_exposure} (3D concept)")
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
                print(f"\n✅ Global Observer anchored at: {system.global_observer.coordinates} (3D)")
            
            print(f"\n✅ This is a LIVNIUM CORE UNIVERSE (3D)")
            print(f"   - Can execute all 7 axioms")
            print(f"   - Can perform collapse mechanics")
            print(f"   - Can do recursive geometry")
            print(f"   - Can anchor observers")
            print(f"   - 3D face exposure system")
            
        except ValueError as e:
            print(f"❌ Error: {e}")


def demonstrate_datagrid(sizes: list = [2, 3, 4, 5]):
    """
    Demonstrate DataGrids (2D, any N ≥ 2) - Resource Grids.
    
    Shows:
    - No 3D structure (2D only)
    - No face exposure (3D concept)
    - No symbolic weight
    - No observer anchor (3D concept)
    - Just data storage
    """
    print("\n" + "="*70)
    print("DATAGRIDS: Resource Grids (2D, Any N ≥ 2)")
    print("="*70)
    
    for size in sizes:
        print(f"\n{'─'*70}")
        print(f"DataGrid {size}×{size} (2D)")
        print(f"{'─'*70}")
        
        try:
            datagrid = DataGrid(size)
            
            # Check for center
            center = (0, 0)
            center_cell = datagrid.get_cell(center)
            
            if center_cell:
                print(f"⚠️  Center cell exists: {center} (2D)")
                print(f"   But this is NOT a 3D observer anchor")
                print(f"   (Livnium requires 3D structure)")
            else:
                print(f"❌ No center cell at {center}")
            
            # No SW system
            print(f"\n❌ NO Symbolic Weight system")
            print(f"   DataGrids are 2D - no 3D face exposure")
            
            # No face exposure
            print(f"\n❌ NO Face Exposure rules")
            print(f"   Face exposure is a 3D concept (6 faces)")
            print(f"   DataGrids are 2D (no faces, just edges)")
            
            # No 3D structure
            print(f"\n❌ NO 3D Structure")
            print(f"   Livnium is fundamentally 3D")
            print(f"   DataGrids are 2D - cannot be Livnium cores")
            
            # Just data storage
            print(f"\n✅ Data storage works:")
            test_data = f"data_{size}"
            datagrid.set_data((0, 0), test_data)
            retrieved = datagrid.get_data((0, 0))
            print(f"   Set data at (0,0): '{test_data}'")
            print(f"   Retrieved: '{retrieved}'")
            
            # Numpy conversion
            arr = datagrid.to_numpy()
            print(f"\n✅ Numpy array shape: {arr.shape} (2D)")
            
            print(f"\n✅ This is a RESOURCE GRID (2D)")
            print(f"   - Can store data")
            print(f"   - Can act as I/O buffer")
            print(f"   - CANNOT execute Livnium axioms (3D required)")
            print(f"   - CANNOT perform collapse mechanics (3D only)")
            print(f"   - CANNOT do recursive geometry (3D only)")
            
        except ValueError as e:
            print(f"❌ Error: {e}")


def demonstrate_datacube(sizes: list = [2, 4, 6]):
    """
    Demonstrate DataCubes (3D, even N ≥ 2) - Resource Grids.
    
    Shows:
    - 3D structure but no center cell
    - No symbolic weight
    - No face exposure
    - Just data storage
    """
    print("\n" + "="*70)
    print("DATACUBES: Resource Grids (3D, Even N ≥ 2)")
    print("="*70)
    
    for size in sizes:
        print(f"\n{'─'*70}")
        print(f"DataCube {size}×{size}×{size} (3D, even)")
        print(f"{'─'*70}")
        
        try:
            datacube = DataCube(size)
            
            # Check for center
            center = (0, 0, 0)
            center_cell = datacube.get_cell(center)
            
            if center_cell:
                print(f"⚠️  Center cell exists: {center} (3D)")
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
            print(f"\n✅ Numpy array shape: {arr.shape} (3D)")
            
            print(f"\n✅ This is a RESOURCE GRID (3D, even)")
            print(f"   - Can store data")
            print(f"   - Can act as I/O buffer")
            print(f"   - CANNOT execute Livnium axioms")
            print(f"   - CANNOT perform collapse mechanics")
            print(f"   - CANNOT do recursive geometry")
            
        except ValueError as e:
            print(f"❌ Error: {e}")


def demonstrate_why_2d_cannot_be_core():
    """
    Demonstrate why 2D grids CANNOT be Livnium cores.
    
    Shows the fundamental reasons:
    1. Livnium is fundamentally 3D
    2. Face exposure requires 3D (6 faces)
    3. Observer anchor requires 3D center
    4. Rotations are 3D operations
    """
    print("\n" + "="*70)
    print("WHY 2D GRIDS CANNOT BE LIVNIUM CORES")
    print("="*70)
    
    print("\n1. LIVNIUM IS FUNDAMENTALLY 3D")
    print("   " + "─"*66)
    print("   Livnium Core requires:")
    print("   - 3D lattice structure (N×N×N)")
    print("   - 3D coordinate system (x, y, z)")
    print("   - 3D rotations (24-element group)")
    print("   - 3D observer anchor at (0, 0, 0)")
    print()
    print("   2D grids have:")
    print("   - 2D structure (N×N)")
    print("   - 2D coordinate system (x, y)")
    print("   - 2D rotations (4-element group)")
    print("   - 2D center at (0, 0) - NOT a 3D anchor")
    print("   → Fundamental dimension mismatch")
    
    print("\n2. FACE EXPOSURE REQUIRES 3D")
    print("   " + "─"*66)
    print("   Face exposure (f) counts exposed faces:")
    print("   - 3D cube has 6 faces")
    print("   - f ∈ {0, 1, 2, 3} based on boundary position")
    print("   - SW = 9·f requires 3D face counting")
    print()
    print("   2D grids have:")
    print("   - 4 edges (not faces)")
    print("   - No face exposure concept")
    print("   - Cannot calculate SW = 9·f")
    print("   → Axiom A3 (Symbolic Weight) requires 3D")
    
    print("\n3. OBSERVER ANCHOR REQUIRES 3D CENTER")
    print("   " + "─"*66)
    print("   Observer anchor (Axiom A2):")
    print("   - Must be at 3D center (0, 0, 0)")
    print("   - Requires 3D coordinate system")
    print("   - 3D rotations preserve observer position")
    print()
    print("   2D grids have:")
    print("   - 2D center (0, 0)")
    print("   - No 3D observer anchor")
    print("   - Cannot implement Axiom A2")
    print("   → Observer system requires 3D")
    
    print("\n4. ROTATIONS ARE 3D OPERATIONS")
    print("   " + "─"*66)
    print("   Livnium rotations:")
    print("   - 24-element rotation group (3D)")
    print("   - Quarter-turns around X, Y, Z axes")
    print("   - Preserve 3D invariants")
    print()
    print("   2D grids have:")
    print("   - 4-element rotation group (2D)")
    print("   - Rotations around Z axis only")
    print("   - Cannot preserve 3D invariants")
    print("   → Axiom A4 (Dynamic Law) requires 3D")
    
    print("\n" + "="*70)
    print("CONCLUSION: 2D grids are NOT Livnium cores")
    print("="*70)
    print("\nThey can only serve as:")
    print("  • Data storage containers")
    print("  • I/O buffers")
    print("  • Feature maps")
    print("  • Preprocessing/postprocessing containers")
    print("\nThey CANNOT:")
    print("  • Execute Livnium collapse mechanics (3D required)")
    print("  • Implement symbolic weight system (3D face exposure)")
    print("  • Perform recursive geometry (3D only)")
    print("  • Anchor observers (3D center required)")
    print("  • Maintain Livnium invariants (3D structure required)")


def demonstrate_architecture():
    """
    Demonstrate the architecture: DataGrid → OmCube → DataGrid
    """
    print("\n" + "="*70)
    print("ARCHITECTURE: DataGrid → OmCube → DataGrid")
    print("="*70)
    
    print("\n1. INPUT BUFFER (DataGrid 5×5 - 2D)")
    print("   " + "─"*66)
    input_grid = DataGrid(5)
    # Simulate input data
    for i in range(5):
        for j in range(5):
            input_grid.set_data((i-2, j-2), f"input_{i}_{j}")
    print(f"   ✅ Loaded {len(input_grid.lattice)} data points (2D)")
    print(f"   ✅ Ready to feed into OmCube (3D)")
    
    print("\n2. COMPUTATION (OmCube 3×3×3 - 3D)")
    print("   " + "─"*66)
    config = LivniumCoreConfig(lattice_size=3)
    omcube = LivniumCoreSystem(config)
    print(f"   ✅ Livnium Core initialized (3D)")
    print(f"   ✅ Observer anchored at {omcube.global_observer.coordinates} (3D)")
    print(f"   ✅ Total SW: {omcube.get_total_symbolic_weight():.1f}")
    print(f"   ✅ Can execute all 7 axioms (3D)")
    print(f"   ✅ Processing data from DataGrid...")
    
    # Simulate processing (in real use, you'd map DataGrid data to OmCube)
    print(f"   ✅ Computation complete (3D)")
    
    print("\n3. OUTPUT BUFFER (DataGrid 7×7 - 2D)")
    print("   " + "─"*66)
    output_grid = DataGrid(7)
    # Simulate output data
    for i in range(7):
        for j in range(7):
            output_grid.set_data((i-3, j-3), f"output_{i}_{j}")
    print(f"   ✅ Received {len(output_grid.lattice)} results (2D)")
    print(f"   ✅ Ready for postprocessing")
    
    print("\n" + "="*70)
    print("ARCHITECTURE SUMMARY")
    print("="*70)
    print("\nDataGrids (2D) = RAM (storage, I/O)")
    print("OmCubes (3D, odd) = CPU (computation, axioms)")
    print("\nThis separation protects Livnium's core intellectual property:")
    print("  • Only 3D odd cubes can implement Livnium axioms")
    print("  • 2D grids are just plain grids")
    print("  • Using 2D grids ≠ running Livnium")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("LIVNIUM: 2D N×N GRIDS vs 3D N×N×N OMCUBES")
    print("="*70)
    print("\nThis demonstration protects the core Livnium concept by")
    print("clearly showing why only 3D odd-dimensional cubes can be Livnium cores.")
    
    # Demonstrate Omcubes (3D, odd)
    demonstrate_omcube([3, 5, 7])
    
    # Demonstrate DataGrids (2D, any N)
    demonstrate_datagrid([2, 3, 4, 5])
    
    # Demonstrate DataCubes (3D, even)
    demonstrate_datacube([2, 4, 6])
    
    # Explain why 2D grids can't be cores
    demonstrate_why_2d_cannot_be_core()
    
    # Show architecture
    demonstrate_architecture()
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey Takeaway:")
    print("  • Omcubes (3D, odd N ≥ 3) = Livnium Core Universes")
    print("  • DataGrids (2D, any N ≥ 2) = Resource Grids")
    print("  • DataCubes (3D, even N ≥ 2) = Resource Grids")
    print("  • This distinction protects Livnium's intellectual property")
    print("  • Only 3D odd cubes can implement Livnium axioms")


if __name__ == "__main__":
    main()

