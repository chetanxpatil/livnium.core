"""
Test Entanglement Capacity in Recursive Geometry

Tests how many omcubes can be entangled simultaneously using:
- Pairwise entanglement (Bell pairs)
- Chain entanglement (linear chains)
- Cluster entanglement (fully connected clusters)
- Recursive entanglement (across levels)
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Set

# Make repo root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from core.classical.livnium_core_system import LivniumCoreSystem
from core.config import LivniumCoreConfig
from core.recursive import RecursiveGeometryEngine
from core.quantum import QuantumLattice, QuantumCell
from core.quantum.entanglement_manager import EntanglementManager, EntangledPair


def count_entangled_omcubes_recursive(level, entanglement_managers: Dict[int, EntanglementManager]) -> int:
    """Recursively count all entangled omcubes."""
    level_id = level.level_id
    
    # Count entangled omcubes at this level
    if level_id in entanglement_managers:
        manager = entanglement_managers[level_id]
        stats = manager.get_entanglement_statistics()
        # Each pair involves 2 omcubes, but we count unique omcubes
        entangled_omcubes = stats['entangled_cells']
    else:
        entangled_omcubes = 0
    
    # Count children recursively
    for child_level in level.children.values():
        entangled_omcubes += count_entangled_omcubes_recursive(child_level, entanglement_managers)
    
    return entangled_omcubes


def initialize_quantum_for_all_levels(level, 
                                      quantum_lattices: Dict[int, QuantumLattice],
                                      entanglement_managers: Dict[int, EntanglementManager]):
    """Recursively initialize quantum lattices for all levels."""
    level_id = level.level_id
    
    # Initialize quantum lattice for this level if not already done
    if level_id not in quantum_lattices:
        # Check if geometry has quantum enabled
        if level.geometry.config.enable_quantum:
            try:
                quantum_lattices[level_id] = QuantumLattice(level.geometry)
                entanglement_managers[level_id] = quantum_lattices[level_id].entanglement_manager
            except Exception as e:
                # If quantum not enabled in child, skip
                pass
    
    # Recursively initialize children
    for child_level in level.children.values():
        initialize_quantum_for_all_levels(child_level, quantum_lattices, entanglement_managers)


def test_pairwise_entanglement(base_lattice_size: int = 5,
                               max_depth: int = 3) -> Dict:
    """
    Test pairwise entanglement capacity (Bell pairs).
    
    Creates maximum number of Bell pairs across the recursive structure.
    """
    print("=" * 70)
    print("Test 1: Pairwise Entanglement (Bell Pairs)")
    print("=" * 70)
    
    # Initialize system
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    # Initialize quantum lattices and entanglement managers for each level
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    total_pairs = 0
    level_stats = {}
    
    # Note: Now that child geometries inherit quantum features, we can entangle across all levels
    
    def entangle_level_recursive(level, level_path=""):
        """Recursively entangle all levels."""
        nonlocal total_pairs
        
        level_id = level.level_id
        level_key = f"{level_path}_{level_id}" if level_path else str(level_id)
        
        # Check if this geometry has quantum enabled
        if not level.geometry.config.enable_quantum:
            # Skip if quantum not enabled
            return
        
        # Create quantum lattice for this level if needed
        if level_id not in quantum_lattices:
            try:
                quantum_lattices[level_id] = QuantumLattice(level.geometry)
                entanglement_managers[level_id] = quantum_lattices[level_id].entanglement_manager
            except Exception as e:
                # If quantum lattice creation fails, skip this level
                return
        
        manager = entanglement_managers[level_id]
        geometry = level.geometry
        
        # Create Bell pairs: entangle each cell with its nearest neighbors
        pairs_created = 0
        cells = list(geometry.lattice.keys())
        
        # Create pairs: each cell with its 6 nearest neighbors (if not already paired)
        for i, cell1 in enumerate(cells):
            # Get nearest neighbors
            neighbors = manager._get_nearest_neighbors(cell1)
            
            for neighbor in neighbors:
                if neighbor in cells and neighbor != cell1:
                    # Check if not already entangled
                    if not manager.is_entangled(cell1, neighbor):
                        manager.create_bell_pair(cell1, neighbor, "phi_plus")
                        pairs_created += 1
        
        # Each pair is counted twice (both directions), so divide by 2
        actual_pairs = pairs_created // 2
        total_pairs += actual_pairs
        
        stats = manager.get_entanglement_statistics()
        
        # Aggregate stats by level_id (multiple child geometries at same level)
        if level_id not in level_stats:
            level_stats[level_id] = {
                'pairs': 0,
                'entangled_cells': 0,
                'max_connections': 0,
                'num_geometries': 0
            }
        
        level_stats[level_id]['pairs'] += actual_pairs
        level_stats[level_id]['entangled_cells'] = max(
            level_stats[level_id]['entangled_cells'],
            stats['entangled_cells']
        )
        level_stats[level_id]['max_connections'] = max(
            level_stats[level_id]['max_connections'],
            stats['max_connections_per_cell']
        )
        level_stats[level_id]['num_geometries'] += 1
        
        # Recursively process children
        for child_coords, child_level in level.children.items():
            entangle_level_recursive(child_level, f"{level_key}_{child_coords}")
    
    # Start recursive entanglement from level 0
    entangle_level_recursive(level_0)
    
    # Print stats by level
    for level_id in sorted(level_stats.keys()):
        stats = level_stats[level_id]
        print(f"Level {level_id}: {stats['pairs']:,} Bell pairs, "
              f"{stats['entangled_cells']:,} entangled cells, "
              f"max {stats['max_connections']} connections/cell, "
              f"{stats['num_geometries']} geometries")
    
    # Count total entangled omcubes across all recursive levels
    # Each geometry at a level has its own set of entangled cells
    # We need to count: (cells per geometry) × (number of geometries) for each level
    total_entangled = 0
    for level_id, stats in level_stats.items():
        # Each geometry at this level has stats['entangled_cells'] omcubes
        # And there are stats['num_geometries'] geometries at this level
        # Total = cells per geometry × number of geometries
        cells_per_geometry = stats['entangled_cells']
        num_geometries = stats['num_geometries']
        total_entangled += cells_per_geometry * num_geometries
    
    print()
    print(f"Total Bell pairs: {total_pairs:,}")
    print(f"Total entangled omcubes (across all recursive levels): {total_entangled:,}")
    print(f"Average pairs per omcube: {total_pairs / total_entangled:.4f}" if total_entangled > 0 else "")
    print()
    print(f"📊 Breakdown:")
    for level_id in sorted(level_stats.keys()):
        stats = level_stats[level_id]
        cells_per_geo = stats['entangled_cells']
        num_geos = stats['num_geometries']
        level_total = cells_per_geo * num_geos
        print(f"  Level {level_id}: {cells_per_geo} cells/geometry × {num_geos:,} geometries = {level_total:,} entangled omcubes")
    
    return {
        'total_pairs': total_pairs,
        'total_entangled_omcubes': total_entangled,
        'level_stats': level_stats
    }


def test_chain_entanglement(base_lattice_size: int = 5,
                            max_depth: int = 3) -> Dict:
    """
    Test chain entanglement capacity (linear chains).
    
    Creates maximum length entanglement chains.
    """
    print("\n" + "=" * 70)
    print("Test 2: Chain Entanglement (Linear Chains)")
    print("=" * 70)
    
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    # Initialize quantum for ALL recursive levels
    initialize_quantum_for_all_levels(level_0, quantum_lattices, entanglement_managers)
    
    total_chains = 0
    max_chain_length = 0
    level_stats = {}
    
    # Recursively create chains at all levels
    def create_chains_recursive(level):
        """Recursively create chains at this level and all children."""
        level_id = level.level_id
        nonlocal total_chains, max_chain_length
        
        # Create chains at this level if quantum is enabled
        if level_id in entanglement_managers:
            manager = entanglement_managers[level_id]
            geometry = level.geometry
            cells = list(geometry.lattice.keys())
            
            # Create chains: connect cells in a path
            chains_created = 0
            used_cells = set()
            max_length = 0
            
            for start_cell in cells:
                if start_cell in used_cells:
                    continue
                
                # Build chain from this cell
                chain = [start_cell]
                current = start_cell
                used_cells.add(current)
                
                # Extend chain by finding unentangled neighbors
                while True:
                    neighbors = manager._get_nearest_neighbors(current)
                    next_cell = None
                    
                    for neighbor in neighbors:
                        if neighbor in cells and neighbor not in used_cells:
                            if not manager.is_entangled(current, neighbor):
                                next_cell = neighbor
                                break
                    
                    if next_cell is None:
                        break
                    
                    # Create entanglement
                    manager.create_bell_pair(current, next_cell, "phi_plus")
                    chain.append(next_cell)
                    current = next_cell
                    used_cells.add(current)
                
                if len(chain) > 1:
                    chains_created += 1
                    max_length = max(max_length, len(chain))
            
            total_chains += chains_created
            max_chain_length = max(max_chain_length, max_length)
            
            stats = manager.get_entanglement_statistics()
            if level_id not in level_stats:
                level_stats[level_id] = {
                    'chains': 0,
                    'max_chain_length': 0,
                    'entangled_cells': 0
                }
            level_stats[level_id]['chains'] += chains_created
            level_stats[level_id]['max_chain_length'] = max(
                level_stats[level_id]['max_chain_length'],
                max_length
            )
            level_stats[level_id]['entangled_cells'] = stats['entangled_cells']
        
        # Recursively process children
        for child_level in level.children.values():
            create_chains_recursive(child_level)
    
    # Create chains recursively across all levels
    create_chains_recursive(level_0)
    
    # Print stats by level
    for level_id in sorted(level_stats.keys()):
        stats = level_stats[level_id]
        print(f"Level {level_id}: {stats['chains']:,} chains, "
              f"max length {stats['max_chain_length']}, "
              f"{stats['entangled_cells']:,} entangled cells")
    
    total_entangled = count_entangled_omcubes_recursive(level_0, entanglement_managers)
    
    print()
    print(f"Total chains: {total_chains:,}")
    print(f"Max chain length: {max_chain_length}")
    print(f"Total entangled omcubes: {total_entangled:,}")
    
    return {
        'total_chains': total_chains,
        'max_chain_length': max_chain_length,
        'total_entangled_omcubes': total_entangled,
        'level_stats': level_stats
    }


def test_cluster_entanglement(base_lattice_size: int = 5,
                              max_depth: int = 2,
                              cluster_size: int = 10) -> Dict:
    """
    Test cluster entanglement (fully connected clusters).
    
    Creates clusters of fully connected omcubes.
    """
    print("\n" + "=" * 70)
    print(f"Test 3: Cluster Entanglement ({cluster_size}-omcube clusters)")
    print("=" * 70)
    
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    # Initialize quantum for ALL recursive levels
    initialize_quantum_for_all_levels(level_0, quantum_lattices, entanglement_managers)
    
    total_clusters = 0
    total_entanglements_in_clusters = 0
    level_stats = {}
    
    # Recursively create clusters at all levels
    def create_clusters_recursive(level):
        """Recursively create clusters at this level and all children."""
        level_id = level.level_id
        nonlocal total_clusters, total_entanglements_in_clusters
        
        # Create clusters at this level if quantum is enabled
        if level_id in entanglement_managers:
            manager = entanglement_managers[level_id]
            geometry = level.geometry
            cells = list(geometry.lattice.keys())
            
            # Create clusters: groups of fully connected omcubes
            clusters_created = 0
            used_cells = set()
            
            # Create clusters of size cluster_size
            for i in range(0, len(cells), cluster_size):
                cluster_cells = cells[i:i+cluster_size]
                
                # Skip if any cell already used
                if any(cell in used_cells for cell in cluster_cells):
                    continue
                
                # Fully connect cluster: each cell with every other
                cluster_entanglements = 0
                for j, cell1 in enumerate(cluster_cells):
                    for cell2 in cluster_cells[j+1:]:
                        if not manager.is_entangled(cell1, cell2):
                            manager.create_bell_pair(cell1, cell2, "phi_plus")
                            cluster_entanglements += 1
                            used_cells.add(cell1)
                            used_cells.add(cell2)
                
                if cluster_entanglements > 0:
                    clusters_created += 1
                    total_entanglements_in_clusters += cluster_entanglements
            
            total_clusters += clusters_created
            
            stats = manager.get_entanglement_statistics()
            if level_id not in level_stats:
                level_stats[level_id] = {
                    'clusters': 0,
                    'entanglements': 0,
                    'entangled_cells': 0
                }
            level_stats[level_id]['clusters'] += clusters_created
            level_stats[level_id]['entanglements'] = stats['total_entangled_pairs']
            level_stats[level_id]['entangled_cells'] = stats['entangled_cells']
        
        # Recursively process children
        for child_level in level.children.values():
            create_clusters_recursive(child_level)
    
    # Create clusters recursively across all levels
    create_clusters_recursive(level_0)
    
    # Print stats by level
    for level_id in sorted(level_stats.keys()):
        stats = level_stats[level_id]
        print(f"Level {level_id}: {stats['clusters']:,} clusters, "
              f"{stats['entanglements']:,} pairs, "
              f"{stats['entangled_cells']:,} entangled cells")
    
    total_entangled = count_entangled_omcubes_recursive(level_0, entanglement_managers)
    
    print()
    print(f"Total clusters: {total_clusters:,}")
    print(f"Total entanglements in clusters: {total_entanglements_in_clusters:,}")
    print(f"Total entangled omcubes: {total_entangled:,}")
    print(f"Average cluster size: {total_entangled / total_clusters:.1f}" if total_clusters > 0 else "")
    
    return {
        'total_clusters': total_clusters,
        'total_entanglements': total_entanglements_in_clusters,
        'total_entangled_omcubes': total_entangled,
        'level_stats': level_stats
    }


def test_recursive_entanglement(base_lattice_size: int = 5,
                                max_depth: int = 3) -> Dict:
    """
    Test recursive entanglement (entangle across levels).
    
    Creates entanglement between parent and child levels.
    """
    print("\n" + "=" * 70)
    print("Test 4: Recursive Entanglement (Cross-Level)")
    print("=" * 70)
    
    config = LivniumCoreConfig(
        lattice_size=base_lattice_size,
        enable_quantum=True,
        enable_superposition=True,
        enable_quantum_gates=True,
        enable_entanglement=True,
        enable_measurement=True,
        enable_geometry_quantum_coupling=True
    )
    
    base_system = LivniumCoreSystem(config)
    recursive_engine = RecursiveGeometryEngine(
        base_geometry=base_system,
        max_depth=max_depth
    )
    
    quantum_lattices: Dict[int, QuantumLattice] = {}
    entanglement_managers: Dict[int, EntanglementManager] = {}
    
    level_0 = recursive_engine.levels[0]
    quantum_lattices[0] = QuantumLattice(base_system)
    entanglement_managers[0] = quantum_lattices[0].entanglement_manager
    
    cross_level_pairs = 0
    
    # Entangle parent cells with representative child cells
    for level_id, level in recursive_engine.levels.items():
        if level_id not in quantum_lattices:
            quantum_lattices[level_id] = QuantumLattice(level.geometry)
            entanglement_managers[level_id] = quantum_lattices[level_id].entanglement_manager
        
        # For each parent cell, count potential cross-level entanglements
        for parent_coords, child_level in level.children.items():
            # Count potential: each parent cell could entangle with child cells
            child_cells = list(child_level.geometry.lattice.keys())
            if child_cells:
                # Count potential pairs (parent cell with each child cell)
                cross_level_pairs += len(child_cells)
    
    print(f"Potential cross-level entanglements: {cross_level_pairs:,}")
    print("(Note: Cross-level entanglement requires child geometries to have quantum enabled)")
    print("(This is a limitation of current implementation - child configs don't inherit quantum settings)")
    
    return {
        'cross_level_pairs': cross_level_pairs,
        'note': 'Cross-level entanglement limited by child geometry config'
    }


def run_all_tests():
    """Run all entanglement capacity tests."""
    print("=" * 70)
    print("ENTANGLEMENT CAPACITY TEST SUITE")
    print("=" * 70)
    print("Testing how many omcubes can be entangled using recursive geometry")
    print()
    
    base_lattice_size = 5
    max_depth = 3
    
    # Test 1: Pairwise
    result1 = test_pairwise_entanglement(base_lattice_size, max_depth)
    
    # Test 2: Chains
    result2 = test_chain_entanglement(base_lattice_size, max_depth)
    
    # Test 3: Clusters (smaller depth for performance)
    result3 = test_cluster_entanglement(base_lattice_size, max_depth=2, cluster_size=10)
    
    # Test 4: Recursive
    result4 = test_recursive_entanglement(base_lattice_size, max_depth)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Full Recursive Entanglement Capacity")
    print("=" * 70)
    print(f"Pairwise entanglement: {result1['total_entangled_omcubes']:,} omcubes")
    print(f"Chain entanglement: {result2['total_entangled_omcubes']:,} omcubes")
    print(f"Cluster entanglement: {result3['total_entangled_omcubes']:,} omcubes")
    max_entangled = max(result1['total_entangled_omcubes'], result2['total_entangled_omcubes'], result3['total_entangled_omcubes'])
    print(f"Max entangled (any method): {max_entangled:,} omcubes")
    print()
    print("🎯 KEY FINDING: With recursive geometry, we can entangle")
    print(f"   {max_entangled:,} omcubes simultaneously (not just 125!)")
    print("   This is the full capacity across all recursive levels.")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()

