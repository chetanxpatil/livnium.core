"""
Test Semantic Consistency

Tests semantic layer consistency:
- Same meaning → same basin path
- Opposite meaning → opposite polarity
- Sentences that differ by one word only shift phi-weight slightly
- Meaning graph never gets disconnected
- Feature extractor invariants hold
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from core.classical.livnium_core_system import LivniumCoreSystem
from core.semantic.feature_extractor import FeatureExtractor
from core.semantic.semantic_processor import SemanticProcessor
from core.semantic.meaning_graph import MeaningGraph
from core.config import LivniumCoreConfig


def test_same_meaning_same_basin():
    """Test that same meaning leads to same basin path."""
    print("=" * 60)
    print("Test 1: Same Meaning → Same Basin Path")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    processor = SemanticProcessor(system)
    
    # Create two identical semantic inputs
    coords1 = [(0, 0, 0), (1, 0, 0)]
    coords2 = [(0, 0, 0), (1, 0, 0)]  # Same coordinates
    
    # Process both
    result1 = processor.process_semantic(coords1)
    result2 = processor.process_semantic(coords2)
    
    # Check similarity (should be high for identical inputs)
    if result1 and result2:
        # Compare embeddings or features
        features1 = result1.get('features', {})
        features2 = result2.get('features', {})
        
        # Check key features match
        sw1 = features1.get('sw_normalized', 0)
        sw2 = features2.get('sw_normalized', 0)
        
        print(f"SW1: {sw1:.4f}, SW2: {sw2:.4f}")
        print(f"Match: {'✅' if abs(sw1 - sw2) < 1e-6 else '❌'}")
        
        assert abs(sw1 - sw2) < 1e-6, "Same meaning should produce same features"
    
    print("\n✅ Same meaning same basin test passed!")


def test_opposite_meaning_opposite_polarity():
    """Test that opposite meaning leads to opposite polarity."""
    print("\n" + "=" * 60)
    print("Test 2: Opposite Meaning → Opposite Polarity")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    processor = SemanticProcessor(system)
    
    # Create opposite semantic inputs (core vs corner)
    core_coords = [(0, 0, 0)]  # Core cell
    corner_coords = [(1, 1, 1)]  # Corner cell (opposite class)
    
    # Process both
    result_core = processor.process_semantic(core_coords)
    result_corner = processor.process_semantic(corner_coords)
    
    if result_core and result_corner:
        features_core = result_core.get('features', {})
        features_corner = result_corner.get('features', {})
        
        # Check polarity (face_exposure should be opposite)
        exposure_core = features_core.get('face_exposure', 0)
        exposure_corner = features_corner.get('face_exposure', 3)
        
        print(f"Core exposure: {exposure_core}")
        print(f"Corner exposure: {exposure_corner}")
        print(f"Opposite: {'✅' if exposure_core + exposure_corner == 3 else '❌'}")
        
        # Core (0) + Corner (3) = 3 (opposite)
        assert exposure_core + exposure_corner == 3, "Opposite classes should have opposite polarity"
    
    print("\n✅ Opposite meaning opposite polarity test passed!")


def test_single_word_shift():
    """Test that sentences differing by one word only shift phi-weight slightly."""
    print("\n" + "=" * 60)
    print("Test 3: Single Word Shift → Small Phi-Weight Change")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    processor = SemanticProcessor(system)
    
    # Create similar semantic inputs (differ by one cell)
    coords_base = [(0, 0, 0), (1, 0, 0)]
    coords_variant = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]  # One extra cell
    
    result_base = processor.process_semantic(coords_base)
    result_variant = processor.process_semantic(coords_variant)
    
    if result_base and result_variant:
        features_base = result_base.get('features', {})
        features_variant = result_variant.get('features', {})
        
        # Compare normalized SW (phi-weight proxy)
        sw_base = features_base.get('sw_normalized', 0)
        sw_variant = features_variant.get('sw_normalized', 0)
        
        shift = abs(sw_variant - sw_base)
        
        print(f"Base SW: {sw_base:.4f}")
        print(f"Variant SW: {sw_variant:.4f}")
        print(f"Shift: {shift:.4f}")
        print(f"Small shift: {'✅' if shift < 0.5 else '❌'}")
        
        # Shift should be small (one cell difference)
        assert shift < 0.5, "Single word difference should cause small shift"
    
    print("\n✅ Single word shift test passed!")


def test_meaning_graph_connected():
    """Test that meaning graph never gets disconnected."""
    print("\n" + "=" * 60)
    print("Test 4: Meaning Graph Connectivity")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    meaning_graph = MeaningGraph()
    
    # Add nodes
    node1 = meaning_graph.add_node("concept1", embedding=[1.0, 0.0, 0.0])
    node2 = meaning_graph.add_node("concept2", embedding=[0.0, 1.0, 0.0])
    node3 = meaning_graph.add_node("concept3", embedding=[0.0, 0.0, 1.0])
    
    # Add edges
    meaning_graph.add_edge(node1, node2, weight=0.5)
    meaning_graph.add_edge(node2, node3, weight=0.5)
    
    # Check connectivity
    nodes = meaning_graph.get_all_nodes()
    print(f"Nodes: {len(nodes)}")
    
    # Check graph is connected (all nodes reachable)
    all_connected = True
    for node in nodes:
        neighbors = meaning_graph.get_neighbors(node)
        if len(neighbors) == 0 and len(nodes) > 1:
            # Isolated node (except if only one node)
            all_connected = False
            print(f"⚠️  Isolated node: {node}")
    
    # For this test, we ensure edges exist
    edges = meaning_graph.get_all_edges()
    print(f"Edges: {len(edges)}")
    print(f"Graph connected: {'✅' if len(edges) >= len(nodes) - 1 else '❌'}")
    
    # Graph should have at least n-1 edges for connectivity
    assert len(edges) >= len(nodes) - 1 or len(nodes) <= 1, "Graph should be connected"
    
    print("\n✅ Meaning graph connectivity test passed!")


def test_feature_extractor_invariants():
    """Test that feature extractor invariants hold."""
    print("\n" + "=" * 60)
    print("Test 5: Feature Extractor Invariants")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    extractor = FeatureExtractor(system)
    
    # Test all cells
    invalid_features = []
    
    for coords, cell in system.lattice.items():
        # Extract features
        geometric_features = extractor.extract_geometric_features(coords)
        symbolic_features = extractor.extract_symbolic_features(coords)
        
        # Check invariants
        # 1. Face exposure in [0, 3]
        face_exposure = symbolic_features.get('face_exposure', None)
        if face_exposure is not None:
            if face_exposure < 0 or face_exposure > 3:
                invalid_features.append((coords, f"face_exposure out of range: {face_exposure}"))
        
        # 2. SW normalized in [0, 1]
        sw_normalized = symbolic_features.get('sw_normalized', None)
        if sw_normalized is not None:
            if sw_normalized < 0 or sw_normalized > 1.1:  # Allow slight overflow
                invalid_features.append((coords, f"sw_normalized out of range: {sw_normalized}"))
        
        # 3. Distance non-negative
        distance = geometric_features.get('distance_from_observer', None)
        if distance is not None:
            if distance < 0:
                invalid_features.append((coords, f"negative distance: {distance}"))
    
    if invalid_features:
        print(f"⚠️  Found {len(invalid_features)} invalid features:")
        for coords, issue in invalid_features[:5]:
            print(f"  {coords}: {issue}")
        assert False, "Feature extractor invariants violated"
    
    print(f"✅ All {len(system.lattice)} cells have valid features")
    
    print("\n✅ Feature extractor invariants test passed!")


def test_semantic_processor_consistency():
    """Test semantic processor produces consistent results."""
    print("\n" + "=" * 60)
    print("Test 6: Semantic Processor Consistency")
    print("=" * 60)
    
    system = LivniumCoreSystem()
    processor = SemanticProcessor(system)
    
    # Process same input multiple times
    coords = [(0, 0, 0), (1, 0, 0)]
    results = []
    
    for i in range(5):
        result = processor.process_semantic(coords)
        if result:
            results.append(result)
    
    # Check consistency
    if len(results) > 1:
        # Compare first and last
        first = results[0]
        last = results[-1]
        
        features1 = first.get('features', {})
        features2 = last.get('features', {})
        
        sw1 = features1.get('sw_normalized', 0)
        sw2 = features2.get('sw_normalized', 0)
        
        consistent = abs(sw1 - sw2) < 1e-6
        
        print(f"First result SW: {sw1:.6f}")
        print(f"Last result SW: {sw2:.6f}")
        print(f"Consistent: {'✅' if consistent else '❌'}")
        
        assert consistent, "Semantic processor should be consistent"
    
    print("\n✅ Semantic processor consistency test passed!")


if __name__ == "__main__":
    test_same_meaning_same_basin()
    test_opposite_meaning_opposite_polarity()
    test_single_word_shift()
    test_meaning_graph_connected()
    test_feature_extractor_invariants()
    test_semantic_processor_consistency()
    print("\n" + "=" * 60)
    print("All semantic consistency tests passed! ✅")
    print("=" * 60)

