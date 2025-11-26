"""
Test assertions for DataCube.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.classical.datacube import DataCube, DataCell


def test_basic_initialization():
    """Test basic DataCube initialization."""
    cube = DataCube(size=2)
    
    assert cube.size == 2, "Size should be 2"
    assert len(cube.lattice) == 8, "2×2×2 cube should have 8 cells"
    
    cube = DataCube(size=4)
    assert cube.size == 4, "Size should be 4"
    assert len(cube.lattice) == 64, "4×4×4 cube should have 64 cells"


def test_invalid_size():
    """Test that invalid sizes raise errors."""
    # Test odd size
    try:
        cube = DataCube(size=3)
        assert False, "Odd size should raise ValueError"
    except ValueError:
        pass  # Expected
    
    # Test size < 2
    try:
        cube = DataCube(size=1)
        assert False, "Size < 2 should raise ValueError"
    except ValueError:
        pass  # Expected
    
    # Test size = 0
    try:
        cube = DataCube(size=0)
        assert False, "Size = 0 should raise ValueError"
    except ValueError:
        pass  # Expected


def test_no_center_cell():
    """Test that even-dimensional cubes have no center cell at (0,0,0)."""
    cube = DataCube(size=4)
    
    # For N=4, coordinates are {-1, 0, 1} (no center at 0,0,0)
    cell = cube.get_cell((0, 0, 0))
    assert cell is not None, "Cell at (0,0,0) should exist for N=4"
    
    # But for N=2, check coordinate range
    cube2 = DataCube(size=2)
    # N=2 should have coordinates {0} only
    cell_000 = cube2.get_cell((0, 0, 0))
    assert cell_000 is not None, "Cell at (0,0,0) should exist for N=2"


def test_data_storage():
    """Test data storage and retrieval."""
    cube = DataCube(size=2)
    
    # Set data
    cube.set_data((0, 0, 0), 42.0)
    assert cube.get_data((0, 0, 0)) == 42.0, "Should retrieve stored data"
    
    # Set different data types
    cube.set_data((0, 0, 1), "test")
    assert cube.get_data((0, 0, 1)) == "test", "Should store string data"
    
    cube.set_data((0, 1, 0), [1, 2, 3])
    assert cube.get_data((0, 1, 0)) == [1, 2, 3], "Should store list data"


def test_invalid_coordinates():
    """Test that invalid coordinates raise errors or return None."""
    cube = DataCube(size=2)
    
    # set_data should raise ValueError for out-of-bounds
    try:
        cube.set_data((10, 10, 10), 1.0)
        assert False, "Out of bounds coordinates should raise ValueError"
    except ValueError:
        pass  # Expected
    
    # get_data should return None for out-of-bounds (doesn't raise)
    result = cube.get_data((10, 10, 10))
    assert result is None, "Out of bounds get_data should return None"


def test_clear():
    """Test clearing all data."""
    cube = DataCube(size=2)
    
    # Set data in multiple cells
    cube.set_data((0, 0, 0), 1.0)
    cube.set_data((0, 0, 1), 2.0)
    cube.set_data((0, 1, 0), 3.0)
    
    # Clear
    cube.clear()
    
    # Check all data is None
    for coords, cell in cube.lattice.items():
        assert cell.get_data() is None, f"Cell at {coords} should be None after clear"


def test_to_numpy():
    """Test conversion to numpy array."""
    cube = DataCube(size=2)
    
    # Set some data
    cube.set_data((0, 0, 0), 1.0)
    cube.set_data((0, 0, 1), 2.0)
    
    arr = cube.to_numpy()
    assert arr.shape == (2, 2, 2), "Array shape should match cube size"
    assert arr.dtype == np.float32, "Default dtype should be float32"
    
    # Test with custom dtype
    arr_int = cube.to_numpy(dtype=np.int32)
    assert arr_int.dtype == np.int32, "Should respect custom dtype"


def test_from_numpy():
    """Test loading from numpy array."""
    cube = DataCube(size=2)
    
    # Create test array
    arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    
    cube.from_numpy(arr)
    
    # Check data was loaded
    assert cube.get_data((0, 0, 0)) == 1.0, "Data should be loaded correctly"
    
    # Test invalid shape
    try:
        invalid_arr = np.array([1, 2, 3])
        cube.from_numpy(invalid_arr)
        assert False, "Invalid array shape should raise ValueError"
    except ValueError:
        pass  # Expected


def test_numpy_roundtrip():
    """Test numpy conversion roundtrip."""
    cube = DataCube(size=4)
    
    # Create random data
    original_arr = np.random.rand(4, 4, 4).astype(np.float32)
    
    # Load into cube
    cube.from_numpy(original_arr)
    
    # Convert back
    recovered_arr = cube.to_numpy()
    
    # Check they match
    assert np.allclose(original_arr, recovered_arr), "Roundtrip should preserve data"


def test_get_all_data():
    """Test getting all data as dictionary."""
    cube = DataCube(size=2)
    
    cube.set_data((0, 0, 0), 1.0)
    cube.set_data((0, 0, 1), 2.0)
    
    all_data = cube.get_all_data()
    assert isinstance(all_data, dict), "Should return dictionary"
    assert len(all_data) == 8, "Should have data for all 8 cells"
    assert all_data[(0, 0, 0)] == 1.0, "Should include set data"


def test_data_cell():
    """Test DataCell class."""
    cell = DataCell(coordinates=(1, 2, 3))
    
    assert cell.coordinates == (1, 2, 3), "Cell should store coordinates"
    assert cell.get_data() is None, "New cell should have None data"
    
    cell.set_data(42.0)
    assert cell.get_data() == 42.0, "Cell should store and retrieve data"


def test_repr():
    """Test string representation."""
    cube = DataCube(size=2)
    repr_str = repr(cube)
    
    assert "DataCube" in repr_str, "Repr should include class name"
    assert "size=2" in repr_str, "Repr should include size"
    assert "cells=8" in repr_str, "Repr should include cell count"


if __name__ == "__main__":
    print("Running DataCube tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_invalid_size()
    print("✓ Invalid size")
    
    test_no_center_cell()
    print("✓ No center cell")
    
    test_data_storage()
    print("✓ Data storage")
    
    test_invalid_coordinates()
    print("✓ Invalid coordinates")
    
    test_clear()
    print("✓ Clear")
    
    test_to_numpy()
    print("✓ To numpy")
    
    test_from_numpy()
    print("✓ From numpy")
    
    test_numpy_roundtrip()
    print("✓ Numpy roundtrip")
    
    test_get_all_data()
    print("✓ Get all data")
    
    test_data_cell()
    print("✓ Data cell")
    
    test_repr()
    print("✓ Repr")
    
    print("\nAll tests passed! ✓")

