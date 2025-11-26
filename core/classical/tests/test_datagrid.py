"""
Test assertions for DataGrid.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.classical.datagrid import DataGrid, GridCell


def test_basic_initialization():
    """Test basic DataGrid initialization."""
    grid = DataGrid(size=2)
    
    assert grid.size == 2, "Size should be 2"
    assert len(grid.lattice) == 4, "2×2 grid should have 4 cells"
    
    grid = DataGrid(size=3)
    assert grid.size == 3, "Size should be 3"
    assert len(grid.lattice) == 9, "3×3 grid should have 9 cells"
    
    grid = DataGrid(size=5)
    assert grid.size == 5, "Size should be 5"
    assert len(grid.lattice) == 25, "5×5 grid should have 25 cells"


def test_invalid_size():
    """Test that invalid sizes raise errors."""
    # Test size < 2
    try:
        grid = DataGrid(size=1)
        assert False, "Size < 2 should raise ValueError"
    except ValueError:
        pass  # Expected
    
    # Test size = 0
    try:
        grid = DataGrid(size=0)
        assert False, "Size = 0 should raise ValueError"
    except ValueError:
        pass  # Expected


def test_coordinate_ranges():
    """Test coordinate ranges for odd and even N."""
    # Even N=2: coordinates should be {0} (or appropriate range)
    grid_even = DataGrid(size=2)
    # Check that coordinates are valid
    cell = grid_even.get_cell((0, 0))
    assert cell is not None, "Cell at (0,0) should exist for N=2"
    
    # Odd N=3: coordinates should be {-1, 0, 1}
    grid_odd = DataGrid(size=3)
    assert grid_odd.get_cell((-1, -1)) is not None, "Cell at (-1,-1) should exist for N=3"
    assert grid_odd.get_cell((0, 0)) is not None, "Cell at (0,0) should exist for N=3"
    assert grid_odd.get_cell((1, 1)) is not None, "Cell at (1,1) should exist for N=3"


def test_data_storage():
    """Test data storage and retrieval."""
    grid = DataGrid(size=3)
    
    # Set data
    grid.set_data((0, 0), 42.0)
    assert grid.get_data((0, 0)) == 42.0, "Should retrieve stored data"
    
    # Set different data types
    grid.set_data((1, 0), "test")
    assert grid.get_data((1, 0)) == "test", "Should store string data"
    
    grid.set_data((0, 1), [1, 2, 3])
    assert grid.get_data((0, 1)) == [1, 2, 3], "Should store list data"


def test_invalid_coordinates():
    """Test that invalid coordinates raise errors."""
    grid = DataGrid(size=3)
    
    try:
        grid.set_data((10, 10), 1.0)
        assert False, "Out of bounds coordinates should raise ValueError"
    except ValueError:
        pass  # Expected


def test_clear():
    """Test clearing all data."""
    grid = DataGrid(size=3)
    
    # Set data in multiple cells
    grid.set_data((0, 0), 1.0)
    grid.set_data((0, 1), 2.0)
    grid.set_data((1, 0), 3.0)
    
    # Clear
    grid.clear()
    
    # Check all data is None
    for coords, cell in grid.lattice.items():
        assert cell.get_data() is None, f"Cell at {coords} should be None after clear"


def test_to_numpy():
    """Test conversion to numpy array."""
    grid = DataGrid(size=2)
    
    # Set some data
    grid.set_data((0, 0), 1.0)
    grid.set_data((0, 1), 2.0)
    
    arr = grid.to_numpy()
    assert arr.shape == (2, 2), "Array shape should match grid size"
    assert arr.dtype == np.float32, "Default dtype should be float32"
    
    # Test with custom dtype
    arr_int = grid.to_numpy(dtype=np.int32)
    assert arr_int.dtype == np.int32, "Should respect custom dtype"


def test_from_numpy():
    """Test loading from numpy array."""
    grid = DataGrid(size=2)
    
    # Create test array
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    grid.from_numpy(arr)
    
    # Check data was loaded
    assert grid.get_data((0, 0)) == 1.0, "Data should be loaded correctly"
    
    # Test invalid shape
    try:
        invalid_arr = np.array([1, 2, 3])
        grid.from_numpy(invalid_arr)
        assert False, "Invalid array shape should raise ValueError"
    except ValueError:
        pass  # Expected


def test_numpy_roundtrip():
    """Test numpy conversion roundtrip."""
    grid = DataGrid(size=5)
    
    # Create random data
    original_arr = np.random.rand(5, 5).astype(np.float32)
    
    # Load into grid
    grid.from_numpy(original_arr)
    
    # Convert back
    recovered_arr = grid.to_numpy()
    
    # Check they match
    assert np.allclose(original_arr, recovered_arr), "Roundtrip should preserve data"


def test_get_all_data():
    """Test getting all data as dictionary."""
    grid = DataGrid(size=2)
    
    grid.set_data((0, 0), 1.0)
    grid.set_data((0, 1), 2.0)
    
    all_data = grid.get_all_data()
    assert isinstance(all_data, dict), "Should return dictionary"
    assert len(all_data) == 4, "Should have data for all 4 cells"
    assert all_data[(0, 0)] == 1.0, "Should include set data"


def test_grid_cell():
    """Test GridCell class."""
    cell = GridCell(coordinates=(1, 2))
    
    assert cell.coordinates == (1, 2), "Cell should store coordinates"
    assert cell.get_data() is None, "New cell should have None data"
    
    cell.set_data(42.0)
    assert cell.get_data() == 42.0, "Cell should store and retrieve data"


def test_repr():
    """Test string representation."""
    grid = DataGrid(size=3)
    repr_str = repr(grid)
    
    assert "DataGrid" in repr_str, "Repr should include class name"
    assert "size=3" in repr_str, "Repr should include size"
    assert "cells=9" in repr_str, "Repr should include cell count"


def test_odd_and_even_sizes():
    """Test grids with various odd and even sizes."""
    for size in [2, 3, 4, 5, 6, 7]:
        grid = DataGrid(size=size)
        assert len(grid.lattice) == size * size, \
            f"Grid of size {size} should have {size*size} cells"
        
        # Test that we can set and get data
        grid.set_data((0, 0), size)
        assert grid.get_data((0, 0)) == size, \
            f"Should be able to store and retrieve data for size {size}"


if __name__ == "__main__":
    print("Running DataGrid tests...")
    
    test_basic_initialization()
    print("✓ Basic initialization")
    
    test_invalid_size()
    print("✓ Invalid size")
    
    test_coordinate_ranges()
    print("✓ Coordinate ranges")
    
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
    
    test_grid_cell()
    print("✓ Grid cell")
    
    test_repr()
    print("✓ Repr")
    
    test_odd_and_even_sizes()
    print("✓ Odd and even sizes")
    
    print("\nAll tests passed! ✓")

