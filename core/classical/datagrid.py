"""
DataGrid: 2D N×N Resource Grids

DataGrids are 2D N×N grids (2×2, 3×3, 4×4, 5×5, ...) that serve
as resource containers, data staging areas, and I/O buffers.

**Critical Distinction:**
- **Omcubes** (3D, odd N ≥ 3): Livnium Core Universes - implement all axioms, collapse, recursion
- **DataGrids** (2D, any N ≥ 2): Resource Grids - NO axioms, NO collapse, NO computation
- **DataCubes** (3D, even N ≥ 2): Resource Grids - NO axioms, NO collapse, NO computation

DataGrids are NOT Livnium cores. They cannot:
- Execute Livnium collapse mechanics
- Implement symbolic weight (SW) system
- Use face exposure rules (3D concept)
- Perform recursive geometry operations
- Anchor observers (no 3D center cell)
- Maintain Livnium invariants

They CAN:
- Store data (lookup tables, feature maps)
- Act as I/O buffers (input snapshots, output surfaces)
- Serve as preprocessing/postprocessing containers
- Hold embeddings or temporary state
- Connect to Omcubes as data sources/sinks

Think of them as "2D RAM blocks around a 3D CPU":
- Omcubes = CPU (3D core geometry, computation)
- DataGrids = 2D RAM (resource/data layers, storage)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class GridCell:
    """
    A single cell in a 2D DataGrid.
    
    Unlike LatticeCell (in Omcubes), this has:
    - NO face exposure calculation (3D concept)
    - NO symbolic weight
    - NO cell class (core/center/edge/corner - 3D concept)
    - NO observer anchor (3D concept)
    - Just coordinates and data storage
    """
    coordinates: Tuple[int, int]
    data: Optional[Any] = None  # Can store any data type
    
    def set_data(self, value: Any) -> None:
        """Set data in this cell."""
        self.data = value
    
    def get_data(self) -> Optional[Any]:
        """Get data from this cell."""
        return self.data


class DataGrid:
    """
    2D resource grid (2×2, 3×3, 4×4, 5×5, ...).
    
    This is NOT a Livnium core. It cannot execute:
    - Livnium collapse mechanics (3D only)
    - Symbolic weight calculations (3D face exposure)
    - Face exposure rules (3D concept)
    - Recursive geometry (3D only)
    - Observer anchoring (3D center cell)
    - Rotation-based transformations (3D rotations)
    
    It is a simple 2D data container for:
    - Input/output buffers
    - Feature maps
    - Lookup tables
    - Temporary storage
    - Embedding carriers
    - Preprocessing/postprocessing data
    """
    
    def __init__(self, size: int):
        """
        Initialize DataGrid.
        
        Args:
            size: Integer ≥ 2 (2, 3, 4, 5, ...)
        
        Raises:
            ValueError: If size < 2
        """
        if size < 2:
            raise ValueError(f"DataGrid size must be >= 2, got {size}")
        
        self.size = size
        self.lattice: Dict[Tuple[int, int], GridCell] = {}
        self._initialize_lattice()
    
    def _initialize_lattice(self) -> None:
        """Initialize all cells in the 2D grid."""
        # Coordinate range: {-(N-1)/2, ..., (N-1)/2} for odd N
        #                  {-(N/2-1), ..., N/2-1} for even N
        if self.size % 2 == 0:
            # Even N: {-(N/2-1), ..., N/2-1}
            coord_range = list(range(-(self.size // 2 - 1), self.size // 2 + 1))
        else:
            # Odd N: {-(N-1)/2, ..., (N-1)/2}
            coord_range = list(range(-(self.size - 1) // 2, (self.size - 1) // 2 + 1))
        
        for x in coord_range:
            for y in coord_range:
                coords = (x, y)
                self.lattice[coords] = GridCell(coords)
    
    def get_cell(self, coords: Tuple[int, int]) -> Optional[GridCell]:
        """Get cell at coordinates."""
        return self.lattice.get(coords)
    
    def set_data(self, coords: Tuple[int, int], data: Any) -> None:
        """Set data at coordinates."""
        if coords not in self.lattice:
            raise ValueError(f"Coordinates {coords} out of bounds for {self.size}×{self.size} DataGrid")
        self.lattice[coords].set_data(data)
    
    def get_data(self, coords: Tuple[int, int]) -> Optional[Any]:
        """Get data at coordinates."""
        cell = self.get_cell(coords)
        return cell.get_data() if cell else None
    
    def get_all_data(self) -> Dict[Tuple[int, int], Any]:
        """Get all data as a dictionary mapping coordinates to values."""
        return {coords: cell.get_data() for coords, cell in self.lattice.items()}
    
    def clear(self) -> None:
        """Clear all data from cells."""
        for cell in self.lattice.values():
            cell.set_data(None)
    
    def to_numpy(self, dtype=np.float32) -> np.ndarray:
        """
        Convert DataGrid to numpy array.
        
        Returns:
            2D numpy array of shape (size, size)
        """
        arr = np.zeros((self.size, self.size), dtype=dtype)
        
        if self.size % 2 == 0:
            coord_range = list(range(-(self.size // 2 - 1), self.size // 2 + 1))
        else:
            coord_range = list(range(-(self.size - 1) // 2, (self.size - 1) // 2 + 1))
        
        for i, x in enumerate(coord_range):
            for j, y in enumerate(coord_range):
                data = self.get_data((x, y))
                if data is not None:
                    try:
                        arr[i, j] = float(data)
                    except (ValueError, TypeError):
                        arr[i, j] = 0.0
        
        return arr
    
    def from_numpy(self, arr: np.ndarray) -> None:
        """
        Load data from numpy array into DataGrid.
        
        Args:
            arr: 2D numpy array of shape (size, size)
        """
        if arr.shape != (self.size, self.size):
            raise ValueError(f"Array shape {arr.shape} does not match DataGrid size {self.size}")
        
        if self.size % 2 == 0:
            coord_range = list(range(-(self.size // 2 - 1), self.size // 2 + 1))
        else:
            coord_range = list(range(-(self.size - 1) // 2, (self.size - 1) // 2 + 1))
        
        for i, x in enumerate(coord_range):
            for j, y in enumerate(coord_range):
                self.set_data((x, y), float(arr[i, j]))
    
    def __repr__(self) -> str:
        return f"DataGrid(size={self.size}, cells={len(self.lattice)})"

