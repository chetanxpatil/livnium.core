"""
Livnium Core System: Fully Generalized N×N×N Implementation

Implements the complete Livnium Core System specification for ANY odd N ≥ 3.

**Fully Generalized - No 3×3×3 Bias:**
- Works for N=3, 5, 7, 9, 11, ... (any odd integer ≥ 3)
- Alphabet Σ(N) scales with N³ (not fixed to 27 symbols)
- All formulas work for general N
- Zero assumptions about lattice size

**All 7 Axioms Defined (A1–A6 fully implemented, A7 infrastructure ready):**
- A1: Canonical Spatial Alphabet (N×N×N lattice, Σ(N) with N³ symbols) ✅
- A2: Observer Anchor (Global Observer at (0,0,0)) ✅
- A3: Symbolic Weight Law (SW = 9·f, independent of N) ✅
- A4: Dynamic Law (90° rotations only, 24-element group) ✅
- A5: Semantic Polarity (cos(θ), N-invariant) ✅
- A6: Activation Rule (Local Observer, N-invariant) ✅
- A7: Cross-Lattice Coupling (Wreath-product, infrastructure ready) ⚠️

**General Formulas (any odd N ≥ 3):**
- Total SW: ΣSW(N) = 54(N-2)² + 216(N-2) + 216
- Class counts: Core=(N-2)³, Centers=6(N-2)², Edges=12(N-2), Corners=8
- Verified: N=3→486, N=5→1350, N=7→2646

With feature switches to enable/disable components.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

# Try to import numba for acceleration (optional dependency)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from ..config import LivniumCoreConfig


class CellClass(Enum):
    """Cell classification based on face exposure."""
    CORE = 0      # f = 0, SW = 0
    CENTER = 1    # f = 1, SW = 9
    EDGE = 2      # f = 2, SW = 18
    CORNER = 3    # f = 3, SW = 27


class RotationAxis(Enum):
    """Rotation axes for 90° quarter-turns."""
    X = 0
    Y = 1
    Z = 2


@dataclass
class LatticeCell:
    """
    A single cell in the 3×3×3 lattice.
    
    Coordinates: (x, y, z) where x, y, z ∈ {-(N-1)/2, ..., (N-1)/2}
    For N=3: coordinates ∈ {-1, 0, 1}
    """
    coordinates: Tuple[int, int, int]
    symbol: Optional[str] = None  # Symbol from alphabet (0, a...z)
    face_exposure: Optional[int] = None  # f ∈ {0, 1, 2, 3}
    symbolic_weight: Optional[float] = None  # SW = 9·f
    cell_class: Optional[CellClass] = None
    
    def __post_init__(self):
        """Calculate face exposure and symbolic weight if enabled."""
        if self.face_exposure is None:
            self.face_exposure = self._calculate_face_exposure()
        
        if self.symbolic_weight is None:
            self.symbolic_weight = 9.0 * self.face_exposure
        
        if self.cell_class is None:
            self.cell_class = self._classify_cell()
    
    def _calculate_face_exposure(self) -> int:
        """
        Calculate face exposure: number of coordinates on boundary.
        
        For N×N×N lattice with coordinates in {-(N-1)/2, ..., (N-1)/2}:
        - f = number of coordinates at boundary (|coord| == (N-1)/2)
        """
        x, y, z = self.coordinates
        N = 3  # Default, should be passed from system
        boundary = (N - 1) // 2
        
        f = 0
        if abs(x) == boundary:
            f += 1
        if abs(y) == boundary:
            f += 1
        if abs(z) == boundary:
            f += 1
        
        return f
    
    def _classify_cell(self) -> CellClass:
        """Classify cell based on face exposure."""
        if self.face_exposure == 0:
            return CellClass.CORE
        elif self.face_exposure == 1:
            return CellClass.CENTER
        elif self.face_exposure == 2:
            return CellClass.EDGE
        elif self.face_exposure == 3:
            return CellClass.CORNER
        else:
            raise ValueError(f"Invalid face exposure: {self.face_exposure}")


@dataclass
class Observer:
    """Observer in the Livnium Core System."""
    coordinates: Tuple[int, int, int]
    is_global: bool = False
    is_local: bool = False
    
    def __init__(self, coordinates: Tuple[int, int, int], is_global: bool = False):
        self.coordinates = coordinates
        self.is_global = is_global
        self.is_local = not is_global


class RotationGroup:
    """
    24-element rotation group for cube rotations.
    
    All rotations are 90° quarter-turns about X, Y, or Z axes.
    """
    
    @staticmethod
    def get_rotation_matrix(axis: RotationAxis, quarter_turns: int = 1) -> np.ndarray:
        """
        Get rotation matrix for 90° quarter-turn.
        
        Args:
            axis: Rotation axis (X, Y, or Z)
            quarter_turns: Number of quarter-turns (1, 2, 3, or 4)
            
        Returns:
            3×3 rotation matrix
        """
        quarter_turns = quarter_turns % 4
        angle = quarter_turns * np.pi / 2
        
        if axis == RotationAxis.X:
            # Rotation about X-axis
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            return np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        elif axis == RotationAxis.Y:
            # Rotation about Y-axis
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            return np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        elif axis == RotationAxis.Z:
            # Rotation about Z-axis
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            return np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
    
    # Original implementation (commented for reference)
    # @staticmethod
    # def rotate_coordinates(coords: Tuple[int, int, int], 
    #                       axis: RotationAxis, 
    #                       quarter_turns: int = 1) -> Tuple[int, int, int]:
    #     """
    #     Rotate coordinates by 90° quarter-turn.
    #     
    #     Args:
    #         coords: Original coordinates (x, y, z)
    #         axis: Rotation axis
    #         quarter_turns: Number of quarter-turns
    #         
    #     Returns:
    #         Rotated coordinates
    #     """
    #     matrix = RotationGroup.get_rotation_matrix(axis, quarter_turns)
    #     coords_array = np.array(coords, dtype=float)
    #     rotated = matrix @ coords_array
    #     
    #     # Round to nearest integer (should be exact for 90° rotations)
    #     return tuple(int(round(x)) for x in rotated)
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _rotate_coordinates_numba(x: float, y: float, z: float, axis: int, quarter_turns: int):
        """
        Rotate coordinates by 90° quarter-turn (numba-accelerated).
        
        Args:
            x, y, z: Original coordinates
            axis: Rotation axis (0=X, 1=Y, 2=Z)
            quarter_turns: Number of quarter-turns (1, 2, 3, or 4)
            
        Returns:
            Rotated coordinates as tuple (rx, ry, rz)
        """
        quarter_turns = quarter_turns % 4
        angle = quarter_turns * np.pi / 2.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if axis == 0:  # X-axis
            rx = x
            ry = cos_a * y - sin_a * z
            rz = sin_a * y + cos_a * z
        elif axis == 1:  # Y-axis
            rx = cos_a * x + sin_a * z
            ry = y
            rz = -sin_a * x + cos_a * z
        else:  # Z-axis
            rx = cos_a * x - sin_a * y
            ry = sin_a * x + cos_a * y
            rz = z
        
        return (rx, ry, rz)
    
    @staticmethod
    def rotate_coordinates(coords: Tuple[int, int, int], 
                          axis: RotationAxis, 
                          quarter_turns: int = 1) -> Tuple[int, int, int]:
        """
        Rotate coordinates by 90° quarter-turn.
        
        Args:
            coords: Original coordinates (x, y, z)
            axis: Rotation axis
            quarter_turns: Number of quarter-turns
            
        Returns:
            Rotated coordinates
        """
        x, y, z = coords
        axis_val = axis.value
        
        if NUMBA_AVAILABLE:
            rotated = RotationGroup._rotate_coordinates_numba(float(x), float(y), float(z), axis_val, quarter_turns)
            return tuple(int(round(r)) for r in rotated)
        else:
            # Fallback to original implementation
            matrix = RotationGroup.get_rotation_matrix(axis, quarter_turns)
            coords_array = np.array(coords, dtype=float)
            rotated = matrix @ coords_array
            return tuple(int(round(x)) for x in rotated)


class LivniumCoreSystem:
    """
    Complete Livnium Core System implementation - Fully Generalized N×N×N.
    
    Implements all 7 axioms with feature switches.
    Works for any odd N ≥ 3 (not just 3×3×3).
    
    Key Features:
    - N×N×N lattice (N = any odd integer ≥ 3)
    - Alphabet Σ(N) with exactly N³ symbols
    - Symbolic Weight SW = 9·f (independent of N)
    - Class counts scale with N
    - Total SW = 54(N-2)² + 216(N-2) + 216
    """
    
    @staticmethod
    def generate_alphabet(n: int) -> List[str]:
        """
        Generate alphabet Σ(N) with exactly N³ symbols.
        
        For N=3: Uses {0, a, b, ..., z} (27 symbols)
        For N>3: Uses numeric encoding with optional prefix
        
        Args:
            n: Lattice size (N)
            
        Returns:
            List of N³ symbols
        """
        total_symbols = n ** 3
        
        if total_symbols <= 27:
            # Use standard alphabet for small N
            base_alphabet = ['0'] + [chr(ord('a') + i) for i in range(26)]
            if total_symbols <= len(base_alphabet):
                return base_alphabet[:total_symbols]
        
        # For larger N, use numeric encoding
        return [f"s{i:04d}" for i in range(total_symbols)]
    
    def __init__(self, config: Optional[LivniumCoreConfig] = None):
        """
        Initialize Livnium Core System.
        
        Args:
            config: Configuration with feature switches (default: all enabled)
        """
        self.config = config or LivniumCoreConfig()
        self.lattice_size = self.config.lattice_size
        
        # Validate lattice size
        if self.lattice_size < 3 or self.lattice_size % 2 == 0:
            raise ValueError(f"lattice_size must be odd and >= 3, got {self.lattice_size}")
        
        # Coordinate range: {-(N-1)/2, ..., (N-1)/2}
        self.coord_range = list(range(-(self.lattice_size - 1) // 2, 
                                      (self.lattice_size - 1) // 2 + 1))
        
        # Initialize lattice
        self.lattice: Dict[Tuple[int, int, int], LatticeCell] = {}
        self._initialize_lattice()
        
        # Observer system
        self.global_observer: Optional[Observer] = None
        self.local_observers: List[Observer] = []
        if self.config.enable_global_observer:
            self.global_observer = Observer((0, 0, 0), is_global=True)
        
        # Symbol mapping
        self.symbol_map: Dict[Tuple[int, int, int], str] = {}
        if self.config.enable_symbol_alphabet:
            self._initialize_symbols()
        
        # Invariants tracking
        self.initial_total_sw: Optional[float] = None
        self.initial_class_counts: Optional[Dict[CellClass, int]] = None
        if self.config.enable_sw_conservation or self.config.enable_class_count_conservation:
            self._record_initial_invariants()
        
        # Rotation history
        self.rotation_history: List[Dict] = []
    
    def _initialize_lattice(self):
        """Initialize all cells in the lattice."""
        for x in self.coord_range:
            for y in self.coord_range:
                for z in self.coord_range:
                    coords = (x, y, z)
                    cell = LatticeCell(coordinates=coords)
                    
                    # Update face exposure calculation with correct N
                    cell.face_exposure = self._calculate_face_exposure(coords)
                    cell.symbolic_weight = 9.0 * cell.face_exposure
                    cell.cell_class = self._classify_cell(cell.face_exposure)
                    
                    self.lattice[coords] = cell
    
    # Original implementation (commented for reference)
    # def _calculate_face_exposure(self, coords: Tuple[int, int, int]) -> int:
    #     """Calculate face exposure for coordinates."""
    #     x, y, z = coords
    #     boundary = (self.lattice_size - 1) // 2
    #     
    #     f = 0
    #     if abs(x) == boundary:
    #         f += 1
    #     if abs(y) == boundary:
    #         f += 1
    #     if abs(z) == boundary:
    #         f += 1
    #     
    #     return f
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_face_exposure_numba(x: int, y: int, z: int, boundary: int) -> int:
        """Calculate face exposure for coordinates (numba-accelerated)."""
        f = 0
        if abs(x) == boundary:
            f += 1
        if abs(y) == boundary:
            f += 1
        if abs(z) == boundary:
            f += 1
        return f
    
    def _calculate_face_exposure(self, coords: Tuple[int, int, int]) -> int:
        """Calculate face exposure for coordinates."""
        x, y, z = coords
        boundary = (self.lattice_size - 1) // 2
        
        if NUMBA_AVAILABLE:
            return self._calculate_face_exposure_numba(x, y, z, boundary)
        else:
            # Fallback to original implementation
            f = 0
            if abs(x) == boundary:
                f += 1
            if abs(y) == boundary:
                f += 1
            if abs(z) == boundary:
                f += 1
            return f
    
    def _classify_cell(self, face_exposure: int) -> CellClass:
        """Classify cell based on face exposure."""
        if face_exposure == 0:
            return CellClass.CORE
        elif face_exposure == 1:
            return CellClass.CENTER
        elif face_exposure == 2:
            return CellClass.EDGE
        elif face_exposure == 3:
            return CellClass.CORNER
        else:
            raise ValueError(f"Invalid face exposure: {face_exposure}")
    
    def _initialize_symbols(self):
        """
        Initialize alphabet Σ(N) with exactly N³ symbols.
        
        Creates reversible bijection: Σ(N) ⟷ ℒ_N
        """
        # Generate alphabet for this N
        alphabet = self.generate_alphabet(self.lattice_size)
        
        # Create bijective mapping: coordinates → symbols
        sorted_coords = sorted(self.lattice.keys())
        
        if len(alphabet) != len(sorted_coords):
            raise ValueError(
                f"Alphabet size mismatch: {len(alphabet)} symbols for {len(sorted_coords)} coordinates"
            )
        
        for idx, coords in enumerate(sorted_coords):
            self.symbol_map[coords] = alphabet[idx]
            self.lattice[coords].symbol = alphabet[idx]
    
    def _record_initial_invariants(self):
        """Record initial invariants for conservation checking."""
        if self.config.enable_sw_conservation:
            self.initial_total_sw = self.get_total_symbolic_weight()
        
        if self.config.enable_class_count_conservation:
            self.initial_class_counts = self.get_class_counts()
    
    def get_cell(self, coordinates: Tuple[int, int, int]) -> Optional[LatticeCell]:
        """Get cell at coordinates."""
        return self.lattice.get(coordinates)
    
    def get_symbol(self, coordinates: Tuple[int, int, int]) -> Optional[str]:
        """Get symbol at coordinates."""
        if not self.config.enable_symbol_alphabet:
            return None
        return self.symbol_map.get(coordinates)
    
    def set_symbol(self, coordinates: Tuple[int, int, int], symbol: str):
        """Set symbol at coordinates."""
        if not self.config.enable_symbol_alphabet:
            raise ValueError("Symbol alphabet not enabled")
        if coordinates not in self.lattice:
            raise ValueError(f"Invalid coordinates: {coordinates}")
        self.symbol_map[coordinates] = symbol
        self.lattice[coordinates].symbol = symbol
    
    # Original implementation (commented for reference)
    # def get_total_symbolic_weight(self) -> float:
    #     """Calculate total symbolic weight (ΣSW)."""
    #     if not self.config.enable_symbolic_weight:
    #         return 0.0
    #     
    #     total = 0.0
    #     for cell in self.lattice.values():
    #         if cell.symbolic_weight is not None:
    #             total += cell.symbolic_weight
    #     return total
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _sum_symbolic_weights_numba(weights: np.ndarray) -> float:
        """Sum symbolic weights (numba-accelerated)."""
        total = 0.0
        for i in prange(len(weights)):
            if not np.isnan(weights[i]):
                total += weights[i]
        return total
    
    def get_total_symbolic_weight(self) -> float:
        """Calculate total symbolic weight (ΣSW)."""
        if not self.config.enable_symbolic_weight:
            return 0.0
        
        if NUMBA_AVAILABLE:
            # Extract weights as numpy array for numba
            weights = np.array([cell.symbolic_weight if cell.symbolic_weight is not None else np.nan 
                               for cell in self.lattice.values()], dtype=np.float64)
            return self._sum_symbolic_weights_numba(weights)
        else:
            # Fallback to original implementation
            total = 0.0
            for cell in self.lattice.values():
                if cell.symbolic_weight is not None:
                    total += cell.symbolic_weight
            return total
    
    def get_class_counts(self) -> Dict[CellClass, int]:
        """Get counts for each cell class."""
        if not self.config.enable_class_structure:
            return {}
        
        counts = {cls: 0 for cls in CellClass}
        for cell in self.lattice.values():
            if cell.cell_class is not None:
                counts[cell.cell_class] += 1
        return counts
    
    def get_expected_total_sw(self) -> float:
        """
        Calculate expected total symbolic weight for N×N×N lattice.
        
        General Formula: ΣSW(N) = 54(N-2)² + 216(N-2) + 216
        
        Verified:
        - N=3 → 486
        - N=5 → 1350
        - N=7 → 2646
        
        This formula is rotationally invariant for all odd N ≥ 3.
        """
        N = self.lattice_size
        if N < 3 or N % 2 == 0:
            raise ValueError(f"N must be odd and >= 3, got {N}")
        return 54 * (N - 2) ** 2 + 216 * (N - 2) + 216
    
    def get_expected_class_counts(self) -> Dict[CellClass, int]:
        """
        Get expected class counts for N×N×N lattice (closed form).
        
        General Formula for any odd N ≥ 3:
        - Core: (N-2)³
        - Centers: 6(N-2)²
        - Edges: 12(N-2)
        - Corners: 8
        
        These counts are exact and rotationally invariant.
        """
        N = self.lattice_size
        if N < 3 or N % 2 == 0:
            raise ValueError(f"N must be odd and >= 3, got {N}")
        return {
            CellClass.CORE: (N - 2) ** 3,
            CellClass.CENTER: 6 * (N - 2) ** 2,
            CellClass.EDGE: 12 * (N - 2),
            CellClass.CORNER: 8
        }
    
    def rotate(self, axis: RotationAxis, quarter_turns: int = 1) -> Dict:
        """
        Apply 90° rotation to the entire lattice.
        
        Args:
            axis: Rotation axis (X, Y, or Z)
            quarter_turns: Number of quarter-turns (1, 2, 3, or 4)
            
        Returns:
            Dictionary with rotation result and invariant checks
        """
        if not self.config.enable_90_degree_rotations:
            raise ValueError("90° rotations not enabled")
        
        if quarter_turns % 4 == 0:
            # Full rotation = identity
            return {'rotated': False, 'reason': 'Full rotation (identity)'}
        
        # Rotate all coordinates
        new_lattice: Dict[Tuple[int, int, int], LatticeCell] = {}
        new_symbol_map: Dict[Tuple[int, int, int], str] = {}
        
        for old_coords, cell in self.lattice.items():
            new_coords = RotationGroup.rotate_coordinates(old_coords, axis, quarter_turns)
            
            # Create new cell with rotated coordinates
            new_cell = LatticeCell(coordinates=new_coords)
            new_cell.face_exposure = self._calculate_face_exposure(new_coords)
            new_cell.symbolic_weight = 9.0 * new_cell.face_exposure
            new_cell.cell_class = self._classify_cell(new_cell.face_exposure)
            
            # Copy symbol if enabled
            if self.config.enable_symbol_alphabet and old_coords in self.symbol_map:
                new_symbol_map[new_coords] = self.symbol_map[old_coords]
                new_cell.symbol = self.symbol_map[old_coords]
            
            new_lattice[new_coords] = new_cell
        
        # Update lattice
        self.lattice = new_lattice
        if self.config.enable_symbol_alphabet:
            self.symbol_map = new_symbol_map
        
        # Record rotation
        rotation_info = {
            'axis': axis.name,
            'quarter_turns': quarter_turns,
            'timestamp': __import__('time').time()
        }
        self.rotation_history.append(rotation_info)
        
        # Check invariants
        result = {
            'rotated': True,
            'axis': axis.name,
            'quarter_turns': quarter_turns,
            'invariants_preserved': True
        }
        
        if self.config.enable_sw_conservation:
            current_sw = self.get_total_symbolic_weight()
            expected_sw = self.get_expected_total_sw()
            sw_preserved = abs(current_sw - expected_sw) < 1e-6
            result['sw_preserved'] = sw_preserved
            result['total_sw'] = current_sw
            result['expected_sw'] = expected_sw
            if not sw_preserved:
                result['invariants_preserved'] = False
        
        if self.config.enable_class_count_conservation:
            current_counts = self.get_class_counts()
            expected_counts = self.get_expected_class_counts()
            counts_preserved = current_counts == expected_counts
            result['class_counts_preserved'] = counts_preserved
            result['current_counts'] = current_counts
            result['expected_counts'] = expected_counts
            if not counts_preserved:
                result['invariants_preserved'] = False
        
        return result
    
    # Original implementation (commented for reference)
    # def calculate_polarity(self, motion_vector: Tuple[float, float, float],
    #                      observer_coords: Optional[Tuple[int, int, int]] = None,
    #                      target_coords: Optional[Tuple[int, int, int]] = None) -> float:
    #     """
    #     Calculate semantic polarity: cos(θ) between motion vector and observer.
    #     
    #     For Global Observer at (0,0,0), polarity is calculated as:
    #     - Motion vector from observer to target, or
    #     - Direct motion vector if target_coords provided
    #     
    #     Args:
    #         motion_vector: Motion vector (vx, vy, vz) OR if target_coords provided, this is ignored
    #         observer_coords: Observer coordinates (default: global observer at (0,0,0))
    #         target_coords: Target coordinates (if provided, motion_vector is calculated from observer to target)
    #         
    #     Returns:
    #         Polarity value in [-1, 1]
    #     """
    #     if not self.config.enable_semantic_polarity:
    #         raise ValueError("Semantic polarity not enabled")
    #     
    #     if observer_coords is None:
    #         if self.global_observer is None:
    #             raise ValueError("No observer available")
    #         observer_coords = self.global_observer.coordinates
    #     
    #     # If target_coords provided, calculate motion vector from observer to target
    #     if target_coords is not None:
    #         observer_vec = np.array(observer_coords, dtype=float)
    #         target_vec = np.array(target_coords, dtype=float)
    #         motion_vec = target_vec - observer_vec
    #     else:
    #         # Use provided motion vector
    #         motion_vec = np.array(motion_vector, dtype=float)
    #         observer_vec = np.array(observer_coords, dtype=float)
    #     
    #     # Calculate angle
    #     dot_product = np.dot(observer_vec, motion_vec)
    #     observer_norm = np.linalg.norm(observer_vec)
    #     motion_norm = np.linalg.norm(motion_vec)
    #     
    #     # Special case: if observer is at origin (0,0,0), use motion vector direction
    #     if observer_norm < 1e-10:
    #         # Observer at origin: polarity is based on motion vector direction
    #         # For origin observer, we use the motion vector itself
    #         if motion_norm < 1e-10:
    #             return 0.0
    #         # Normalize motion vector and use its direction
    #         motion_normalized = motion_vec / motion_norm
    #         # For origin observer, polarity is the sign of motion toward positive direction
    #         # Simplified: use first non-zero component
    #         return float(np.clip(motion_normalized[0] if abs(motion_normalized[0]) > 1e-10 else motion_normalized[1] if abs(motion_normalized[1]) > 1e-10 else motion_normalized[2], -1.0, 1.0))
    #     
    #     if motion_norm < 1e-10:
    #         return 0.0
    #     
    #     cos_theta = dot_product / (observer_norm * motion_norm)
    #     return float(np.clip(cos_theta, -1.0, 1.0))
    
    # Numba-accelerated version
    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_polarity_numba(obs_x: float, obs_y: float, obs_z: float,
                                  mot_x: float, mot_y: float, mot_z: float) -> float:
        """Calculate polarity (numba-accelerated)."""
        # Calculate norms
        observer_norm = np.sqrt(obs_x*obs_x + obs_y*obs_y + obs_z*obs_z)
        motion_norm = np.sqrt(mot_x*mot_x + mot_y*mot_y + mot_z*mot_z)
        
        # Special case: observer at origin
        if observer_norm < 1e-10:
            if motion_norm < 1e-10:
                return 0.0
            # Use first non-zero component
            if abs(mot_x) > 1e-10:
                return np.clip(mot_x / motion_norm, -1.0, 1.0)
            elif abs(mot_y) > 1e-10:
                return np.clip(mot_y / motion_norm, -1.0, 1.0)
            else:
                return np.clip(mot_z / motion_norm, -1.0, 1.0)
        
        if motion_norm < 1e-10:
            return 0.0
        
        # Calculate dot product and cosine
        dot_product = obs_x*mot_x + obs_y*mot_y + obs_z*mot_z
        cos_theta = dot_product / (observer_norm * motion_norm)
        return np.clip(cos_theta, -1.0, 1.0)
    
    def calculate_polarity(self, motion_vector: Tuple[float, float, float],
                         observer_coords: Optional[Tuple[int, int, int]] = None,
                         target_coords: Optional[Tuple[int, int, int]] = None) -> float:
        """
        Calculate semantic polarity: cos(θ) between motion vector and observer.
        
        For Global Observer at (0,0,0), polarity is calculated as:
        - Motion vector from observer to target, or
        - Direct motion vector if target_coords provided
        
        Args:
            motion_vector: Motion vector (vx, vy, vz) OR if target_coords provided, this is ignored
            observer_coords: Observer coordinates (default: global observer at (0,0,0))
            target_coords: Target coordinates (if provided, motion_vector is calculated from observer to target)
            
        Returns:
            Polarity value in [-1, 1]
        """
        if not self.config.enable_semantic_polarity:
            raise ValueError("Semantic polarity not enabled")
        
        if observer_coords is None:
            if self.global_observer is None:
                raise ValueError("No observer available")
            observer_coords = self.global_observer.coordinates
        
        # If target_coords provided, calculate motion vector from observer to target
        if target_coords is not None:
            observer_vec = np.array(observer_coords, dtype=float)
            target_vec = np.array(target_coords, dtype=float)
            motion_vec = target_vec - observer_vec
        else:
            # Use provided motion vector
            motion_vec = np.array(motion_vector, dtype=float)
            observer_vec = np.array(observer_coords, dtype=float)
        
        if NUMBA_AVAILABLE:
            return float(self._calculate_polarity_numba(
                float(observer_vec[0]), float(observer_vec[1]), float(observer_vec[2]),
                float(motion_vec[0]), float(motion_vec[1]), float(motion_vec[2])
            ))
        else:
            # Fallback to original implementation
            dot_product = np.dot(observer_vec, motion_vec)
            observer_norm = np.linalg.norm(observer_vec)
            motion_norm = np.linalg.norm(motion_vec)
            
            if observer_norm < 1e-10:
                if motion_norm < 1e-10:
                    return 0.0
                motion_normalized = motion_vec / motion_norm
                return float(np.clip(motion_normalized[0] if abs(motion_normalized[0]) > 1e-10 else motion_normalized[1] if abs(motion_normalized[1]) > 1e-10 else motion_normalized[2], -1.0, 1.0))
            
            if motion_norm < 1e-10:
                return 0.0
            
            cos_theta = dot_product / (observer_norm * motion_norm)
            return float(np.clip(cos_theta, -1.0, 1.0))
    
    def set_local_observer(self, coordinates: Tuple[int, int, int]) -> Observer:
        """
        Set a Local Observer at specified coordinates.
        
        Args:
            coordinates: Observer coordinates
            
        Returns:
            Created Local Observer
        """
        if not self.config.enable_local_observer:
            raise ValueError("Local observer not enabled")
        
        observer = Observer(coordinates, is_global=False)
        observer.is_local = True
        self.local_observers.append(observer)
        return observer
    
    def get_system_summary(self) -> Dict:
        """Get complete system summary."""
        summary = {
            'lattice_size': self.lattice_size,
            'total_cells': len(self.lattice),
            'features_enabled': {
                '3x3x3_lattice': self.config.enable_3x3x3_lattice,
                'symbol_alphabet': self.config.enable_symbol_alphabet,
                'symbolic_weight': self.config.enable_symbolic_weight,
                'face_exposure': self.config.enable_face_exposure,
                'class_structure': self.config.enable_class_structure,
                '90_degree_rotations': self.config.enable_90_degree_rotations,
                'rotation_group': self.config.enable_rotation_group,
                'global_observer': self.config.enable_global_observer,
                'local_observer': self.config.enable_local_observer,
                'semantic_polarity': self.config.enable_semantic_polarity,
                'cross_lattice_coupling': self.config.enable_cross_lattice_coupling,
                'sw_conservation': self.config.enable_sw_conservation,
                'class_count_conservation': self.config.enable_class_count_conservation,
            }
        }
        
        if self.config.enable_symbolic_weight:
            summary['total_symbolic_weight'] = self.get_total_symbolic_weight()
            summary['expected_symbolic_weight'] = self.get_expected_total_sw()
        
        if self.config.enable_class_structure:
            summary['class_counts'] = self.get_class_counts()
            summary['expected_class_counts'] = self.get_expected_class_counts()
        
        if self.config.enable_global_observer:
            summary['global_observer'] = {
                'coordinates': self.global_observer.coordinates if self.global_observer else None
            }
        
        if self.config.enable_local_observer:
            summary['local_observers'] = [
                {'coordinates': obs.coordinates} for obs in self.local_observers
            ]
        
        summary['num_rotations'] = len(self.rotation_history)
        
        return summary

