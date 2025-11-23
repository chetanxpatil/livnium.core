"""
Entangled Basins: Classical Hidden-Variable Model

Implements Idea A - "Entangled Basins" via Shared Seed.

This creates deterministic correlation between two machines through shared
initial conditions (seed + config). Both machines evolve identically for the
same inputs, creating apparent "non-local correlation" without communication.

Key Concept:
- Same seed + same input → same basin signature
- Demonstrates classical hidden-variable model
- Proves determinism of Livnium geometry dynamics
"""

import random
import numpy as np
import hashlib
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

from ..classical.livnium_core_system import LivniumCoreSystem
from ..config import LivniumCoreConfig
from ..search.multi_basin_search import MultiBasinSearch, Basin


@dataclass
class CorrelationResult:
    """Result of correlation verification."""
    correlated: bool
    signature_a: Tuple
    signature_b: Tuple
    match_details: Dict[str, Any]


class SharedSeedManager:
    """
    Manages shared random seeds for deterministic initialization.
    
    Ensures both machines use identical random seeds to guarantee
    deterministic evolution.
    """
    
    @staticmethod
    def set_seed(seed: int):
        """
        Set random seed for deterministic behavior.
        
        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
    
    @staticmethod
    def initialize_shared_system(
        seed: int,
        lattice_size: int = 3,
        config: Optional[LivniumCoreConfig] = None
    ) -> LivniumCoreSystem:
        """
        Initialize identical systems on both machines.
        
        Args:
            seed: Random seed for deterministic initialization
            lattice_size: Size of the lattice (default: 3 for 3×3×3)
            config: Optional custom config (if None, uses defaults)
            
        Returns:
            Initialized LivniumCoreSystem
        """
        # Set seeds for deterministic behavior
        SharedSeedManager.set_seed(seed)
        
        # Use provided config or create default
        if config is None:
            config = LivniumCoreConfig(
                lattice_size=lattice_size,
                enable_semantic_polarity=True,
                enable_symbolic_weight=True,
                enable_face_exposure=True
            )
        
        system = LivniumCoreSystem(config)
        return system


class BasinSignatureGenerator:
    """
    Generates deterministic signatures from basin states.
    
    Creates hash-like signatures that can be compared between machines
    to verify correlation.
    """
    
    @staticmethod
    def compute_basin_signature(
        system: LivniumCoreSystem,
        active_coords: Optional[List[Tuple[int, int, int]]] = None
    ) -> Tuple:
        """
        Compute a deterministic signature for a basin state.
        
        Args:
            system: LivniumCoreSystem
            active_coords: Optional list of active coordinates.
                          If None, uses all cells with non-zero SW.
            
        Returns:
            Tuple signature that uniquely identifies the basin state
        """
        if active_coords is None:
            # Use all cells with non-zero symbolic weight
            active_coords = [
                coords for coords, cell in system.lattice.items()
                if cell.symbolic_weight > 0.1
            ]
        
        if not active_coords:
            return tuple()  # Empty basin
        
        # Create signature from (coords, SW) pairs
        signature_parts = []
        for coords in sorted(active_coords):
            cell = system.get_cell(coords)
            if cell:
                # Round to avoid floating point issues
                sw = round(cell.symbolic_weight, 3)
                signature_parts.append((coords, sw))
        
        return tuple(signature_parts)
    
    @staticmethod
    def compute_basin_hash(signature: Tuple) -> str:
        """
        Compute a hash string from basin signature.
        
        Args:
            signature: Basin signature tuple
            
        Returns:
            Hexadecimal hash string
        """
        # Convert signature to string and hash
        sig_str = str(signature).encode('utf-8')
        return hashlib.sha256(sig_str).hexdigest()[:16]  # First 16 chars


class TextEncoder:
    """
    Simple text encoder that maps text to initial geometric patterns.
    
    For Idea A, we need a deterministic way to encode text into the system.
    This creates initial coordinates based on text content.
    """
    
    @staticmethod
    def encode_text_to_coords(
        text: str,
        system: LivniumCoreSystem,
        max_coords: int = 10
    ) -> List[Tuple[int, int, int]]:
        """
        Encode text into initial coordinates deterministically.
        
        Args:
            text: Input text string
            system: LivniumCoreSystem
            max_coords: Maximum number of coordinates to generate
            
        Returns:
            List of coordinates to use as initial basin
        """
        # Hash text to get deterministic seed
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)  # Use first 8 hex chars as seed
        
        # Set seed for deterministic coordinate generation
        random.seed(seed)
        np.random.seed(seed)
        
        # Get lattice bounds
        lattice_size = system.config.lattice_size
        bound = (lattice_size - 1) // 2
        
        # Generate coordinates deterministically
        coords = []
        for i in range(max_coords):
            x = random.randint(-bound, bound)
            y = random.randint(-bound, bound)
            z = random.randint(-bound, bound)
            coords.append((x, y, z))
        
        return coords


class EntangledBasinsProcessor:
    """
    Main processor for Idea A: Entangled Basins.
    
    Handles the full protocol:
    1. Initialize shared system
    2. Process input to basin
    3. Extract basin signature
    """
    
    def __init__(
        self,
        seed: int,
        lattice_size: int = 3,
        max_evolution_steps: int = 100
    ):
        """
        Initialize processor.
        
        Args:
            seed: Shared random seed
            lattice_size: Lattice size (default: 3)
            max_evolution_steps: Maximum steps for basin evolution
        """
        self.seed = seed
        self.lattice_size = lattice_size
        self.max_evolution_steps = max_evolution_steps
        self.system: Optional[LivniumCoreSystem] = None
        self.search: Optional[MultiBasinSearch] = None
    
    def initialize(self) -> LivniumCoreSystem:
        """
        Initialize the shared system.
        
        Returns:
            Initialized LivniumCoreSystem
        """
        self.system = SharedSeedManager.initialize_shared_system(
            self.seed,
            self.lattice_size
        )
        self.search = MultiBasinSearch(
            max_basins=5,
            base_alpha=0.10,
            base_beta=0.15
        )
        return self.system
    
    def process_to_basin(
        self,
        input_text: str,
        verbose: bool = False
    ) -> Tuple:
        """
        Process input text and let system evolve to basin.
        
        Args:
            input_text: Input text string
            verbose: If True, print progress
            
        Returns:
            Basin signature tuple
        """
        if self.system is None:
            self.initialize()
        
        # Encode text to initial coordinates
        initial_coords = TextEncoder.encode_text_to_coords(
            input_text,
            self.system
        )
        
        if verbose:
            print(f"Encoded '{input_text}' to {len(initial_coords)} coordinates")
        
        # Create initial basin
        basin = self.search.add_basin(initial_coords, self.system)
        
        # Evolve system until convergence
        for step in range(self.max_evolution_steps):
            self.search.update_all_basins(self.system)
            
            # Check for convergence (single alive basin)
            alive_basins = [b for b in self.search.basins if b.is_alive]
            if len(alive_basins) <= 1:
                if verbose:
                    print(f"Converged after {step + 1} steps")
                break
        
        # Get final basin coordinates
        alive_basins = [b for b in self.search.basins if b.is_alive]
        if alive_basins:
            final_basin = alive_basins[0]
            final_coords = final_basin.active_coords
        else:
            # Fallback: use initial coordinates
            final_coords = initial_coords
        
        # Compute signature
        signature = BasinSignatureGenerator.compute_basin_signature(
            self.system,
            final_coords
        )
        
        return signature


class CorrelationVerifier:
    """
    Verifies correlation between two basin signatures.
    """
    
    @staticmethod
    def verify_correlation(
        signature_a: Tuple,
        signature_b: Tuple,
        tolerance: float = 0.01
    ) -> CorrelationResult:
        """
        Verify that both machines reached the same basin.
        
        Args:
            signature_a: Basin signature from machine A
            signature_b: Basin signature from machine B
            tolerance: Tolerance for floating point comparison
            
        Returns:
            CorrelationResult with verification details
        """
        # Exact match
        exact_match = signature_a == signature_b
        
        # Detailed comparison
        match_details = {
            'exact_match': exact_match,
            'length_a': len(signature_a),
            'length_b': len(signature_b),
            'length_match': len(signature_a) == len(signature_b)
        }
        
        if exact_match:
            match_details['match_type'] = 'exact'
        elif len(signature_a) == len(signature_b) and len(signature_a) > 0:
            # Check if coordinates match (SW might differ slightly)
            coords_a = set(c[0] for c in signature_a)
            coords_b = set(c[0] for c in signature_b)
            coords_match = coords_a == coords_b
            match_details['coords_match'] = coords_match
            match_details['match_type'] = 'coords_only' if coords_match else 'different'
        else:
            match_details['match_type'] = 'different'
        
        return CorrelationResult(
            correlated=exact_match,
            signature_a=signature_a,
            signature_b=signature_b,
            match_details=match_details
        )


# Convenience functions for easy usage

def initialize_shared_system(seed: int, lattice_size: int = 3) -> LivniumCoreSystem:
    """
    Initialize identical systems on both machines.
    
    Args:
        seed: Random seed
        lattice_size: Lattice size
        
    Returns:
        Initialized LivniumCoreSystem
    """
    return SharedSeedManager.initialize_shared_system(seed, lattice_size)


def process_to_basin(
    system: LivniumCoreSystem,
    input_text: str,
    max_steps: int = 100
) -> Tuple:
    """
    Process input and let system fall into basin.
    
    Args:
        system: LivniumCoreSystem
        input_text: Input text string
        max_steps: Maximum evolution steps
        
    Returns:
        Basin signature tuple
    """
    # Create processor and initialize search
    processor = EntangledBasinsProcessor(
        seed=42,  # Default seed, will be overridden by system state
        max_evolution_steps=max_steps
    )
    processor.system = system
    processor.search = MultiBasinSearch(
        max_basins=5,
        base_alpha=0.10,
        base_beta=0.15
    )
    return processor.process_to_basin(input_text)


def verify_correlation(signature_a: Tuple, signature_b: Tuple) -> bool:
    """
    Verify that both machines reached same basin.
    
    Args:
        signature_a: Basin signature from machine A
        signature_b: Basin signature from machine B
        
    Returns:
        True if correlated, False otherwise
    """
    result = CorrelationVerifier.verify_correlation(signature_a, signature_b)
    return result.correlated

