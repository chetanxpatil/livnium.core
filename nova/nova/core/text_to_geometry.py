"""
Text-to-Geometry Interface: Raw Sentence → Geometric Collapse

This is NOT NLP encoding. This is pure physics.

Every character produces a tiny geometric impulse:
- SW perturbation
- Tension micro-vector  
- Polarity hint
- Divergence ripple

The geometry then naturally collapses into one meaning basin.

No tokenization. No embeddings. No preprocessing.
Just: raw text → field → collapse → meaning.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path

# Add project root to path (go up two levels: core -> nova -> project_root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.classical.livnium_core_system import LivniumCoreSystem, LivniumCoreConfig
from nova.core.geometric_token_learner import GeometricTokenLearner


class TextToGeometry:
    """
    Minimal interface to inject raw text directly into Livnium geometry.
    
    This is the simplest possible mapping:
    - 1 line of code per character
    - no vocabulary
    - no tokens
    - no embeddings
    
    Just physics: character → geometric impulse → collapse.
    """
    
    def __init__(self, geometry: Optional[LivniumCoreSystem] = None, 
                 lattice_size: int = 3,
                 impulse_scale: float = 0.1,
                 num_clusters: int = 100,
                 break_symmetry_for_snli: bool = False):
        """
        Initialize text-to-geometry interface.
        
        Args:
            geometry: Existing geometry (or creates new one)
            lattice_size: Size of geometry cube (3, 5, 7, ...)
            impulse_scale: Scale factor for character impulses (default: 0.1)
            num_clusters: Number of clusters for token learning
            break_symmetry_for_snli: If True, add small random noise to break perfect symmetry (SNLI only)
        """
        if geometry is None:
            config = LivniumCoreConfig(
                lattice_size=lattice_size,
                enable_symbolic_weight=True,
                enable_90_degree_rotations=True
            )
            self.geometry = LivniumCoreSystem(config)
        else:
            self.geometry = geometry
        
        self.impulse_scale = impulse_scale
        self.break_symmetry_for_snli = break_symmetry_for_snli
        self.injection_history = []
        # Track injected impulses per coordinate (preserved across rotations)
        self.injected_impulses: Dict[Tuple[int, int, int], float] = {}
        
        # Initialize geometric token learner with matching lattice size
        self.learner = GeometricTokenLearner(lattice_size=lattice_size, num_clusters=num_clusters)
        
        # SNLI ONLY: Break perfect symmetry to enable angular variation
        if self.break_symmetry_for_snli:
            self._apply_symmetry_breaking()
    
    def char_to_impulse(self, char: str) -> float:
        """
        Convert character to geometric impulse.
        
        This is the ONLY "encoding" needed - and it's not really encoding.
        It's just a physical mapping: character → energy.
        
        Returns:
            Impulse value (tiny SW perturbation)
        """
        # Simplest possible mapping: ord(char) % 27 → impulse
        # This gives 0-26 range, scaled by impulse_scale
        impulse = (ord(char) % 27) * self.impulse_scale
        return impulse
    
    def inject_sentence(self, sentence: str, 
                       collapse_steps: int = 10,
                       verbose: bool = False) -> Dict:
        """
        Inject entire raw sentence into geometry using token-based hashing.
        
        Process:
        1. Tokenize sentence
        2. For each token, hash to (x,y,z,impulse) and inject
        3. Geometry naturally collapses into meaning basin
        4. Return final state
        
        Args:
            sentence: Raw text string (no preprocessing)
            collapse_steps: Number of rotation steps for collapse
            verbose: Print progress
            
        Returns:
            Dict with final geometry state and metrics
        """
        if verbose:
            print(f"Injecting sentence: '{sentence[:50]}...' ({len(sentence)} chars)")
        
        # Step 1: Tokenize sentence
        tokens = self.learner.tokenize(sentence)
        
        # Step 2: Inject all token impulses using MD5 hash
        total_impulse = 0.0
        impulse_map = {}  # Track impulses per cell
        
        for token in tokens:
            # Get (x, y, z) coordinates and impulse from token hash
            coords, impulse = self.learner.token_hash(token)
            # Scale impulse
            impulse = impulse * self.impulse_scale
            total_impulse += abs(impulse)
            
            # Map hash coordinates to actual lattice coordinates
            # Hash produces [0, lattice_size-1], but lattice uses [-(N-1)/2, ..., (N-1)/2]
            x, y, z = coords
            coord_range = self.geometry.coord_range
            # Map 0..N-1 to coord_range
            x_mapped = coord_range[x % len(coord_range)]
            y_mapped = coord_range[y % len(coord_range)]
            z_mapped = coord_range[z % len(coord_range)]
            coords_tuple = (x_mapped, y_mapped, z_mapped)
            
            # Add impulse to cell's SW
            if coords_tuple not in impulse_map:
                impulse_map[coords_tuple] = 0.0
            impulse_map[coords_tuple] += impulse
        
        # Apply all impulses at once (synchronous injection)
        initial_sw = self.geometry.get_total_symbolic_weight()
        
        # Store impulses (will be preserved across rotations)
        for coords, impulse_value in impulse_map.items():
            # Accumulate impulses (multiple sentences can add to same cell)
            if coords in self.injected_impulses:
                self.injected_impulses[coords] += impulse_value
            else:
                self.injected_impulses[coords] = impulse_value
        
        # Apply stored impulses to current geometry
        self._apply_stored_impulses()
        
        final_sw_after_injection = self.geometry.get_total_symbolic_weight()
        
        if verbose:
            print(f"  Injected {total_impulse:.2f} total impulse across {len(impulse_map)} cells")
            print(f"  SW: {initial_sw:.2f} → {final_sw_after_injection:.2f}")
        
        # Step 2: Let geometry collapse naturally with physics-based collapse
        if verbose:
            print(f"  Collapsing geometry ({collapse_steps} steps) with physics-based collapse...")
        
        collapse_history = []
        for step in range(collapse_steps):
            # Apply physics-based collapse (entropy, diffusion, gentle decay)
            self._physics_collapse_step(step, collapse_steps)
            
            # Track metrics
            current_sw = self.geometry.get_total_symbolic_weight()
            sw_distribution = [cell.symbolic_weight for cell in self.geometry.lattice.values()]
            mean_sw = np.mean(sw_distribution)
            std_sw = np.std(sw_distribution)
            
            collapse_history.append({
                'step': step,
                'total_sw': current_sw,
                'mean_sw': mean_sw,
                'std_sw': std_sw
            })
        
        # Step 3: Extract final meaning state
        final_sw = self.geometry.get_total_symbolic_weight()
        final_sw_distribution = [cell.symbolic_weight for cell in self.geometry.lattice.values()]
        
        result = {
            'sentence': sentence,
            'sentence_length': len(sentence),
            'total_impulse': total_impulse,
            'initial_sw': initial_sw,
            'final_sw': final_sw,
            'sw_change': final_sw - initial_sw,
            'final_mean_sw': np.mean(final_sw_distribution),
            'final_std_sw': np.std(final_sw_distribution),
            'collapse_history': collapse_history,
            'geometry': self.geometry
        }
        
        # Record injection
        self.injection_history.append(result)
        
        if verbose:
            print(f"  Final SW: {final_sw:.2f} (change: {result['sw_change']:.2f})")
            print(f"  Collapse complete: mean={result['final_mean_sw']:.2f}, std={result['final_std_sw']:.2f}")
        
        return result
    
    def get_meaning_signature(self, sentence: str, collapse_steps: int = 10) -> np.ndarray:
        """
        Get meaning signature (SW distribution) for a sentence.
        
        This is the "fingerprint" of the sentence's meaning in geometry space.
        Different sentences → different SW distributions → different meanings.
        
        Returns:
            numpy array of final SW values (one per cell)
        """
        result = self.inject_sentence(sentence, collapse_steps=collapse_steps, verbose=False)
        sw_distribution = np.array([cell.symbolic_weight for cell in self.geometry.lattice.values()])
        return sw_distribution
    
    def get_signature_with_divergence(self, premise: str, hypothesis: str, collapse_steps: int = 12) -> np.ndarray:
        """
        Get signature for premise+hypothesis pair WITH divergence primitive.
        
        OM = direction of meaning of the premise
        LO = direction of meaning of the hypothesis
        Same geometry system, two inputs → two directions
        
        CRITICAL: LO (hypothesis) gets directional bias to break OM=LO symmetry.
        
        This implements the Divergence Law: divergence = 0.38 - alignment
        
        Returns:
            Extended signature array with divergence primitives
        """
        # 1. Premise signature (OM) - normal collapse
        premise_sig = self.get_meaning_signature(premise, collapse_steps=collapse_steps)
        self.reset_geometry()
        
        OM = premise_sig / (np.linalg.norm(premise_sig) + 1e-8)
        
        # 2. Hypothesis signature (LO) - with directional bias to break symmetry
        # Apply small rotation to LO's local frame before collapse
        from core.classical.livnium_core_system import RotationAxis
        import random
        
        # Rotate LO's geometry by small random angle to break symmetry
        # This ensures OM ≠ LO even for similar inputs
        rotation_axis = random.choice(list(RotationAxis))
        self.geometry.rotate(rotation_axis, quarter_turns=1)
        
        hyp_sig = self.get_meaning_signature(hypothesis, collapse_steps=collapse_steps)
        self.reset_geometry()
        
        LO = hyp_sig / (np.linalg.norm(hyp_sig) + 1e-8)
        
        # 3. Alignment
        cos_theta = float(np.dot(OM, LO))
        alignment = (cos_theta + 1.0) / 2.0
        
        # 4. Divergence law
        divergence = 0.38 - alignment
        
        # 5. Fracture (change in alignment)
        fracture = abs(alignment)
        
        # Combine: [premise_SW, hypothesis_SW, alignment, divergence, fracture]
        extended_sig = np.concatenate([
            premise_sig,
            hyp_sig,
            np.array([alignment, divergence, fracture])
        ])
        
        return extended_sig
    
    def _apply_symmetry_breaking(self):
        """
        SNLI ONLY: Break perfect symmetry by adding tiny random noise to SW values.
        
        NOTE: This is a secondary effect. The PRIMARY symmetry breaking happens in
        `get_signature_with_divergence()` where we tilt the LO (hypothesis) direction vector.
        That LO tilt is what actually fixes alignment (OM/LO identical → alignment=1.0 → divergence constant).
        
        This SW noise may help with other geometric aspects but does NOT affect alignment computation.
        
        DO NOT use this for law extraction (which requires perfect symmetry for invariant measurement).
        """
        if not self.break_symmetry_for_snli:
            return
        
        # Get current total SW (must be preserved)
        target_sw_sum = self.geometry.get_total_symbolic_weight()
        
        # Add small random noise to each cell's SW
        # Noise scale: 0.5% of mean SW per cell
        num_cells = len(self.geometry.lattice)
        mean_sw = target_sw_sum / num_cells if num_cells > 0 else 0.0
        noise_scale = mean_sw * 0.005  # 0.5% noise
        
        # Apply noise to each cell
        for coords, cell in self.geometry.lattice.items():
            if cell.symbolic_weight is not None:
                noise = np.random.normal(0, noise_scale)
                cell.symbolic_weight = max(0.0, cell.symbolic_weight + noise)  # Ensure non-negative
        
        # Renormalize to preserve total SW (conservation law)
        current_sw_sum = self.geometry.get_total_symbolic_weight()
        if current_sw_sum > 0:
            scale_factor = target_sw_sum / current_sw_sum
            for cell in self.geometry.lattice.values():
                if cell.symbolic_weight is not None:
                    cell.symbolic_weight *= scale_factor
    
    def compare_sentences(self, sentence1: str, sentence2: str, 
                         collapse_steps: int = 10) -> Dict:
        """
        Compare two sentences by their geometric meaning signatures.
        
        Returns:
            Dict with similarity metrics
        """
        # Get meaning signatures
        sig1 = self.get_meaning_signature(sentence1, collapse_steps)
        sig2 = self.get_meaning_signature(sentence2, collapse_steps)
        
        # Compute similarity metrics
        cosine_sim = np.dot(sig1, sig2) / (np.linalg.norm(sig1) * np.linalg.norm(sig2) + 1e-10)
        euclidean_dist = np.linalg.norm(sig1 - sig2)
        manhattan_dist = np.sum(np.abs(sig1 - sig2))
        
        return {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'manhattan_distance': manhattan_dist,
            'signature1': sig1,
            'signature2': sig2
        }
    
    def _apply_stored_impulses(self):
        """
        Apply stored impulses to current geometry.
        
        This is called after rotations to preserve injected impulses
        (since rotations reset SW to canonical values).
        """
        for coords, impulse_value in self.injected_impulses.items():
            if coords in self.geometry.lattice:
                cell = self.geometry.lattice[coords]
                # Get base SW (from face exposure) and add impulse
                base_sw = 9.0 * cell.face_exposure
                cell.symbolic_weight = base_sw + impulse_value
    
    def _physics_collapse_step(self, step: int, total_steps: int):
        """
        Physics-based collapse step with entropy, diffusion, and gentle decay.
        
        This prevents the collapse from being too deterministic and too strong,
        allowing patterns to survive and creating variation between different inputs.
        
        Steps:
        1. Add random thermal jitter (entropy) - prevents immediate basin-locking
        2. Apply neighbor diffusion - creates texture and preserves patterns
        3. Gentle decay (0.98-0.99) - allows patterns to survive instead of dying instantly
        4. Rotate occasionally to explore state space
        """
        from core.classical.livnium_core_system import RotationAxis
        import random
        
        # 1. Add random thermal jitter (entropy) - prevents immediate basin-locking
        # Scale jitter based on step (less jitter as we converge)
        jitter_scale = max(0.0, 0.05 * (1.0 - step / max(total_steps, 1)))  # Increased from 0.01 to 0.05
        for coords, cell in self.geometry.lattice.items():
            if cell.symbolic_weight is not None and jitter_scale > 0:
                # Use absolute value of SW to ensure positive scale
                sw_magnitude = abs(cell.symbolic_weight) + 1e-10
                jitter = np.random.normal(0, jitter_scale * sw_magnitude)
                cell.symbolic_weight = max(0.0, cell.symbolic_weight + jitter)
        
        # 2. Apply neighbor diffusion - creates texture and preserves patterns
        # Exchange 30% with neighbors (increased from 20% to 30%)
        diffusion_rate = 0.3
        new_sw = {}
        
        for coords, cell in self.geometry.lattice.items():
            if cell.symbolic_weight is None:
                continue
            
            # Get neighbor coordinates (6 neighbors: ±x, ±y, ±z)
            x, y, z = coords
            neighbors = [
                (x+1, y, z), (x-1, y, z),
                (x, y+1, z), (x, y-1, z),
                (x, y, z+1), (x, y, z-1)
            ]
            
            # Average neighbor SW (only count valid neighbors)
            neighbor_sw_sum = 0.0
            neighbor_count = 0
            for n_coords in neighbors:
                if n_coords in self.geometry.lattice:
                    n_cell = self.geometry.lattice[n_coords]
                    if n_cell.symbolic_weight is not None:
                        neighbor_sw_sum += n_cell.symbolic_weight
                        neighbor_count += 1
            
            if neighbor_count > 0:
                neighbor_avg = neighbor_sw_sum / neighbor_count
                # Mix current SW with neighbor average
                new_sw[coords] = (1.0 - diffusion_rate) * cell.symbolic_weight + diffusion_rate * neighbor_avg
            else:
                new_sw[coords] = cell.symbolic_weight
        
        # Apply diffusion
        for coords, sw_value in new_sw.items():
            if coords in self.geometry.lattice:
                self.geometry.lattice[coords].symbolic_weight = max(0.0, sw_value)
        
        # 3. Gentle decay (0.95-0.99) - allows patterns to survive
        # Reduced decay strength (was 0.98-0.99, now 0.95-0.99)
        decay_factor = 0.95 + 0.04 * (step / max(total_steps, 1))  # Increases from 0.95 to 0.99
        for cell in self.geometry.lattice.values():
            if cell.symbolic_weight is not None:
                cell.symbolic_weight *= decay_factor
        
        # 4. Rotate occasionally to explore state space (every 3 steps)
        if step % 3 == 0:
            axis = random.choice(list(RotationAxis))
            self.geometry.rotate(axis, quarter_turns=1)
            # Re-apply impulses after rotation
            self._apply_stored_impulses()
        
        # Renormalize to preserve total SW (conservation law)
        # But do it gently - don't force everything to center
        current_sw = self.geometry.get_total_symbolic_weight()
        if current_sw > 1e-10:
            # Only renormalize if SW drifted significantly
            initial_sw = self.geometry.get_expected_total_sw()
            if abs(current_sw - initial_sw) / initial_sw > 0.01:  # More than 1% drift
                scale_factor = initial_sw / current_sw
                for cell in self.geometry.lattice.values():
                    if cell.symbolic_weight is not None:
                        cell.symbolic_weight *= scale_factor
    
    def reset_geometry(self):
        """
        Reset geometry to initial state (clears all injected impulses).
        
        For SNLI: Automatically applies symmetry breaking after reset if break_symmetry_for_snli=True.
        For law extraction: Keeps perfect symmetry (no breaking).
        """
        # Clear injection history
        self.injection_history = []
        self.injected_impulses.clear()
        
        # Reset all cells to canonical state
        for cell in self.geometry.lattice.values():
            cell.symbolic_weight = cell.face_exposure * 9.0  # SW = 9f
        
        # SNLI ONLY: Break perfect symmetry after reset
        # This enables angular variation between premise/hypothesis
        if self.break_symmetry_for_snli:
            self._apply_symmetry_breaking()

