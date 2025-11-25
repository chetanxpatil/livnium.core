#!/usr/bin/env python3
"""
Shadow Rule 30 Model (Phase 6)

Phase 4 + Livnium geometric influence operator.

Livnium adds a small steering force to guide the PCA trajectory:
    y_{t+1} = F(y_t) + noise + Livnium(y_t)

This is the minimal Livnium implementation - just a geometric bias function.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import from PHASE2 to get the null space and reconstruction
phase2_code = project_root / "experiments" / "rule30" / "PHASE2" / "code"
sys.path.insert(0, str(phase2_code))

from four_bit_chaos_tracker import ChaosTracker15D

# Import Livnium force module (Phase 6 addition)
phase6_code = Path(__file__).parent
sys.path.insert(0, str(phase6_code))
from livnium_force import LivniumForce, create_default_livnium


class ShadowRule30Phase6:
    """
    Shadow Rule 30 with Livnium geometric influence.
    
    This extends Phase 4 by adding a minimal Livnium force operator
    that applies a small directional bias to guide the trajectory.
    """
    
    def __init__(self, pca_model, dynamics_model, decoder_model, tracker, 
                 n_components=8, target_energy=None, livnium_force=None):
        """
        Initialize Shadow Rule 30 with Livnium.
        
        Args:
            pca_model: Fitted PCA model (from sklearn)
            dynamics_model: Fitted dynamics model (dict with 'model' key)
            decoder_model: Fitted decoder model (LogisticRegression)
            tracker: ChaosTracker15D instance (for reconstruction)
            n_components: Number of PCA components to use
            target_energy: Target energy level (L2 norm) to maintain. If None, computed from training data.
            livnium_force: LivniumForce instance. If None, creates a default one.
        """
        self.pca_model = pca_model
        self.dynamics_model = dynamics_model
        self.decoder = decoder_model
        self.tracker = tracker
        self.n_components = n_components
        
        # Check if dynamics model uses polynomial features
        self.use_poly = 'poly_features' in dynamics_model
        
        # Set target energy (energy injection to prevent collapse)
        self.target_energy = target_energy
        
        # Stochastic driver (noise parameters - initialized by fit_residuals)
        self.residuals = None
        self.noise_mean = None
        self.noise_cov = None
        
        # Livnium force (Phase 6 addition)
        if livnium_force is None:
            self.livnium_force = create_default_livnium(n_components=n_components)
        else:
            self.livnium_force = livnium_force
    
    def fit_residuals(self, trajectory_data: np.ndarray, verbose: bool = True):
        """
        Learn noise from training data residuals.
        
        Computes the difference between true and predicted next states,
        then fits a multivariate Gaussian to model the stochastic component.
        
        Args:
            trajectory_data: (num_steps, n_components) PCA trajectory
            verbose: Print noise statistics
        """
        X_t = trajectory_data[:-1]
        X_tp1_true = trajectory_data[1:]
        
        # Predict using dynamics model
        if self.use_poly:
            X_t_poly = self.dynamics_model['poly_features'].transform(X_t)
            X_tp1_pred = self.dynamics_model['model'].predict(X_t_poly)
        else:
            X_tp1_pred = self.dynamics_model['model'].predict(X_t)
        
        # Compute residuals (noise)
        self.residuals = X_tp1_true - X_tp1_pred
        self.noise_mean = np.mean(self.residuals, axis=0)
        self.noise_cov = np.cov(self.residuals, rowvar=False)
        
        if verbose:
            print(f"  Stochastic Driver Enabled. Noise Scale (Trace): {np.trace(self.noise_cov):.6f}")
    
    def step(self, y_t: np.ndarray) -> np.ndarray:
        """
        Apply one step of dynamics: y_{t+1} = F(y_t) + noise + Livnium(y_t)
        
        **PHASE 6 CHANGE:**
        After applying dynamics and noise, we add the Livnium geometric influence.
        
        Args:
            y_t: Current PCA coordinates of shape (n_components,)
        
        Returns:
            Next PCA coordinates y_{t+1} of shape (n_components,)
        """
        y_t = y_t.reshape(1, -1)  # Ensure 2D
        
        # Existing dynamics
        if self.use_poly:
            y_t_poly = self.dynamics_model['poly_features'].transform(y_t)
            y_tp1 = self.dynamics_model['model'].predict(y_t_poly)
        else:
            y_tp1 = self.dynamics_model['model'].predict(y_t)
        
        y_tp1 = y_tp1.flatten()
        
        # Add stochastic noise (if residuals were fitted)
        if self.noise_mean is not None and self.noise_cov is not None:
            try:
                noise = np.random.multivariate_normal(self.noise_mean, self.noise_cov)
                y_tp1 = y_tp1 + noise
            except np.linalg.LinAlgError:
                # If covariance matrix is singular, use diagonal approximation
                noise = np.random.normal(self.noise_mean, np.sqrt(np.diag(self.noise_cov)))
                y_tp1 = y_tp1 + noise
        
        # PHASE 6: Add Livnium geometric influence
        # This is the minimal Livnium - just a small steering force
        livnium_bias = self.livnium_force.apply_livnium_force(y_t.flatten())
        y_tp1 = y_tp1 + livnium_bias
        
        return y_tp1
    
    def simulate(self, y0: np.ndarray, num_steps: int) -> np.ndarray:
        """
        Simulate trajectory starting from initial condition.
        
        **ENERGY INJECTION FIX:**
        After each step, we re-normalize the state to maintain target energy.
        This prevents the dynamics from collapsing to zero (damped oscillation).
        
        Args:
            y0: Initial PCA coordinates of shape (n_components,)
            num_steps: Number of steps to simulate
        
        Returns:
            Trajectory of shape (num_steps, n_components)
        """
        trajectory = np.zeros((num_steps, self.n_components))
        y = y0.copy()
        
        for i in range(num_steps):
            trajectory[i] = y
            
            # Apply dynamics step (includes Livnium)
            y_next = self.step(y)
            
            # Energy injection: prevent collapse by maintaining target energy
            current_energy = np.linalg.norm(y_next)
            
            if current_energy > 1e-9 and self.target_energy is not None:
                # Re-normalize to target energy level
                y_next = y_next * (self.target_energy / current_energy)
            
            y = y_next
        
        return trajectory
    
    def reconstruct_center_column(self, trajectory_pca: np.ndarray) -> np.ndarray:
        """
        Reconstruct center column bits from PCA trajectory using non-linear decoder.
        
        Args:
            trajectory_pca: (num_steps, n_components) PCA coordinates
        
        Returns:
            Center column bits of shape (num_steps,) - binary values (0 or 1)
        """
        # Use decoder to predict binary bits
        probabilities = self.decoder.predict_proba(trajectory_pca)[:, 1]  # Probability of class 1
        center_bits = self.decoder.predict(trajectory_pca)  # Binary prediction
        
        return center_bits.astype(int)
    
    def compute_statistics(self, trajectory_pca: np.ndarray, 
                          center_column: np.ndarray) -> dict:
        """
        Compute statistics of the shadow trajectory.
        
        Args:
            trajectory_pca: (num_steps, n_components) PCA coordinates
            center_column: (num_steps,) center column bits (0 or 1)
        
        Returns:
            Dict with statistics
        """
        stats = {
            'trajectory_mean': trajectory_pca.mean(axis=0).tolist(),
            'trajectory_std': trajectory_pca.std(axis=0).tolist(),
            'center_mean': float(center_column.mean()),
            'center_std': float(center_column.std()),
            'center_min': int(center_column.min()),
            'center_max': int(center_column.max()),
            'center_ones_fraction': float((center_column == 1).sum() / len(center_column)),
            'livnium_force_scale': float(self.livnium_force.get_force_scale()),
            'livnium_force_type': self.livnium_force.force_type
        }
        
        return stats


def load_models(results_dir: Path, decoder_dir: Path = None):
    """
    Load PCA model, dynamics model, and decoder.
    
    Args:
        results_dir: Directory containing PCA and dynamics models (Phase 3 results)
        decoder_dir: Directory containing decoder model (Phase 4 results, defaults to results_dir)
    """
    results_dir = Path(results_dir)
    if decoder_dir is None:
        decoder_dir = results_dir
    else:
        decoder_dir = Path(decoder_dir)
    
    # Load PCA model
    with open(results_dir / 'pca_model.pkl', 'rb') as f:
        pca_model = pickle.load(f)
    
    # Load metadata to get number of components
    with open(results_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    n_components = metadata['n_components']
    
    # Try to load best dynamics model
    # Prefer polynomial_degree_3 (most complex), then degree_2, then linear
    dynamics_model_path = None
    for name in ['polynomial_degree_3_dynamics_model.pkl', 'polynomial_degree_2_dynamics_model.pkl',
                 'linear_dynamics_model.pkl']:
        if (results_dir / name).exists():
            dynamics_model_path = results_dir / name
            break
    
    if dynamics_model_path is None:
        raise FileNotFoundError("Dynamics model not found. Run fit_full_dynamics.py first.")
    
    with open(dynamics_model_path, 'rb') as f:
        dynamics_model = pickle.load(f)
    
    # Load decoder model
    decoder_path = decoder_dir / 'center_decoder.pkl'
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder model not found at {decoder_path}. Run fit_center_decoder.py first.")
    
    with open(decoder_path, 'rb') as f:
        decoder_model = pickle.load(f)
    
    return pca_model, dynamics_model, decoder_model, n_components


def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Shadow Rule 30 Phase 6: Simulate dynamics with Livnium geometric influence"
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../PHASE3/results',
        help='Directory containing PCA and dynamics models (default: ../PHASE3/results)'
    )
    
    parser.add_argument(
        '--decoder-dir',
        type=str,
        default='../PHASE4/results',
        help='Directory containing decoder model (default: ../PHASE4/results)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5000,
        help='Number of steps to simulate (default: 5000)'
    )
    
    parser.add_argument(
        '--initial-condition',
        type=str,
        choices=['random', 'mean', 'from_data'],
        default='from_data',
        help='Initial condition strategy (default: from_data)'
    )
    
    parser.add_argument(
        '--livnium-scale',
        type=float,
        default=0.01,
        help='Livnium force scaling factor (default: 0.01 = 1%% influence)'
    )
    
    parser.add_argument(
        '--livnium-type',
        type=str,
        choices=['vector', 'matrix'],
        default='vector',
        help='Livnium force type: vector (constant bias) or matrix (state-dependent) (default: vector)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed output'
    )
    
    args = parser.parse_args()
    
    # Load models
    data_dir = Path(args.data_dir)
    decoder_dir = Path(args.decoder_dir)
    
    pca_model, dynamics_model, decoder_model, n_components = load_models(
        data_dir, decoder_dir
    )
    
    if args.verbose:
        print(f"Loaded models:")
        print(f"  PCA model: {data_dir}")
        print(f"  Decoder model: {decoder_dir}")
        print(f"  PCA components: {n_components}")
        print(f"  Dynamics model: {dynamics_model['type']}")
    
    # Compute target energy from training PCA data
    with open(data_dir / 'data_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    
    train_pca = splits['train']['pca'][:, :n_components]
    target_energy = float(np.mean(np.linalg.norm(train_pca, axis=1)))
    
    if args.verbose:
        print(f"\nEnergy Injection:")
        print(f"  Target energy (mean training norm): {target_energy:.6f}")
    
    # Create Livnium force (Phase 6)
    livnium_force = create_default_livnium(
        n_components=n_components,
        force_scale=args.livnium_scale,
        force_type=args.livnium_type
    )
    
    if args.verbose:
        print(f"\nLivnium Force (Phase 6):")
        print(f"  Force scale: {args.livnium_scale}")
        print(f"  Force type: {args.livnium_type}")
        print(f"  This adds a small geometric bias to guide the trajectory")
    
    # Load or create tracker for reconstruction
    tracker = ChaosTracker15D(verbose=False)
    
    # Create Shadow Rule 30 with Livnium
    shadow = ShadowRule30Phase6(
        pca_model, dynamics_model, decoder_model, tracker, 
        n_components=n_components, 
        target_energy=target_energy,
        livnium_force=livnium_force
    )
    
    # Fit residuals for stochastic driver
    if args.verbose:
        print(f"\nFitting stochastic driver from training data...")
    trajectory_pca_full = np.load(data_dir / 'trajectory_pca.npy')
    shadow.fit_residuals(trajectory_pca_full[:, :n_components], verbose=args.verbose)
    
    # Determine initial condition
    if args.initial_condition == 'from_data':
        with open(data_dir / 'data_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        y0 = splits['train']['pca'][0, :n_components]
    elif args.initial_condition == 'mean':
        with open(data_dir / 'data_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        y0 = splits['train']['pca'][:, :n_components].mean(axis=0)
    else:  # random
        with open(data_dir / 'data_splits.pkl', 'rb') as f:
            splits = pickle.load(f)
        train_pca = splits['train']['pca'][:, :n_components]
        mean = train_pca.mean(axis=0)
        std = train_pca.std(axis=0)
        y0 = np.random.normal(mean, std)
    
    if args.verbose:
        print(f"\nInitial condition: {y0}")
    
    # Simulate
    if args.verbose:
        print(f"\nSimulating {args.num_steps} steps with Livnium...")
    
    trajectory_pca = shadow.simulate(y0, args.num_steps)
    
    # Reconstruct center column using decoder
    center_column = shadow.reconstruct_center_column(trajectory_pca)
    
    # Compute statistics
    stats = shadow.compute_statistics(trajectory_pca, center_column)
    
    if args.verbose:
        print(f"\nShadow Rule 30 (Phase 6) Statistics:")
        print(f"  Trajectory shape: {trajectory_pca.shape}")
        print(f"  Center column (bits): mean={stats['center_mean']:.4f}, std={stats['center_std']:.4f}")
        print(f"  Center column range: [{stats['center_min']}, {stats['center_max']}]")
        print(f"  Fraction of ones: {stats['center_ones_fraction']:.4f}")
        print(f"  Livnium force scale: {stats['livnium_force_scale']}")
        print(f"  Livnium force type: {stats['livnium_force_type']}")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'shadow_trajectory_pca.npy', trajectory_pca)
    np.save(output_dir / 'shadow_center_column.npy', center_column)
    
    with open(output_dir / 'shadow_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    if args.verbose:
        print(f"\nResults saved to: {output_dir}")
        print(f"  - shadow_trajectory_pca.npy")
        print(f"  - shadow_center_column.npy")
        print(f"  - shadow_statistics.json")


if __name__ == "__main__":
    main()

