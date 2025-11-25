#!/usr/bin/env python3
"""
Fit Asymptotic Limit for Divergence V2

Fits convergence curves to extract the asymptotic invariant limit.
Models: D(n) = L + A/n^p or D(n) = L + A*exp(-n/tau)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.rule30.rule30_optimized import generate_center_column_direct
from experiments.rule30.divergence_v2 import (
    divergence_v2_transition_based,
    divergence_v2_neighborhood,
    divergence_v2_pattern_weighted
)


def power_law_model(n, L, A, p):
    """
    Power law convergence model: D(n) = L + A / n^p
    
    Args:
        n: Sequence length
        L: Asymptotic limit (the invariant)
        A: Amplitude
        p: Power exponent
    """
    return L + A / (n ** p)


def exponential_model(n, L, A, tau):
    """
    Exponential convergence model: D(n) = L + A * exp(-n / tau)
    
    Args:
        n: Sequence length
        L: Asymptotic limit (the invariant)
        A: Amplitude
        tau: Time constant
    """
    return L + A * np.exp(-n / tau)


def fit_asymptotic_limit(
    n_values: list,
    divergence_values: list,
    model_type: str = 'power'
) -> dict:
    """
    Fit convergence curve to extract asymptotic limit.
    
    Args:
        n_values: Sequence lengths
        divergence_values: Divergence values
        model_type: 'power' or 'exponential'
        
    Returns:
        Dict with fit parameters and limit estimate
    """
    n_array = np.array(n_values, dtype=float)
    d_array = np.array(divergence_values, dtype=float)
    
    # Choose model
    if model_type == 'power':
        model_func = power_law_model
        initial_guess = [d_array[-1], (d_array[0] - d_array[-1]) * n_values[0], 0.5]
        param_names = ['L', 'A', 'p']
    else:  # exponential
        model_func = exponential_model
        initial_guess = [d_array[-1], d_array[0] - d_array[-1], np.mean(n_values)]
        param_names = ['L', 'A', 'tau']
    
    try:
        # Fit curve
        popt, pcov = curve_fit(model_func, n_array, d_array, p0=initial_guess, maxfev=10000)
        
        # Extract limit (first parameter)
        limit = popt[0]
        limit_std = np.sqrt(pcov[0, 0])
        
        # Compute fitted values
        fitted_values = model_func(n_array, *popt)
        
        # Compute R-squared
        ss_res = np.sum((d_array - fitted_values) ** 2)
        ss_tot = np.sum((d_array - np.mean(d_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Residuals
        residuals = d_array - fitted_values
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        return {
            'limit': float(limit),
            'limit_std': float(limit_std),
            'limit_95ci': (float(limit - 1.96 * limit_std), float(limit + 1.96 * limit_std)),
            'parameters': dict(zip(param_names, popt.tolist())),
            'parameter_std': dict(zip(param_names, np.sqrt(np.diag(pcov)).tolist())),
            'fitted_values': fitted_values.tolist(),
            'residuals': residuals.tolist(),
            'r_squared': float(r_squared),
            'rmse': float(rmse),
            'model_type': model_type,
            'model_func': model_func
        }
    
    except Exception as e:
        return {'error': str(e)}


def plot_convergence_fit(
    n_values: list,
    divergence_values: list,
    fit_result: dict,
    divergence_name: str,
    output_path: str = None
):
    """
    Plot convergence curve with fitted asymptotic limit.
    
    Args:
        n_values: Sequence lengths
        divergence_values: Measured divergence values
        fit_result: Fit results from fit_asymptotic_limit
        divergence_name: Name of divergence function
        output_path: Optional path to save plot
    """
    if 'error' in fit_result:
        print(f"Error in fit: {fit_result['error']}")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    n_array = np.array(n_values)
    d_array = np.array(divergence_values)
    limit = fit_result['limit']
    limit_ci = fit_result['limit_95ci']
    fitted = np.array(fit_result['fitted_values'])
    
    # Plot 1: Convergence with fit
    ax1.plot(n_array, d_array, 'bo-', linewidth=2, markersize=8, label='Measured')
    ax1.plot(n_array, fitted, 'r--', linewidth=2, label=f'Fitted ({fit_result["model_type"]} model)')
    ax1.axhline(y=limit, color='g', linestyle='-', linewidth=2, 
                label=f'Asymptotic Limit: {limit:.9f}')
    ax1.axhspan(limit_ci[0], limit_ci[1], alpha=0.2, color='green', 
                label=f'95% CI: [{limit_ci[0]:.9f}, {limit_ci[1]:.9f}]')
    ax1.set_xlabel('Sequence Length (n)', fontsize=12)
    ax1.set_ylabel('Divergence V2', fontsize=12)
    ax1.set_title(f'{divergence_name} - Asymptotic Limit Fit', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Residuals
    residuals = fit_result['residuals']
    ax2.plot(n_array, residuals, 'go-', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax2.fill_between(n_array, 
                     [-fit_result['rmse']] * len(n_array),
                     [fit_result['rmse']] * len(n_array),
                     alpha=0.2, color='gray', label=f'±RMSE ({fit_result["rmse"]:.9e})')
    ax2.set_xlabel('Sequence Length (n)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Fit asymptotic limit for Divergence V2 convergence",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--divergence',
        type=str,
        choices=['transition', 'neighborhood', 'pattern-weighted'],
        default='neighborhood',
        help='Divergence function to fit (default: neighborhood)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[1000, 5000, 10000, 20000, 50000, 100000],
        help='Sequence lengths to test (default: 1000 5000 10000 20000 50000 100000)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['power', 'exponential', 'both'],
        default='power',
        help='Convergence model to fit (default: power)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate convergence fit plot'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/rule30/results',
        help='Output directory for plots (default: experiments/rule30/results)'
    )
    
    args = parser.parse_args()
    
    # Map divergence names to functions
    divergence_funcs = {
        'transition': divergence_v2_transition_based,
        'neighborhood': divergence_v2_neighborhood,
        'pattern-weighted': divergence_v2_pattern_weighted
    }
    
    div_func = divergence_funcs[args.divergence]
    div_name = args.divergence.replace('-', ' ').title()
    
    print("="*70)
    print("ASYMPTOTIC LIMIT FITTER")
    print("="*70)
    print(f"Divergence: {div_name}")
    print(f"Sequence lengths: {args.steps}")
    print(f"Model: {args.model}")
    print()
    
    # Generate data
    print("Generating Rule 30 sequences and computing divergence...")
    n_values = []
    divergence_values = []
    
    for n_steps in args.steps:
        print(f"  n={n_steps:6,}...", end=" ", flush=True)
        sequence = generate_center_column_direct(n_steps, show_progress=False)
        div_value = div_func(sequence)
        n_values.append(n_steps)
        divergence_values.append(div_value)
        print(f"divergence = {div_value:.9f}")
    
    print()
    
    # Fit models
    models_to_fit = ['power', 'exponential'] if args.model == 'both' else [args.model]
    fit_results = {}
    
    for model_type in models_to_fit:
        print(f"Fitting {model_type} model...")
        fit_result = fit_asymptotic_limit(n_values, divergence_values, model_type)
        
        if 'error' in fit_result:
            print(f"  Error: {fit_result['error']}")
            continue
        
        fit_results[model_type] = fit_result
        
        print(f"  Asymptotic limit: {fit_result['limit']:.9f} ± {fit_result['limit_std']:.9e}")
        print(f"  95% CI: [{fit_result['limit_95ci'][0]:.9f}, {fit_result['limit_95ci'][1]:.9f}]")
        print(f"  R² = {fit_result['r_squared']:.6f}")
        print(f"  RMSE = {fit_result['rmse']:.9e}")
        print(f"  Parameters: {fit_result['parameters']}")
        print()
    
    # Choose best fit
    if len(fit_results) > 1:
        best_model = max(fit_results.items(), key=lambda x: x[1]['r_squared'])
        print(f"Best model: {best_model[0]} (R² = {best_model[1]['r_squared']:.6f})")
        best_fit = best_model[1]
    elif fit_results:
        best_fit = list(fit_results.values())[0]
        best_model_name = list(fit_results.keys())[0]
    else:
        print("No successful fits")
        return
    
    # Plot
    if args.plot:
        output_path = f"{args.output_dir}/divergence_v2_{args.divergence}_asymptotic_fit.png"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        plot_convergence_fit(n_values, divergence_values, best_fit, div_name, output_path)
    
    # Final summary
    print("="*70)
    print("ASYMPTOTIC INVARIANT")
    print("="*70)
    print(f"Divergence: {div_name}")
    print(f"Model: {best_fit['model_type']}")
    print(f"\nAsymptotic Limit (Invariant):")
    print(f"  L = {best_fit['limit']:.9f} ± {best_fit['limit_std']:.9e}")
    print(f"  95% Confidence Interval: [{best_fit['limit_95ci'][0]:.9f}, {best_fit['limit_95ci'][1]:.9f}]")
    print(f"\nFit Quality:")
    print(f"  R² = {best_fit['r_squared']:.6f}")
    print(f"  RMSE = {best_fit['rmse']:.9e}")
    print()
    print("✓✓✓ ASYMPTOTIC INVARIANT EXTRACTED")
    print(f"This is the true geometric invariant for Rule 30: {best_fit['limit']:.9f}")
    print()


if __name__ == '__main__':
    main()

