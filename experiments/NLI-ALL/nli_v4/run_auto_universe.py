"""
Autonomous Universe Driver

Runs the semantic universe in continuous cycles.
The universe watches itself and adjusts its own physics.

Usage:
    python3 run_auto_universe.py

This will:
1. Run unsupervised training cycles
2. Generate numerical reports
3. Read cluster summaries
4. Auto-adjust physics parameters
5. Write overrides for next cycle
6. Keep the brain evolving (no reset after cycle 0)
"""

import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "nli" / "data"
CLUSTER_DIR = BASE_DIR / "clusters"
REPORT_PATH = BASE_DIR / "numerical_report.json"  # generate_report.py creates this
OVERRIDES_PATH = BASE_DIR / "auto_physics_overrides.json"


def run_unsupervised_cycle(cycle_id: int, train_samples: int = 10000):
    """
    Run one unsupervised training cycle.
    
    Args:
        cycle_id: Cycle number (0 = first cycle, uses --clean)
        train_samples: Number of training examples
    """
    print(f"\n{'=' * 70}")
    print(f"CYCLE {cycle_id}: Running Unsupervised Training")
    print(f"{'=' * 70}\n")
    
    cmd = [
        "python3",
        str(BASE_DIR / "train_v4.py"),
        "--unsupervised",
        "--train", str(train_samples),
        "--data-dir", str(DATA_DIR),
        "--cluster-output", str(CLUSTER_DIR),
    ]
    
    # Only clean on first cycle - after that, let brain evolve
    if cycle_id == 0:
        cmd.append("--clean")
        print("  → Starting fresh (--clean)")
    else:
        print("  → Continuing evolution (no --clean)")
    
    print(f"  → Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        print(f"  ⚠️  Cycle {cycle_id} failed with exit code {result.returncode}")
        return False
    
    print(f"\n  ✓ Cycle {cycle_id} complete\n")
    return True


def load_cluster_summary() -> Optional[Dict]:
    """Load cluster summary from last cycle."""
    summary_path = CLUSTER_DIR / "cluster_summary.json"
    if not summary_path.exists():
        print(f"  ⚠️  No cluster summary found at {summary_path}")
        return None
    
    try:
        with summary_path.open() as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠️  Error loading cluster summary: {e}")
        return None


def load_report() -> Optional[Dict]:
    """Load numerical report from last cycle."""
    if REPORT_PATH.exists():
        try:
            with REPORT_PATH.open() as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️  Error loading report: {e}")
            return None
    return None


def auto_adjust_physics(cluster_summary: Optional[Dict], 
                       current_overrides: Dict) -> Dict:
    """
    Auto-adjust physics parameters based on universe state.
    
    This is the 'brain stem' - tweaks auto-physics parameters
    based on what the universe is doing.
    
    Args:
        cluster_summary: Cluster statistics from last cycle
        current_overrides: Current physics overrides
        
    Returns:
        Dict of parameter updates
    """
    if cluster_summary is None:
        print("  ⚠️  No cluster summary - using defaults")
        return {}
    
    stats = cluster_summary.get("statistics", {})
    if not stats:
        return {}
    
    # Get basin counts
    cold = stats.get("basin_0_cold", {}).get("count", 0)
    far = stats.get("basin_1_far", {}).get("count", 0)
    city = stats.get("basin_2_city", {}).get("count", 0)
    total = cluster_summary.get("total_entries", 1)
    
    if total == 0:
        return {}
    
    cold_ratio = cold / total
    far_ratio = far / total
    city_ratio = city / total
    
    print(f"\n  📊 Universe State:")
    print(f"     Cold: {cold} ({cold_ratio*100:.1f}%)")
    print(f"     Far:  {far} ({far_ratio*100:.1f}%)")
    print(f"     City: {city} ({city_ratio*100:.1f}%)")
    
    # Start from current overrides or defaults
    entropy_scale = current_overrides.get("entropy_scale", 0.02)
    repulsion_strength = current_overrides.get("repulsion_strength", 0.3)
    turbulence_scale = current_overrides.get("turbulence_scale", 0.2)
    
    updates = {}
    
    # Rule 1: If city is too dominant, shake things more
    if city_ratio > 0.8:
        new_entropy = min(entropy_scale + 0.01, 0.05)
        new_repulsion = min(repulsion_strength + 0.1, 1.0)
        new_turbulence = min(turbulence_scale + 0.05, 0.5)
        
        if new_entropy != entropy_scale:
            updates["entropy_scale"] = new_entropy
            print(f"     🔥 City dominates ({city_ratio*100:.1f}%) → Increasing entropy: {new_entropy:.4f}")
        
        if new_repulsion != repulsion_strength:
            updates["repulsion_strength"] = new_repulsion
            print(f"     🔥 City dominates → Increasing repulsion: {new_repulsion:.4f}")
        
        if new_turbulence != turbulence_scale:
            updates["turbulence_scale"] = new_turbulence
            print(f"     🔥 City dominates → Increasing turbulence: {new_turbulence:.4f}")
    
    # Rule 2: If cold vs far very imbalanced, nudge repulsion/entropy
    imbalance = abs(cold_ratio - far_ratio)
    if imbalance > 0.15:  # More than 15% difference
        new_entropy = min(entropy_scale + 0.005, 0.05)
        if new_entropy != entropy_scale:
            updates.setdefault("entropy_scale", new_entropy)
            print(f"     ⚖️  Cold/Far imbalanced ({imbalance*100:.1f}%) → Increasing entropy: {new_entropy:.4f}")
    
    # Rule 3: If cold and far are balanced but city is small, reduce turbulence
    if city_ratio < 0.3 and imbalance < 0.1:
        new_turbulence = max(turbulence_scale - 0.02, 0.05)
        if new_turbulence != turbulence_scale:
            updates["turbulence_scale"] = new_turbulence
            print(f"     ✅ Balanced universe → Reducing turbulence: {new_turbulence:.4f}")
    
    if not updates:
        print(f"     ✓ Universe stable - no adjustments needed")
    
    return updates


def write_auto_physics_overrides(updates: Dict):
    """
    Write physics overrides to file.
    
    AutoPhysicsEngine will read this on next cycle.
    """
    if not updates:
        return
    
    # Load existing overrides and merge
    existing = {}
    if OVERRIDES_PATH.exists():
        try:
            with OVERRIDES_PATH.open() as f:
                existing = json.load(f)
        except Exception:
            pass
    
    # Merge updates
    existing.update(updates)
    
    # Write back
    with OVERRIDES_PATH.open("w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"\n  💾 Updated physics overrides: {existing}")


def load_current_overrides() -> Dict:
    """Load current physics overrides."""
    if OVERRIDES_PATH.exists():
        try:
            with OVERRIDES_PATH.open() as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def generate_report():
    """Generate numerical report for current state."""
    print(f"\n  📊 Generating numerical report...")
    cmd = ["python3", str(BASE_DIR / "generate_report.py")]
    result = subprocess.run(cmd, check=False, capture_output=True)
    
    if result.returncode == 0:
        print(f"  ✓ Report generated")
    else:
        print(f"  ⚠️  Report generation failed (non-critical)")


def main():
    """Main autonomous universe loop."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run autonomous semantic universe')
    parser.add_argument('--cycles', type=int, default=20,
                        help='Number of cycles to run (default: 20)')
    parser.add_argument('--train-samples', type=int, default=10000,
                        help='Training samples per cycle (default: 10000)')
    parser.add_argument('--skip-report', action='store_true',
                        help='Skip report generation (faster)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AUTONOMOUS SEMANTIC UNIVERSE")
    print("=" * 70)
    print(f"Cycles: {args.cycles}")
    print(f"Training samples per cycle: {args.train_samples}")
    print(f"Cluster output: {CLUSTER_DIR}")
    print(f"Overrides file: {OVERRIDES_PATH}")
    print("=" * 70)
    
    # Ensure directories exist
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load current overrides
    current_overrides = load_current_overrides()
    if current_overrides:
        print(f"\n📋 Starting with physics overrides: {current_overrides}")
    
    for cycle in range(args.cycles):
        print(f"\n{'=' * 70}")
        print(f"CYCLE {cycle + 1} / {args.cycles}")
        print(f"{'=' * 70}")
        
        # Run training cycle
        success = run_unsupervised_cycle(cycle, args.train_samples)
        
        if not success:
            print(f"  ⚠️  Cycle {cycle} failed - continuing anyway...")
            time.sleep(2)
            continue
        
        # Generate report (optional)
        if not args.skip_report:
            generate_report()
        
        # Load results
        cluster_summary = load_cluster_summary()
        
        # Auto-adjust physics
        updates = auto_adjust_physics(cluster_summary, current_overrides)
        
        # Write overrides
        if updates:
            write_auto_physics_overrides(updates)
            current_overrides.update(updates)
        
        # Small sleep for readability
        if cycle < args.cycles - 1:  # Don't sleep after last cycle
            time.sleep(1)
    
    print(f"\n{'=' * 70}")
    print("UNIVERSE EVOLUTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nFinal physics overrides: {current_overrides}")
    print(f"Cluster summary: {CLUSTER_DIR / 'cluster_summary.json'}")
    print(f"\nThe universe has evolved through {args.cycles} cycles.")
    print("Check the cluster files to see what meaning emerged.\n")


if __name__ == "__main__":
    main()

