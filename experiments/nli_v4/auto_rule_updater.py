"""
Auto Rule Updater: Self-Evolving Physical Law

This implements the automated loop:
Geometry → Features → Rule Discovery → Classifier → Next Epoch → New Features → New Rules

The system evolves its own rules based on geometry's discoveries.
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Optional


class AutoRuleUpdater:
    """
    Automatically updates Layer 7 rules based on geometry discoveries.
    
    This is the self-evolving loop:
    1. Collect features during training
    2. Discover rules from features
    3. Reload rules into Layer 7
    4. Continue training with new rules
    """
    
    def __init__(self, 
                 features_file: str = "experiments/nli_v4/features.csv",
                 discovered_rules_file: str = "experiments/nli_v4/discovered_rules.json",
                 update_interval: int = 1000):
        """
        Initialize auto rule updater.
        
        Args:
            features_file: Path to CSV file with geometric features
            discovered_rules_file: Path to save discovered rules JSON
            update_interval: Update rules every N training steps
        """
        self.features_file = features_file
        self.discovered_rules_file = discovered_rules_file
        self.update_interval = update_interval
        self.last_update_step = 0
    
    def should_update(self, current_step: int) -> bool:
        """Check if rules should be updated."""
        return (current_step - self.last_update_step) >= self.update_interval
    
    def discover_rules(self, max_depth: int = 4) -> bool:
        """
        Run rule discovery on collected features.
        
        Returns:
            True if rules were discovered successfully
        """
        if not os.path.exists(self.features_file):
            print(f"⚠️  Features file not found: {self.features_file}")
            return False
        
        # Count lines in features file
        with open(self.features_file, 'r') as f:
            num_lines = sum(1 for line in f) - 1  # Subtract header
        
        if num_lines < 100:
            print(f"⚠️  Not enough features ({num_lines} examples). Need at least 100.")
            return False
        
        print(f"🔍 Discovering rules from {num_lines} examples...")
        
        try:
            # Run rule discovery
            cmd = [
                "python3",
                "experiments/nli_v4/rule_discovery.py",
                "--features", self.features_file,
                "--max-depth", str(max_depth),
                "--output", self.discovered_rules_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            
            if result.returncode == 0:
                print(f"✓ Rules discovered and saved to: {self.discovered_rules_file}")
                return True
            else:
                print(f"⚠️  Rule discovery failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"⚠️  Error during rule discovery: {e}")
            return False
    
    def update_layer7(self, layer7_instance) -> bool:
        """
        Update Layer 7 with discovered rules.
        
        Args:
            layer7_instance: Layer7Decision instance to update
            
        Returns:
            True if update was successful
        """
        if not os.path.exists(self.discovered_rules_file):
            print(f"⚠️  Discovered rules file not found: {self.discovered_rules_file}")
            return False
        
        try:
            # Update auto classifier if it exists
            if hasattr(layer7_instance, 'auto_classifier') and layer7_instance.auto_classifier is not None:
                layer7_instance.auto_classifier.update_from_discovery(self.discovered_rules_file)
                return True
            else:
                print("⚠️  Layer 7 auto classifier not available")
                return False
                
        except Exception as e:
            print(f"⚠️  Error updating Layer 7: {e}")
            return False
    
    def update_loop(self, current_step: int, layer7_instance) -> bool:
        """
        Complete update loop: discover rules and update Layer 7.
        
        Args:
            current_step: Current training step
            layer7_instance: Layer7Decision instance to update
            
        Returns:
            True if update was successful
        """
        if not self.should_update(current_step):
            return False
        
        print(f"\n{'='*70}")
        print(f"AUTO RULE UPDATE (Step {current_step})")
        print(f"{'='*70}")
        
        # Step 1: Discover rules
        if not self.discover_rules():
            return False
        
        # Step 2: Update Layer 7
        if not self.update_layer7(layer7_instance):
            return False
        
        self.last_update_step = current_step
        
        print(f"{'='*70}\n")
        
        return True


def main():
    """Test auto rule updater."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto Rule Updater')
    parser.add_argument('--features', type=str, default='experiments/nli_v4/features.csv',
                       help='Path to features CSV file')
    parser.add_argument('--discover', action='store_true',
                       help='Run rule discovery')
    parser.add_argument('--max-depth', type=int, default=4,
                       help='Max depth for decision tree')
    
    args = parser.parse_args()
    
    updater = AutoRuleUpdater(features_file=args.features)
    
    if args.discover:
        updater.discover_rules(max_depth=args.max_depth)


if __name__ == '__main__':
    main()

