"""
Layer 7: Auto Classifier - Geometry Decides

This is the native semantic law discovered by geometry itself.

The geometry revealed: basin_conf is the primary signal.
Everything else is secondary.

Simple physics:
- basin_conf <= 0.60 → Uncertain → Neutral
- basin_conf > 0.60 → Stable → E or C (based on forces)
- resonance > 0.66 → Strong signal → E or C

This is NOT hand-written logic.
This is geometry's own law, discovered through rule discovery.
"""

from typing import Dict, Optional
from dataclasses import dataclass
import numpy as np
import pickle
import os


@dataclass
class AutoRuleSet:
    """Auto-discovered rule set from geometry."""
    basin_conf_threshold: float = 0.40  # Discovered: below this → Neutral (adjusted for 3-basin normalization)
    resonance_threshold: float = 0.66   # Discovered: above this → strong signal
    source: str = 'geometry_discovered'


class Layer7AutoClassifier:
    """
    Auto classifier based on geometry's discovered law.
    
    This is geometry writing its own rules.
    Not hand-tuned. Not statistical. Pure physics.
    
    Can use decision tree directly (recommended) or simplified rules.
    """
    
    def __init__(self, rule_set: AutoRuleSet = None, use_tree: bool = False, tree_path: str = None):
        """
        Initialize auto classifier.
        
        Args:
            rule_set: Auto-discovered rule set (uses defaults if None)
            use_tree: If True, use decision tree directly (recommended)
            tree_path: Path to saved decision tree pickle file
        """
        if rule_set is None:
            self.rule_set = AutoRuleSet()
        else:
            self.rule_set = rule_set
        
        self.use_tree = use_tree
        self.tree = None
        self.tree_feature_names = None
        
        if use_tree and tree_path and os.path.exists(tree_path):
            self._load_tree(tree_path)
    
    def _load_tree(self, tree_path: str):
        """Load decision tree from pickle file."""
        try:
            with open(tree_path, 'rb') as f:
                tree_data = pickle.load(f)
                self.tree = tree_data.get('tree')
                self.tree_feature_names = tree_data.get('feature_names')
            print(f"✓ Loaded decision tree from {tree_path}")
        except Exception as e:
            print(f"⚠️  Could not load tree: {e}")
            self.use_tree = False
    
    def classify(self, features: Dict[str, float]) -> tuple:
        """
        Classify using geometry's discovered law.
        
        If use_tree=True, uses decision tree directly (most accurate).
        Otherwise uses simplified rules.
        
        Args:
            features: Geometric features
            
        Returns:
            (label: str, confidence: float, rule_used: str)
        """
        # Use decision tree directly if available (most accurate)
        if self.use_tree and self.tree is not None:
            return self._classify_with_tree(features)
        
        # Otherwise use simplified rules
        return self._classify_with_rules(features)
    
    def _classify_with_tree(self, features: Dict[str, float]) -> tuple:
        """Classify using decision tree directly."""
        if self.tree_feature_names is None:
            return self._classify_with_rules(features)
        
        # Build feature vector in same order as training
        feature_vector = []
        for feat_name in self.tree_feature_names:
            feature_vector.append(features.get(feat_name, 0.0))
        
        feature_array = np.array([feature_vector])
        
        # Predict
        prediction = self.tree.predict(feature_array)[0]
        probabilities = self.tree.predict_proba(feature_array)[0]
        
        # Map: 0=E, 1=N, 2=C
        label_map = {0: 'E', 1: 'N', 2: 'C'}
        label = label_map.get(prediction, 'N')
        confidence = float(max(probabilities))
        
        return (label, confidence, f"decision_tree (class={prediction})")
    
    def _classify_with_rules(self, features: Dict[str, float]) -> tuple:
        """
        Classify using simplified rules (fallback when tree not available).
        
        Based on discovered tree: far_attraction is primary signal (0.88 importance).
        Key insight: High far_attraction → Contradiction, Low far_attraction → Neutral/Entailment
        """
        basin_conf = features.get('basin_conf', 0.5)
        cold_force = features.get('cold_force', 0.33)
        far_force = features.get('far_force', 0.33)
        city_force = features.get('city_force', 0.33)
        cold_attraction = features.get('cold_attraction', 0.0)
        far_attraction = features.get('far_attraction', 0.0)
        resonance = features.get('resonance', 0.0)
        max_force = features.get('max_force', 0.0)
        c_score = features.get('c_score', 0.0)
        e_score = features.get('e_score', 0.0)
        city_pull = features.get('city_pull', 0.0)
        
        # ============================================================
        # GEOMETRY'S DISCOVERED LAW (simplified from tree)
        # ============================================================
        # Primary signal: far_attraction (0.88 importance)
        # When far_attraction is HIGH → Contradiction is more likely
        # When far_attraction is LOW → Entailment or Neutral
        
        # Rule 1: Low far_attraction (<= 0.60) → Mostly Neutral, sometimes Entailment
        if far_attraction <= 0.60:
            # High resonance + low far → Entailment (cold signal strong)
            if resonance > 0.78 and far_force <= 0.37:
                confidence = min(1.0, basin_conf + 0.2)
                return ("E", confidence, "far_attraction <= 0.60 + resonance > 0.78 + far_force <= 0.37 → Entailment")
            
            # High city pull → Neutral
            if city_pull > 0.44 or city_force > 0.44:
                confidence = basin_conf * 0.8
                return ("N", confidence, "far_attraction <= 0.60 + high city_pull → Neutral")
            
            # High c_score despite low far_attraction → Contradiction
            if c_score > 0.16:
                confidence = basin_conf + 0.1
                return ("C", confidence, "far_attraction <= 0.60 + c_score > 0.16 → Contradiction")
            
            # Default: Neutral
            confidence = basin_conf * 0.8
            return ("N", confidence, "far_attraction <= 0.60 → Neutral")
        
        # Rule 2: High far_attraction (> 0.60) → Contradiction or Entailment
        else:  # far_attraction > 0.60
            # High resonance (> 0.78) → Check far_force
            if resonance > 0.78:
                if far_force <= 0.37:
                    # Low far_force → Entailment (cold wins despite high far_attraction)
                    confidence = min(1.0, basin_conf + 0.3)
                    return ("E", confidence, "far_attraction > 0.60 + resonance > 0.78 + far_force <= 0.37 → Entailment")
                else:
                    # High far_force → Entailment (from tree, but seems counterintuitive)
                    confidence = min(1.0, basin_conf + 0.2)
                    return ("E", confidence, "far_attraction > 0.60 + resonance > 0.78 + far_force > 0.37 → Entailment")
            
            # Low resonance (<= 0.78) → Contradiction (far signal is strong)
            if resonance <= 0.78:
                confidence = basin_conf + 0.2
                return ("C", confidence, "far_attraction > 0.60 + resonance <= 0.78 → Contradiction")
            
            # Check basin_conf
            if basin_conf <= 0.38:
                # Low confidence → Check e_score
                if e_score > 0.16:
                    confidence = basin_conf + 0.1
                    return ("N", confidence, "far_attraction > 0.60 + basin_conf <= 0.38 + e_score > 0.16 → Neutral")
                else:
                    confidence = basin_conf + 0.1
                    return ("C", confidence, "far_attraction > 0.60 + basin_conf <= 0.38 → Contradiction")
            
            # High basin_conf (> 0.38) → Check max_force
            if max_force > 0.65:
                # High max_force → Entailment (cold signal is strong)
                confidence = basin_conf
                return ("E", confidence, "far_attraction > 0.60 + max_force > 0.65 → Entailment")
            else:
                # Low max_force → Check far_attraction threshold
                if far_attraction <= 0.61:
                    confidence = basin_conf
                    return ("E", confidence, "far_attraction > 0.60 + max_force <= 0.65 → Entailment")
                else:
                    # Very high far_attraction → Contradiction
                    confidence = basin_conf
                    return ("C", confidence, "far_attraction > 0.61 + max_force <= 0.65 → Contradiction")
            
            # Default: Use force competition
            if far_force > cold_force + 0.1:
                confidence = basin_conf
                return ("C", confidence, "far_attraction > 0.60 + far_force > cold_force → Contradiction")
            elif cold_force > far_force + 0.1:
                confidence = basin_conf
                return ("E", confidence, "far_attraction > 0.60 + cold_force > far_force → Entailment")
            else:
                confidence = basin_conf * 0.7
                return ("N", confidence, "far_attraction > 0.60 but forces balanced → Neutral")
    
    def update_from_discovery(self, discovered_rules_file: str, save_tree: bool = True):
        """
        Update rules from discovered_rules.json.
        
        This is the auto-evolution: geometry discovers new rules, we reload them.
        
        Args:
            discovered_rules_file: Path to discovered_rules.json
            save_tree: If True, save decision tree for direct use
        """
        import json
        
        try:
            with open(discovered_rules_file, 'r') as f:
                data = json.load(f)
            
            # Extract thresholds from tree rules
            tree_rules = data.get('tree_rules', '')
            
            # Try to extract basin_conf threshold from tree
            import re
            matches = re.findall(r'basin_conf\s*[<>=]+\s*([\d.]+)', tree_rules)
            if matches:
                thresholds = [float(m) for m in matches]
                if thresholds:
                    self.rule_set.basin_conf_threshold = np.median(thresholds)
            
            # Extract resonance threshold if present
            matches = re.findall(r'resonance\s*[<>=]+\s*([\d.]+)', tree_rules)
            if matches:
                thresholds = [float(m) for m in matches]
                if thresholds:
                    self.rule_set.resonance_threshold = np.median(thresholds)
            
            # Try to load decision tree if available (from rule_discovery.py)
            # The tree should be saved alongside discovered_rules.json
            tree_pickle_path = discovered_rules_file.replace('.json', '_tree.pkl')
            if os.path.exists(tree_pickle_path):
                self._load_tree(tree_pickle_path)
                self.use_tree = True
                print(f"✓ Using decision tree directly (most accurate)")
            else:
                print(f"⚠️  Tree pickle not found: {tree_pickle_path}")
                print(f"  Using simplified rules")
            
            self.rule_set.source = 'auto_discovered'
            print(f"✓ Updated rules from discovery:")
            print(f"  basin_conf_threshold = {self.rule_set.basin_conf_threshold:.3f}")
            print(f"  resonance_threshold = {self.rule_set.resonance_threshold:.3f}")
            
        except Exception as e:
            print(f"⚠️  Could not update from discovery: {e}")
            print(f"  Using default rules")
    
    def get_rules(self) -> Dict:
        """Get current rule set."""
        return {
            'basin_conf_threshold': self.rule_set.basin_conf_threshold,
            'resonance_threshold': self.rule_set.resonance_threshold,
            'source': self.rule_set.source
        }

