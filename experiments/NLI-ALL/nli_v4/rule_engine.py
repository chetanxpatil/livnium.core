"""
Rule Engine: Clean Symbolic Logic from Geometry

Converts discovered rules into clean if-then logic.
This is the "priest of rules" - it watches geometry and translates it.

Philosophy:
- Geometry produces meaning (wild, unsupervised)
- Rules interpret geometry (clean, symbolic)
- Labels are human artifacts (we translate)
"""

import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class RuleSet:
    """A set of symbolic rules for classification."""
    rules: List[Dict]  # List of rule dicts with conditions and label
    accuracy: float
    source: str  # 'discovered', 'hand_tuned', 'evolved'
    metadata: Dict


class RuleEngine:
    """
    Clean symbolic rule engine that interprets geometry.
    
    This is the translator between geometry (wild) and labels (human artifacts).
    """
    
    def __init__(self, rule_set: Optional[RuleSet] = None):
        """
        Initialize rule engine.
        
        Args:
            rule_set: Optional RuleSet to use, otherwise uses default rules
        """
        if rule_set is None:
            # Default rules (will be replaced by discovered rules)
            self.rule_set = self._create_default_rules()
        else:
            self.rule_set = rule_set
    
    def _create_default_rules(self) -> RuleSet:
        """Create default rule set (current Layer 7 logic)."""
        rules = [
            {
                'condition': 'max_force < 0.05',
                'label': 'N',
                'description': 'Weak forces → Neutral'
            },
            {
                'condition': 'force_ratio < 0.15',
                'label': 'N',
                'description': 'Balanced forces → Neutral'
            },
            {
                'condition': 'cold_attraction > far_attraction',
                'label': 'E',
                'description': 'Cold wins → Entailment'
            },
            {
                'condition': 'else',
                'label': 'C',
                'description': 'Default → Contradiction'
            }
        ]
        
        return RuleSet(
            rules=rules,
            accuracy=0.0,
            source='default',
            metadata={'version': '1.0'}
        )
    
    def classify(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """
        Classify using symbolic rules.
        
        Args:
            features: Geometric features dict
            
        Returns:
            (label, confidence, rule_used)
        """
        # Evaluate rules in order (first match wins)
        for rule in self.rule_set.rules:
            condition = rule['condition']
            
            if condition == 'else':
                # Default rule
                return rule['label'], 0.5, rule['description']
            
            if self._evaluate_condition(condition, features):
                # Rule matched
                confidence = self._compute_confidence(rule, features)
                return rule['label'], confidence, rule['description']
        
        # Fallback (shouldn't happen if rules are complete)
        return 'N', 0.5, 'fallback'
    
    def _evaluate_condition(self, condition: str, features: Dict[str, float]) -> bool:
        """
        Evaluate a condition string against features.
        
        Supports simple comparisons: 'feature > value', 'feature < value', etc.
        """
        # Parse condition
        condition = condition.strip()
        
        # Handle compound conditions (simple AND/OR)
        if ' and ' in condition:
            parts = condition.split(' and ')
            return all(self._evaluate_condition(p.strip(), features) for p in parts)
        
        if ' or ' in condition:
            parts = condition.split(' or ')
            return any(self._evaluate_condition(p.strip(), features) for p in parts)
        
        # Helper to evaluate a value expression (feature or number or arithmetic)
        def eval_value(expr: str) -> float:
            expr = expr.strip()
            # Check if it's a feature name
            if expr in features:
                return features[expr]
            # Check if it's arithmetic (feature + number or feature - number)
            if ' + ' in expr:
                parts = expr.split(' + ')
                return eval_value(parts[0]) + float(parts[1])
            if ' - ' in expr:
                parts = expr.split(' - ')
                return eval_value(parts[0]) - float(parts[1])
            # Otherwise it's a number
            return float(expr)
        
        # Simple comparison (check >= and <= first to avoid splitting on > or <)
        if '>=' in condition:
            parts = condition.split('>=')
            if len(parts) == 2:
                left = eval_value(parts[0].strip())
                right = eval_value(parts[1].strip())
                return left >= right
        
        if '<=' in condition:
            parts = condition.split('<=')
            if len(parts) == 2:
                left = eval_value(parts[0].strip())
                right = eval_value(parts[1].strip())
                return left <= right
        
        if '>' in condition:
            parts = condition.split('>')
            if len(parts) == 2:
                left = eval_value(parts[0].strip())
                right = eval_value(parts[1].strip())
                return left > right
        
        if '<' in condition:
            parts = condition.split('<')
            if len(parts) == 2:
                left = eval_value(parts[0].strip())
                right = eval_value(parts[1].strip())
                return left < right
        
        if '==' in condition:
            parts = condition.split('==')
            if len(parts) == 2:
                feature = parts[0].strip()
                value = float(parts[1].strip())
                return abs(features.get(feature, 0.0) - value) < 1e-6
        
        # Unknown condition format
        return False
    
    def _compute_confidence(self, rule: Dict, features: Dict[str, float]) -> float:
        """
        Compute confidence for a matched rule.
        
        Uses basin_conf as base, adjusts based on rule type.
        """
        basin_conf = features.get('basin_conf', 0.5)
        
        # Boost confidence if multiple signals align
        if rule['label'] == 'E':
            cold_force = features.get('cold_force', 0.0)
            resonance = features.get('resonance', 0.0)
            # Higher confidence if cold is strong and resonance is positive
            boost = (cold_force * 0.3) + (max(0, resonance) * 0.2)
            return min(1.0, basin_conf + boost)
        
        elif rule['label'] == 'C':
            far_force = features.get('far_force', 0.0)
            resonance = features.get('resonance', 0.0)
            # Higher confidence if far is strong and resonance is negative
            boost = (far_force * 0.3) + (max(0, -resonance) * 0.2)
            return min(1.0, basin_conf + boost)
        
        else:  # N
            city_force = features.get('city_force', 0.0)
            force_ratio = features.get('force_ratio', 1.0)
            # Higher confidence if city is strong or forces are balanced
            boost = (city_force * 0.3) + ((1.0 - min(force_ratio, 1.0)) * 0.2)
            return min(1.0, basin_conf + boost)
    
    @staticmethod
    def from_discovered_rules(json_file: str) -> 'RuleEngine':
        """
        Load rules from discovered_rules.json (decision tree output).
        
        This extracts the key patterns from the tree and converts to clean rules.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract key thresholds from feature importance
        feature_importance = data.get('feature_importance', {})
        accuracy = data.get('accuracy', 0.0)
        
        # Parse tree rules to extract thresholds
        # This is simplified - full implementation would parse tree structure
        tree_rules = data.get('tree_rules', '')
        
        # Extract patterns from tree (simplified extraction)
        # In practice, you'd parse the tree structure properly
        rules = []
        
        # Pattern 1: High basin_conf + cold → E
        rules.append({
            'condition': 'basin_conf > 0.70 and cold_attraction > 0.3',
            'label': 'E',
            'description': 'High confidence + cold attraction → Entailment'
        })
        
        # Pattern 2: High basin_conf + far → C
        rules.append({
            'condition': 'basin_conf > 0.70 and far_force > 0.4',
            'label': 'C',
            'description': 'High confidence + far force → Contradiction'
        })
        
        # Pattern 3: Low basin_conf → N or C
        rules.append({
            'condition': 'basin_conf < 0.30',
            'label': 'N',
            'description': 'Low confidence → Neutral'
        })
        
        # Pattern 4: Medium basin_conf + balanced → N
        rules.append({
            'condition': 'basin_conf >= 0.30 and basin_conf <= 0.70 and force_ratio < 0.20',
            'label': 'N',
            'description': 'Medium confidence + balanced forces → Neutral'
        })
        
        # Pattern 5: High resonance + cold → E
        rules.append({
            'condition': 'resonance > 0.5 and cold_force > far_force',
            'label': 'E',
            'description': 'High resonance + cold wins → Entailment'
        })
        
        # Pattern 6: Negative resonance + far → C
        rules.append({
            'condition': 'resonance < -0.3 and far_force > cold_force',
            'label': 'C',
            'description': 'Negative resonance + far wins → Contradiction'
        })
        
        # Default: use force competition
        rules.append({
            'condition': 'cold_force > far_force',
            'label': 'E',
            'description': 'Cold > Far → Entailment'
        })
        
        rules.append({
            'condition': 'else',
            'label': 'C',
            'description': 'Default → Contradiction'
        })
        
        rule_set = RuleSet(
            rules=rules,
            accuracy=accuracy,
            source='discovered',
            metadata={
                'feature_importance': feature_importance,
                'tree_rules': tree_rules[:500]  # Truncate for storage
            }
        )
        
        return RuleEngine(rule_set=rule_set)
    
    @staticmethod
    def from_hand_tuned_rules() -> 'RuleEngine':
        """
        Create hand-tuned rules based on geometry patterns.
        
        These are clean, interpretable rules that align geometry with labels.
        """
        rules = [
            # Rule 1: Very high confidence → trust the basin
            {
                'condition': 'basin_conf > 0.80 and cold_force > 0.5',
                'label': 'E',
                'description': 'Very high confidence + cold → Entailment'
            },
            {
                'condition': 'basin_conf > 0.80 and far_force > 0.5',
                'label': 'C',
                'description': 'Very high confidence + far → Contradiction'
            },
            
            # Rule 2: High confidence + clear signal
            {
                'condition': 'basin_conf > 0.65 and cold_attraction > 0.4 and resonance > 0.2',
                'label': 'E',
                'description': 'High confidence + cold signal → Entailment'
            },
            {
                'condition': 'basin_conf > 0.65 and far_attraction > 0.3 and resonance < -0.2',
                'label': 'C',
                'description': 'High confidence + far signal → Contradiction'
            },
            
            # Rule 3: Balanced forces → Neutral
            {
                'condition': 'force_ratio < 0.15 and max_force > 0.05',
                'label': 'N',
                'description': 'Balanced forces → Neutral'
            },
            
            # Rule 4: Weak forces → Neutral
            {
                'condition': 'max_force < 0.05',
                'label': 'N',
                'description': 'Weak forces → Neutral'
            },
            
            # Rule 5: City pull dominates → Neutral
            {
                'condition': 'city_force > 0.6',
                'label': 'N',
                'description': 'City force dominates → Neutral'
            },
            
            # Rule 6: Medium confidence + geometry hints
            {
                'condition': 'basin_conf >= 0.40 and basin_conf <= 0.65 and cold_force > far_force + 0.15',
                'label': 'E',
                'description': 'Medium confidence + cold advantage → Entailment'
            },
            {
                'condition': 'basin_conf >= 0.40 and basin_conf <= 0.65 and far_force > cold_force + 0.15',
                'label': 'C',
                'description': 'Medium confidence + far advantage → Contradiction'
            },
            
            # Rule 7: Low confidence → Neutral (uncertain)
            {
                'condition': 'basin_conf < 0.40',
                'label': 'N',
                'description': 'Low confidence → Neutral (uncertain)'
            },
            
            # Rule 8: Force competition (fallback)
            {
                'condition': 'cold_force > far_force',
                'label': 'E',
                'description': 'Cold > Far → Entailment'
            },
            
            # Default
            {
                'condition': 'else',
                'label': 'C',
                'description': 'Default → Contradiction'
            }
        ]
        
        rule_set = RuleSet(
            rules=rules,
            accuracy=0.0,  # Will be evaluated
            source='hand_tuned',
            metadata={'version': '2.0', 'description': 'Clean geometry-aligned rules'}
        )
        
        return RuleEngine(rule_set=rule_set)
    
    def save_rules(self, output_file: str):
        """Save rules to JSON file."""
        data = {
            'rules': self.rule_set.rules,
            'accuracy': self.rule_set.accuracy,
            'source': self.rule_set.source,
            'metadata': self.rule_set.metadata
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_rules(self, input_file: str):
        """Load rules from JSON file."""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        self.rule_set = RuleSet(
            rules=data['rules'],
            accuracy=data.get('accuracy', 0.0),
            source=data.get('source', 'unknown'),
            metadata=data.get('metadata', {})
        )
    
    def evaluate(self, features_list: List[Dict], true_labels: List[str]) -> Dict:
        """
        Evaluate rule set on labeled examples.
        
        Returns accuracy and confusion matrix.
        """
        predictions = []
        for features in features_list:
            label, _, _ = self.classify(features)
            predictions.append(label)
        
        # Compute accuracy
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        accuracy = correct / len(true_labels) if true_labels else 0.0
        
        # Confusion matrix
        labels = ['E', 'N', 'C']
        cm = {}
        for true_label in labels:
            cm[true_label] = {}
            for pred_label in labels:
                count = sum(1 for p, t in zip(predictions, true_labels) 
                           if t == true_label and p == pred_label)
                cm[true_label][pred_label] = count
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'predictions': predictions
        }
    
    def print_rules(self):
        """Print rules in human-readable format."""
        print("=" * 70)
        print("RULE ENGINE")
        print("=" * 70)
        print(f"Source: {self.rule_set.source}")
        print(f"Accuracy: {self.rule_set.accuracy:.4f}")
        print()
        print("Rules (evaluated in order):")
        print("-" * 70)
        for i, rule in enumerate(self.rule_set.rules, 1):
            print(f"{i}. IF {rule['condition']}")
            print(f"   THEN {rule['label']}")
            print(f"   ({rule['description']})")
            print()
        print("-" * 70)


def main():
    """Test rule engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Rule Engine')
    parser.add_argument('--load', type=str, default=None,
                       help='Load rules from JSON file')
    parser.add_argument('--discovered', type=str, default=None,
                       help='Load from discovered_rules.json')
    parser.add_argument('--hand-tuned', action='store_true',
                       help='Use hand-tuned rules')
    parser.add_argument('--save', type=str, default=None,
                       help='Save rules to JSON file')
    
    args = parser.parse_args()
    
    # Create rule engine
    if args.discovered:
        engine = RuleEngine.from_discovered_rules(args.discovered)
    elif args.load:
        engine = RuleEngine()
        engine.load_rules(args.load)
    elif args.hand_tuned:
        engine = RuleEngine.from_hand_tuned_rules()
    else:
        engine = RuleEngine.from_hand_tuned_rules()  # Default to hand-tuned
    
    # Print rules
    engine.print_rules()
    
    # Save if requested
    if args.save:
        engine.save_rules(args.save)
        print(f"\n✓ Rules saved to: {args.save}")


if __name__ == '__main__':
    main()

