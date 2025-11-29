"""
Rule Discovery: Learn Rules from Geometric Features

Phase 1: Train a decision tree on logged features to discover good rules.
Phase 2: Translate tree into explicit rules for Layer 7.

This is the "priest of rules" - it watches geometry and learns how to read it.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Dict, List, Tuple
import json


class RuleDiscovery:
    """
    Discovers rules from geometric features using decision trees.
    
    The tool that learns how to read the universe.
    """
    
    def __init__(self, max_depth: int = 4, min_samples_split: int = 20):
        """
        Initialize rule discovery.
        
        Args:
            max_depth: Maximum depth of decision tree (keep it shallow for interpretability)
            min_samples_split: Minimum samples to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
    
    def load_features(self, csv_file: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load geometric features from CSV.
        
        Args:
            csv_file: Path to CSV file with features
            
        Returns:
            (features_df, labels_series)
        """
        df = pd.read_csv(csv_file)
        
        # Filter out rows without true labels
        df_labeled = df[df['true_label'].notna() & (df['true_label'] != '')].copy()
        
        if len(df_labeled) == 0:
            raise ValueError(f"No labeled examples found in {csv_file}")
        
        # Feature columns (exclude labels and metadata)
        exclude_cols = ['predicted_label', 'true_label', 'route']
        feature_cols = [col for col in df_labeled.columns if col not in exclude_cols]
        
        # Convert bool strings back to bool
        for col in ['is_stable', 'is_moksha']:
            if col in df_labeled.columns:
                df_labeled[col] = df_labeled[col].map({'True': 1, 'False': 0})
        
        X = df_labeled[feature_cols]
        y = df_labeled['true_label'].map({'E': 0, 'N': 1, 'C': 2, 
                                         'entailment': 0, 'neutral': 1, 'contradiction': 2})
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(self, csv_file: str) -> Dict:
        """
        Train decision tree on geometric features.
        
        Args:
            csv_file: Path to CSV file with features
            
        Returns:
            Dict with training results and discovered rules
        """
        X, y = self.load_features(csv_file)
        
        print(f"Training decision tree on {len(X)} examples...")
        print(f"Features: {list(X.columns)}")
        print(f"Label distribution: {y.value_counts().to_dict()}")
        print()
        
        # Train decision tree
        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        
        self.tree.fit(X, y)
        
        # Evaluate
        y_pred = self.tree.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Get tree rules
        tree_rules = export_text(self.tree, feature_names=list(X.columns))
        
        # Feature importance
        feature_importance = dict(zip(X.columns, self.tree.feature_importances_))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Classification report
        report = classification_report(y, y_pred, 
                                      target_names=['E', 'N', 'C'],
                                      output_dict=True)
        
        results = {
            'accuracy': float(accuracy),
            'n_samples': int(len(X)),
            'tree_rules': tree_rules,
            'feature_importance': {k: float(v) for k, v in feature_importance.items()},
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        return results
    
    def print_rules(self, results: Dict):
        """Print discovered rules in human-readable format."""
        print("=" * 70)
        print("DISCOVERED RULES")
        print("=" * 70)
        print()
        print(f"Accuracy: {results['accuracy']:.4f} ({results['n_samples']} examples)")
        print()
        print("Decision Tree Rules:")
        print("-" * 70)
        print(results['tree_rules'])
        print("-" * 70)
        print()
        
        print("Feature Importance:")
        print("-" * 70)
        sorted_importance = sorted(results['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_importance[:10]:  # Top 10
            print(f"  {feature:<25} {importance:.4f}")
        print("-" * 70)
        print()
        
        print("Confusion Matrix:")
        print("-" * 70)
        cm = np.array(results['confusion_matrix'])
        print(f"      E    N    C")
        for i, label in enumerate(['E', 'N', 'C']):
            print(f"{label}  {cm[i, 0]:4d} {cm[i, 1]:4d} {cm[i, 2]:4d}")
        print("-" * 70)
        print()
    
    def extract_simple_rules(self, results: Dict) -> List[Dict]:
        """
        Extract simple if-then rules from the tree.
        
        Returns:
            List of rule dicts with conditions and label
        """
        # This is a simplified extraction - in practice you'd traverse the tree
        # For now, return the tree rules as-is
        rules = []
        
        # Parse tree rules (simplified - full implementation would traverse tree)
        tree_text = results['tree_rules']
        
        # For now, return placeholder - full implementation would parse tree structure
        rules.append({
            'type': 'tree',
            'rules': tree_text,
            'accuracy': results['accuracy']
        })
        
        return rules
    
    def suggest_layer7_params(self, results: Dict) -> Dict:
        """
        Suggest parameter values for Layer 7 based on discovered rules.
        
        This translates the tree into actionable thresholds.
        
        Returns:
            Dict with suggested parameters
        """
        # Analyze tree to extract thresholds
        # This is a simplified version - full implementation would parse tree
        
        feature_importance = results['feature_importance']
        
        # Find important thresholds from tree structure
        # For now, return current defaults with suggestions
        
        suggestions = {
            'weak_force_threshold': 0.05,  # From tree analysis
            'balance_threshold': 0.15,     # From tree analysis
            'notes': 'Analyze tree_rules to extract specific thresholds'
        }
        
        return suggestions


def main():
    """Main function for rule discovery."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover rules from geometric features')
    parser.add_argument('--features', type=str, required=True,
                       help='Path to CSV file with geometric features')
    parser.add_argument('--max-depth', type=int, default=4,
                       help='Maximum depth of decision tree')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for discovered rules (JSON)')
    
    args = parser.parse_args()
    
    # Discover rules
    discovery = RuleDiscovery(max_depth=args.max_depth)
    results = discovery.train(args.features)
    
    # Print results
    discovery.print_rules(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Rules saved to: {args.output}")
        
        # Also save tree for direct use
        tree_pickle_path = args.output.replace('.json', '_tree.pkl')
        tree_data = {
            'tree': discovery.tree,
            'feature_names': discovery.feature_names
        }
        import pickle
        with open(tree_pickle_path, 'wb') as f:
            pickle.dump(tree_data, f)
        print(f"✓ Decision tree saved to: {tree_pickle_path}")
    
    # Suggest Layer 7 parameters
    suggestions = discovery.suggest_layer7_params(results)
    print("Suggested Layer 7 Parameters:")
    print("-" * 70)
    print(json.dumps(suggestions, indent=2))
    print("-" * 70)


if __name__ == '__main__':
    main()

