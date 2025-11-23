"""
Feature Logger: Collect Geometric Signals for Rule Discovery

The "priest of rules" observing the universe.
Logs geometric features + true labels for rule discovery.
"""

import csv
import os
from typing import Dict, Optional
from pathlib import Path


class FeatureLogger:
    """
    Logs geometric features from the classifier for rule discovery.
    
    This is Phase 1: collect data to discover good rules.
    """
    
    def __init__(self, output_file: str = "geometric_features.csv"):
        """
        Initialize feature logger.
        
        Args:
            output_file: Path to CSV file for logging features
        """
        self.output_file = output_file
        self.fieldnames = None
        self.csv_file = None
        self.writer = None
        self._initialize_file()
    
    def _initialize_file(self):
        """Initialize CSV file with headers."""
        # Field names for geometric features + labels
        self.fieldnames = [
            # Basin info
            'basin_id',
            'basin_conf',
            
            # Core forces
            'cold_attraction',
            'far_attraction',
            'city_pull',
            
            # Normalized basin forces
            'cold_force',
            'far_force',
            'city_force',
            
            # Geometry signals
            'resonance',
            'curvature',
            'max_force',
            'force_ratio',
            'cold_density',
            'distance',
            
            # Scores
            'e_score',
            'c_score',
            'n_score',
            
            # Stability signals
            'is_stable',
            'is_moksha',
            'route',
            
            # Labels (for supervised learning)
            'predicted_label',  # E/N/C from decide()
            'true_label',       # E/N/C from SNLI (if available)
        ]
        
        # Create directory if needed
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists
        file_exists = output_path.exists()
        
        # Open file for writing
        self.csv_file = open(self.output_file, 'a', newline='')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        
        # Write header if new file
        if not file_exists:
            self.writer.writeheader()
    
    def log(self, features: Dict, predicted_label: str, true_label: Optional[str] = None):
        """
        Log geometric features and labels.
        
        Args:
            features: Geometric features from extract_geometric_features()
            predicted_label: E/N/C label from decide()
            true_label: True label from SNLI (if available)
        """
        row = features.copy()
        row['predicted_label'] = predicted_label
        row['true_label'] = true_label if true_label else ''
        
        # Convert bools to strings for CSV
        for key in ['is_stable', 'is_moksha']:
            if key in row:
                row[key] = 'True' if row[key] else 'False'
        
        self.writer.writerow(row)
        self.csv_file.flush()  # Ensure data is written
    
    def close(self):
        """Close the CSV file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

