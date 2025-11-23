"""
Cluster Tracker: Geometry-Discovered Meaning Buckets

Tracks which basin each sentence falls into.
Over time, clusters emerge naturally from geometry.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import json
from pathlib import Path


class ClusterTracker:
    """
    Tracks geometry-discovered clusters (basins).
    
    Meaning emerges as sentences fall into energy wells.
    No labels required - just physics.
    """
    
    def __init__(self):
        """Initialize cluster tracker."""
        # Basin 0 = Cold (entailment-like)
        # Basin 1 = Far (contradiction-like)
        # Basin 2 = City (neutral-like)
        self.clusters: Dict[int, List[Dict]] = {
            0: [],  # Cold basin
            1: [],  # Far basin
            2: []   # City basin
        }
        
        # Track cluster statistics
        self.cluster_stats = {
            0: {'count': 0, 'avg_confidence': 0.0, 'total_confidence': 0.0},
            1: {'count': 0, 'avg_confidence': 0.0, 'total_confidence': 0.0},
            2: {'count': 0, 'avg_confidence': 0.0, 'total_confidence': 0.0}
        }
    
    def add(self, basin_index: int, premise: str, hypothesis: str, 
            confidence: float, layer_states: Dict, predicted_label: str = None):
        """
        Add a sentence pair to a cluster.
        
        Args:
            basin_index: Which basin (0, 1, or 2)
            premise: Premise sentence
            hypothesis: Hypothesis sentence
            confidence: Confidence in assignment
            layer_states: Physics state (for analysis)
            predicted_label: Predicted E/N/C label from force competition
        """
        entry = {
            'premise': premise,
            'hypothesis': hypothesis,
            'confidence': float(confidence),
            'resonance': layer_states.get('resonance', 0.0),
            'cold_attraction': layer_states.get('cold_attraction', 0.0),
            'far_attraction': layer_states.get('far_attraction', 0.0),
            'basin_forces': layer_states.get('basin_forces', {})
        }
        
        # Add predicted E/N/C label if provided
        if predicted_label is not None:
            entry['predicted'] = predicted_label
        
        self.clusters[basin_index].append(entry)
        
        # Update statistics
        stats = self.cluster_stats[basin_index]
        stats['count'] += 1
        stats['total_confidence'] += confidence
        stats['avg_confidence'] = stats['total_confidence'] / stats['count']
    
    def get_statistics(self) -> Dict:
        """Get cluster statistics."""
        return {
            'basin_0_cold': {
                'count': self.cluster_stats[0]['count'],
                'avg_confidence': float(self.cluster_stats[0]['avg_confidence']),
                'description': 'Cold basin (entailment-like patterns)'
            },
            'basin_1_far': {
                'count': self.cluster_stats[1]['count'],
                'avg_confidence': float(self.cluster_stats[1]['avg_confidence']),
                'description': 'Far basin (contradiction-like patterns)'
            },
            'basin_2_city': {
                'count': self.cluster_stats[2]['count'],
                'avg_confidence': float(self.cluster_stats[2]['avg_confidence']),
                'description': 'City basin (neutral-like patterns)'
            }
        }
    
    def export_clusters(self, output_dir: Path):
        """
        Export clusters to JSON files.
        
        Args:
            output_dir: Directory to save cluster files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each cluster
        for basin_idx, cluster_name in [(0, 'cluster_0_cold'), 
                                        (1, 'cluster_1_far'), 
                                        (2, 'cluster_2_city')]:
            cluster_file = output_dir / f'{cluster_name}.json'
            
            cluster_data = {
                'basin_index': basin_idx,
                'basin_name': cluster_name,
                'count': len(self.clusters[basin_idx]),
                'statistics': {
                    'avg_confidence': self.cluster_stats[basin_idx]['avg_confidence'],
                    'total_entries': self.cluster_stats[basin_idx]['count']
                },
                'entries': self.clusters[basin_idx]
            }
            
            with open(cluster_file, 'w') as f:
                json.dump(cluster_data, f, indent=2)
        
        # Export summary
        summary_file = output_dir / 'cluster_summary.json'
        summary = {
            'total_clusters': 3,
            'statistics': self.get_statistics(),
            'total_entries': sum(len(c) for c in self.clusters.values())
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def print_statistics(self):
        """Print cluster statistics."""
        print("\n" + "=" * 70)
        print("GEOMETRY-DISCOVERED CLUSTERS")
        print("=" * 70)
        print()
        
        stats = self.get_statistics()
        for basin_name, basin_stats in stats.items():
            print(f"{basin_name.upper()}:")
            print(f"  Count: {basin_stats['count']}")
            print(f"  Avg Confidence: {basin_stats['avg_confidence']:.4f}")
            print(f"  Description: {basin_stats['description']}")
            print()
        
        total = sum(s['count'] for s in stats.values())
        print(f"Total entries: {total}")
        print("=" * 70)
        print()

