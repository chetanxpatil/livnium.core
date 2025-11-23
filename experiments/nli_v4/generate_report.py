"""
Generate Numerical Report for Livnium NLI v4

Extracts all numerical metrics, parameters, and statistics from the system.
"""

import os
import sys
import json
import pickle
from typing import Dict, Any
from pathlib import Path

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.nli_v4.layer2_basin import Layer2Basin
from experiments.nli_v4.layer1_curvature import Layer1Curvature
from experiments.nli_v4.layer5_temporal_stability import Layer5TemporalStability
from experiments.nli_v4.layer7_decision import Layer7Decision
from experiments.nli_v4.auto_physics import AutoPhysicsEngine, load_overrides
from experiments.nli_v4.autonomous_meaning_engine import AutonomousMeaningEngine
from experiments.nli_simple.native_chain import SimpleLexicon


def analyze_layer_parameters() -> Dict[str, Any]:
    """Extract all numerical parameters from layers with maximum detail."""
    params = {}
    
    # Layer 1: Curvature (Cold & Distance)
    layer1 = Layer1Curvature()
    params['layer1'] = {
        'history_window': layer1.history_window,
        'entropy_scale': float(layer1.entropy_scale),
        'entropy_base': 0.01,  # From AutoPhysicsEngine
        'entropy_max': 0.01 + 0.02,  # base + scale (max imbalance)
        'description': 'Cold & Distance Curvature',
        'computations': {
            'cold_density': 'max(0.0, resonance_with_entropy) * stability',
            'distance': 'max(0.0, -resonance_with_entropy) + 0.3 * edge_distance',
            'stability': '1.0 - min(resonance_variance, 1.0)',
            'curvature': 'abs(r_t - 2.0 * r_t1 + r_t2)'
        }
    }
    
    # Layer 2: Basin (Cold & Far)
    basin = Layer2Basin()
    params['layer2'] = {
        'reinforcement_rate': basin.reinforcement_rate,
        'decay_rate': basin.decay_rate,
        'capacity': basin.capacity,
        'cold_basin_depth': float(basin.cold_basin_depth),
        'far_basin_depth': float(basin.far_basin_depth),
        'monopoly_threshold': basin.monopoly_threshold,
        'heat_wave_strength': basin.heat_wave_strength,
        'temperature_penalty_threshold': 0.7,  # When basin ratio > 0.7, apply penalty
        'temperature_penalty_max': 0.3,  # Up to 30% flattening
        'description': 'Cold & Far Basins',
        'computations': {
            'cold_attraction': 'cold_weight * (1.0 + curvature) * (1.0 + cold_density)',
            'far_attraction': 'far_weight * (1.0 + curvature) * (1.0 + distance) * repulsion_boost',
            'repulsion': 'max(0.0, -resonance) * (distance**2)',
            'repulsion_boost': '1.0 + (repulsion * 0.3)',
            'basin_temperature_penalty': '(basin_ratio - 0.7) * 0.3 if basin_ratio > 0.7'
        },
        'prediction_statistics': {
            'total_predictions': Layer2Basin._shared_total_predictions,
            'entailment_count': Layer2Basin._shared_prediction_counts.get('entailment', 0),
            'contradiction_count': Layer2Basin._shared_prediction_counts.get('contradiction', 0),
            'neutral_count': Layer2Basin._shared_prediction_counts.get('neutral', 0)
        }
    }
    
    # Layer 3: Valley (City)
    params['layer3'] = {
        'city_threshold': 0.15,
        'city_gravity': 0.7,
        'attraction_ratio_threshold': 0.35,
        'min_attraction_for_city': 0.05,
        'city_pull_min': 0.1,  # Minimum pull to ensure city has mass
        'description': 'The City (Neutral Valley)',
        'computations': {
            'attraction_ratio': 'abs(cold_attraction - far_attraction) / max_attraction',
            'overlap_strength': '1.0 - min(attraction_ratio, 1.0)',
            'city_pull': 'overlap_strength * 0.7 * (min_attraction + 0.1)',
            'city_forms': 'attraction_ratio < 0.15 and max_attraction > 0.05'
        }
    }
    
    # Layer 4: Meta Routing
    params['layer4'] = {
        'valley_route_threshold': 0.7,
        'peak_route_threshold': 0.8,
        'routing_history_window': 100,
        'description': 'Meta Routing',
        'routes': ['valley', 'peak', 'transition']
    }
    
    # Layer 5: Temporal Stability (Thermodynamics)
    layer5 = Layer5TemporalStability()
    params['layer5'] = {
        'stability_window': layer5.stability_window,
        'temperature': float(layer5.temperature),
        'moksha_confidence_threshold': 0.8,
        'stability_confidence_threshold': 0.7,
        'frozen_threshold': 0.1,
        'overheated_threshold': 0.9,
        'description': 'Temporal Stability & Thermodynamics',
        'computations': {
            'temperature': 'min(1.0, max(0.0, avg_entropy * 50.0))',
            'is_frozen': 'temperature < 0.1',
            'is_overheated': 'temperature > 0.9',
            'is_stable': 'label_stable and confidence_stable and not frozen and not overheated',
            'is_moksha': 'is_stable and avg_confidence > 0.8'
        }
    }
    
    # Layer 6: Semantic Memory
    params['layer6'] = {
        'polarity_update_strength': 0.15,
        'description': 'Semantic Memory (Word Polarities)',
        'polarity_dimensions': 3  # [entailment, contradiction, neutral]
    }
    
    # Layer 7: Decision (Force-Competition)
    layer7 = Layer7Decision()
    params['layer7'] = {
        'weak_force_threshold': layer7.weak_force_threshold,
        'balance_threshold': layer7.balance_threshold,
        'decision_rule': 'force_competition',
        'description': 'Force-Competition Decision Layer',
        'rules': {
            'rule1_weak_forces': 'if max_force < 0.05 → neutral (city)',
            'rule2_balanced_forces': 'if ratio < 0.15 → neutral (city)',
            'rule3_force_competition': 'if cold > far → entailment, else → contradiction'
        }
    }
    
    # Auto-Physics Engine
    auto_physics = AutoPhysicsEngine()
    overrides = load_overrides()  # Load current overrides
    params['auto_physics'] = {
        'entropy_base': auto_physics.entropy_base,
        'entropy_scale': auto_physics.entropy_scale,
        'repulsion_strength': auto_physics.repulsion_strength,
        'dominance_threshold': auto_physics.dominance_threshold,
        'repulsion_boost_cap': 1.2,  # Max 20% boost
        'far_depth_max_ratio': 1.5,  # Far can't exceed cold by 1.5x
        'flatten_factor_max': 0.2,  # Up to 20% reduction
        'exploration_boost': 0.1,  # 10% boost for exploration
        'description': 'Auto-Physics Engine (Self-Organizing Universe)',
        'laws': {
            'law1_entropy': 'entropy = 0.01 + 0.02 * class_imbalance',
            'law2_repulsion': 'repulsion_boost = 1.0 + (repulsion_strength * cold_ratio)',
            'law3_anti_monopoly': 'if max_ratio > 0.6: flatten dominant basin by up to 20%'
        },
        'thermodynamic_loop': 'curvature → basins → gravity → memory → resonance → curvature',
        'overrides_active': len(overrides) > 0,
        'overrides': overrides if overrides else {}
    }
    
    # Autonomous Meaning Engine (AME)
    ame = AutonomousMeaningEngine()
    params['ame'] = {
        'turbulence_base': ame.turbulence_base,
        'turbulence_scale': ame.turbulence_scale,
        'city_dominance_threshold': ame.city_dominance_threshold,
        'split_threshold': ame.split_threshold,
        'max_basins': ame.max_basins,
        'curvature_threshold': ame.curvature_threshold,
        'hysteresis_alpha': ame.hysteresis_alpha,
        'alignment_strength': ame.alignment_strength,
        'description': 'Autonomous Meaning Engine (Full Semantic Cosmology)',
        'seven_mechanisms': {
            'mechanism1_turbulence': 'Entropy scales with city dominance',
            'mechanism2_competitive_polarity': 'Basins compete for words',
            'mechanism3_basin_splitting': 'Basins split when >70%',
            'mechanism4_curvature_routing': 'High curvature pushes out of city',
            'mechanism5_hysteresis': 'Meaning has inertia (60% current, 40% previous)',
            'mechanism6_long_range_alignment': 'Basin centers pull sentences',
            'mechanism7_continuous_evolution': 'Universe runs itself'
        },
        'basin_statistics': ame.get_statistics(),
        'total_assignments': ame.total_assignments
    }
    
    return params


def analyze_cluster_statistics() -> Dict[str, Any]:
    """Analyze geometry-discovered clusters."""
    cluster_dir = Path(__file__).parent / 'clusters'
    summary_path = cluster_dir / 'cluster_summary.json'
    
    stats = {
        'clusters_exist': summary_path.exists(),
        'total_clusters': 3,
        'basin_statistics': {}
    }
    
    if summary_path.exists():
        try:
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            stats['total_entries'] = summary.get('total_entries', 0)
            stats['statistics'] = summary.get('statistics', {})
            
            # Extract basin counts and ratios
            for basin_name, basin_stats in stats['statistics'].items():
                stats['basin_statistics'][basin_name] = {
                    'count': basin_stats.get('count', 0),
                    'avg_confidence': basin_stats.get('avg_confidence', 0.0),
                    'ratio': basin_stats.get('count', 0) / max(stats['total_entries'], 1)
                }
        except Exception as e:
            stats['load_error'] = str(e)
    
    return stats


def analyze_brain_state() -> Dict[str, Any]:
    """Analyze the saved brain state with maximum detail."""
    brain_path = Path(__file__).parent / 'brain_state.pkl'
    
    stats = {
        'brain_file_exists': brain_path.exists(),
        'brain_file_size_kb': 0,
        'brain_file_size_bytes': 0,
        'words_learned': 0,
        'polarity_distribution': {},
        'polarity_statistics': {},
        'word_samples': {}
    }
    
    if brain_path.exists():
        file_size = brain_path.stat().st_size
        stats['brain_file_size_kb'] = round(file_size / 1024, 2)
        stats['brain_file_size_bytes'] = file_size
        
        try:
            lexicon = SimpleLexicon()
            lexicon.load_from_file(str(brain_path))
            
            stats['words_learned'] = len(lexicon.polarity_store)
            
            # Analyze polarity distribution
            polarity_counts = {'entailment': 0, 'contradiction': 0, 'neutral': 0}
            polarity_strengths = {'entailment': [], 'contradiction': [], 'neutral': []}
            strong_polarity_words = {'entailment': [], 'contradiction': [], 'neutral': []}
            
            for word, polarity in lexicon.polarity_store.items():
                if len(polarity) >= 3:
                    # Find dominant class
                    max_idx = max(range(3), key=lambda i: polarity[i])
                    max_strength = polarity[max_idx]
                    
                    if max_idx == 0:
                        polarity_counts['entailment'] += 1
                        polarity_strengths['entailment'].append(max_strength)
                        if max_strength > 0.6:
                            strong_polarity_words['entailment'].append((word, max_strength))
                    elif max_idx == 1:
                        polarity_counts['contradiction'] += 1
                        polarity_strengths['contradiction'].append(max_strength)
                        if max_strength > 0.6:
                            strong_polarity_words['contradiction'].append((word, max_strength))
                    else:
                        polarity_counts['neutral'] += 1
                        polarity_strengths['neutral'].append(max_strength)
                        if max_strength > 0.6:
                            strong_polarity_words['neutral'].append((word, max_strength))
            
            stats['polarity_distribution'] = polarity_counts
            
            # Compute statistics
            for label in ['entailment', 'contradiction', 'neutral']:
                if polarity_strengths[label]:
                    stats['polarity_statistics'][label] = {
                        'count': polarity_counts[label],
                        'avg_strength': float(sum(polarity_strengths[label]) / len(polarity_strengths[label])),
                        'max_strength': float(max(polarity_strengths[label])),
                        'min_strength': float(min(polarity_strengths[label])),
                        'strong_words_count': len(strong_polarity_words[label])
                    }
            
            # Sample words by strength
            for label in ['entailment', 'contradiction', 'neutral']:
                sorted_words = sorted(strong_polarity_words[label], key=lambda x: x[1], reverse=True)[:5]
                stats['word_samples'][label] = [{'word': w, 'strength': float(s)} for w, s in sorted_words]
            
            # Overall sample words
            sample_words = list(lexicon.polarity_store.keys())[:10]
            stats['sample_words'] = sample_words
            
            # Memory efficiency
            stats['bytes_per_word'] = round(file_size / max(len(lexicon.polarity_store), 1), 2)
            
        except Exception as e:
            stats['load_error'] = str(e)
    
    return stats


def analyze_architecture() -> Dict[str, Any]:
    """Analyze the architecture structure with maximum detail."""
    arch = {
        'total_layers': 8,  # Layer 0-7
        'layer_names': [
            'Layer 0: Pure Resonance',
            'Layer 1: Cold & Distance Curvature',
            'Layer 2: Cold & Far Basins',
            'Layer 3: The City (Valley)',
            'Layer 4: Meta Routing',
            'Layer 5: Temporal Stability & Thermodynamics',
            'Layer 6: Semantic Memory',
            'Layer 7: Force-Competition Decision'
        ],
        'metaphor': {
            'entailment': 'Cold Region (dense, stable, pulls inward)',
            'contradiction': 'Far Lands (high distance, edge of continent)',
            'neutral': 'The City (balance point where forces overlap)'
        },
        'physics': {
            'cold_density_computation': 'max(0.0, resonance_with_entropy) * stability',
            'distance_computation': 'max(0.0, -resonance_with_entropy) + 0.3 * edge_distance',
            'city_formation': 'attraction_ratio < 0.15 and max_attraction > 0.05',
            'city_gravity': 'overlap_strength * 0.7 * (min_attraction + 0.1)',
            'entropy_injection': 'np.random.normal(0.0, entropy_scale)',
            'repulsion_field': 'max(0.0, -resonance) * (distance**2)',
            'force_competition': 'if max_force < 0.05 → city, elif ratio < 0.15 → city, else → cold vs far'
        },
        'thermodynamic_loop': {
            'closed': True,
            'cycle': 'curvature → basins → gravity → memory → resonance → curvature',
            'auto_physics': True,
            'self_organizing': True
        },
        'three_laws': {
            'law1_entropy': 'Automatic Entropy Injection (scales with class imbalance)',
            'law2_repulsion': 'Repulsion Field for Contradiction (far lands push away)',
            'law3_anti_monopoly': 'Dynamic Basin Depth (prevents collapse)'
        }
    }
    
    return arch


def count_code_metrics() -> Dict[str, Any]:
    """Count lines of code and complexity metrics."""
    metrics = {
        'files': {},
        'total_lines': 0,
        'total_files': 0
    }
    
    nli_v4_dir = Path(__file__).parent
    
    for py_file in nli_v4_dir.glob('*.py'):
        if py_file.name == 'generate_report.py':
            continue
        
        with open(py_file, 'r') as f:
            lines = f.readlines()
            code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            
            metrics['files'][py_file.name] = {
                'total_lines': len(lines),
                'code_lines': len(code_lines),
                'comment_lines': len([l for l in lines if l.strip().startswith('#')])
            }
            
            metrics['total_lines'] += len(lines)
            metrics['total_files'] += 1
    
    return metrics


def generate_full_report() -> Dict[str, Any]:
    """Generate complete numerical report."""
    report = {
        'system': 'Livnium NLI v4: Planet Architecture',
        'version': '4.0',
        'architecture': analyze_architecture(),
        'layer_parameters': analyze_layer_parameters(),
        'brain_statistics': analyze_brain_state(),
        'cluster_statistics': analyze_cluster_statistics(),
        'code_metrics': count_code_metrics(),
        'key_numbers': {}
    }
    
    # Extract key numbers with maximum detail
    basin = Layer2Basin()
    layer1 = Layer1Curvature()
    layer5 = Layer5TemporalStability()
    layer7 = Layer7Decision()
    auto_physics = AutoPhysicsEngine()
    ame = AutonomousMeaningEngine()
    overrides = load_overrides()
    
    # Compute prediction distribution
    pred_counts = Layer2Basin._shared_prediction_counts
    total_preds = Layer2Basin._shared_total_predictions
    pred_ratios = {k: v / total_preds if total_preds > 0 else 0.0 
                   for k, v in pred_counts.items()}
    class_imbalance = max(pred_ratios.values()) - min(pred_ratios.values()) if total_preds > 0 else 0.0
    
    # Get AME statistics
    ame_stats = ame.get_statistics()
    ame_total = ame.total_assignments
    
    report['key_numbers'] = {
        # Architecture
        'total_layers': 8,
        'total_code_lines': report['code_metrics']['total_lines'],
        'total_python_files': report['code_metrics']['total_files'],
        
        # Basin System
        'cold_basin_depth': float(basin.cold_basin_depth),
        'far_basin_depth': float(basin.far_basin_depth),
        'basin_depth_ratio': float(basin.cold_basin_depth / (basin.far_basin_depth + 1e-6)),
        'reinforcement_rate': basin.reinforcement_rate,
        'decay_rate': basin.decay_rate,
        'basin_capacity': basin.capacity,
        'monopoly_threshold': basin.monopoly_threshold,
        'heat_wave_strength': basin.heat_wave_strength,
        
        # Curvature & Entropy
        'curvature_history_window': layer1.history_window,
        'entropy_scale': float(layer1.entropy_scale),
        'entropy_base': auto_physics.entropy_base,
        'entropy_scale_factor': auto_physics.entropy_scale,
        'current_entropy': float(auto_physics.entropy_base + auto_physics.entropy_scale * class_imbalance),
        
        # City (Neutral Valley)
        'city_gravity': 0.7,
        'city_threshold': 0.15,
        'city_pull_min': 0.1,
        'attraction_ratio_threshold': 0.35,
        
        # Decision Layer
        'weak_force_threshold': layer7.weak_force_threshold,
        'balance_threshold': layer7.balance_threshold,
        
        # Thermodynamics
        'system_temperature': float(layer5.temperature),
        'stability_window': layer5.stability_window,
        'frozen_threshold': 0.1,
        'overheated_threshold': 0.9,
        
        # Repulsion Field
        'repulsion_strength': auto_physics.repulsion_strength,
        'repulsion_boost_cap': 1.2,
        'far_depth_max_ratio': 1.5,
        
        # Anti-Monopoly
        'dominance_threshold': auto_physics.dominance_threshold,
        'flatten_factor_max': 0.2,
        'exploration_boost': 0.1,
        
        # Prediction Statistics
        'total_predictions': total_preds,
        'entailment_predictions': pred_counts.get('entailment', 0),
        'contradiction_predictions': pred_counts.get('contradiction', 0),
        'neutral_predictions': pred_counts.get('neutral', 0),
        'entailment_ratio': float(pred_ratios.get('entailment', 0.0)),
        'contradiction_ratio': float(pred_ratios.get('contradiction', 0.0)),
        'neutral_ratio': float(pred_ratios.get('neutral', 0.0)),
        'class_imbalance': float(class_imbalance),
        
        # Brain
        'words_learned': report['brain_statistics']['words_learned'],
        'brain_size_kb': report['brain_statistics']['brain_file_size_kb'],
        'brain_size_bytes': report['brain_statistics'].get('brain_file_size_bytes', 0),
        'bytes_per_word': report['brain_statistics'].get('bytes_per_word', 0.0),
        
        # AME (Autonomous Meaning Engine)
        'turbulence_base': ame.turbulence_base,
        'turbulence_scale': ame.turbulence_scale,
        'city_dominance_threshold': ame.city_dominance_threshold,
        'split_threshold': ame.split_threshold,
        'curvature_threshold': ame.curvature_threshold,
        'hysteresis_alpha': ame.hysteresis_alpha,
        'alignment_strength': ame.alignment_strength,
        'ame_total_assignments': ame_total,
        'ame_active_basins': len(ame_stats) if ame_stats else 3,
        
        # Auto-Physics Overrides
        'overrides_active': len(overrides) > 0,
        'override_entropy_scale': overrides.get('entropy_scale', None),
        'override_repulsion_strength': overrides.get('repulsion_strength', None),
        'override_turbulence_scale': overrides.get('turbulence_scale', None),
        
        # Cluster Statistics (if available)
        'cluster_total_entries': report['cluster_statistics'].get('total_entries', 0),
        'cluster_basin_0_count': report['cluster_statistics'].get('basin_statistics', {}).get('basin_0_cold', {}).get('count', 0),
        'cluster_basin_1_count': report['cluster_statistics'].get('basin_statistics', {}).get('basin_1_far', {}).get('count', 0),
        'cluster_basin_2_count': report['cluster_statistics'].get('basin_statistics', {}).get('basin_2_city', {}).get('count', 0),
        'cluster_city_ratio': report['cluster_statistics'].get('basin_statistics', {}).get('basin_2_city', {}).get('ratio', 0.0) if report['cluster_statistics'].get('basin_statistics') else 0.0
    }
    
    return report


def print_report(report: Dict[str, Any]):
    """Print formatted report."""
    print("=" * 80)
    print(f"{report['system']} - NUMERICAL REPORT")
    print("=" * 80)
    print()
    
    print("📊 KEY NUMBERS")
    print("-" * 80)
    for key, value in report['key_numbers'].items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:>15.4f}")
        else:
            print(f"  {key:.<40} {value:>15}")
    print()
    
    print("🏗️  ARCHITECTURE")
    print("-" * 80)
    print(f"  Total Layers: {report['architecture']['total_layers']}")
    print()
    for i, layer_name in enumerate(report['architecture']['layer_names']):
        print(f"  {i}. {layer_name}")
    print()
    
    print("🌍 PLANET METAPHOR")
    print("-" * 80)
    for key, value in report['architecture']['metaphor'].items():
        print(f"  {key.capitalize():.<20} {value}")
    print()
    
    print("⚙️  LAYER PARAMETERS (MAXIMUM DETAIL)")
    print("-" * 80)
    for layer_name, params in report['layer_parameters'].items():
        print(f"\n  {layer_name.upper()}: {params.get('description', 'N/A')}")
        for key, value in params.items():
            if key not in ['description', 'computations', 'rules', 'laws', 'prediction_statistics', 'routes']:
                if isinstance(value, float):
                    print(f"    {key:.<30} {value:>15.4f}")
                elif isinstance(value, dict):
                    print(f"    {key}:")
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            print(f"      {sub_key:.<28} {sub_value:>15.4f}")
                        else:
                            print(f"      {sub_key:.<28} {sub_value:>15}")
                else:
                    print(f"    {key:.<30} {value:>15}")
        if 'computations' in params:
            print(f"    Computations:")
            for comp_key, comp_value in params['computations'].items():
                print(f"      {comp_key:.<28} {comp_value}")
        if 'rules' in params:
            print(f"    Decision Rules:")
            for rule_key, rule_value in params['rules'].items():
                print(f"      {rule_key:.<28} {rule_value}")
        if 'laws' in params:
            print(f"    Three Laws:")
            for law_key, law_value in params['laws'].items():
                print(f"      {law_key:.<28} {law_value}")
        if 'prediction_statistics' in params:
            print(f"    Prediction Statistics:")
            for stat_key, stat_value in params['prediction_statistics'].items():
                print(f"      {stat_key:.<28} {stat_value:>15}")
        if 'seven_mechanisms' in params:
            print(f"    Seven Mechanisms:")
            for mech_key, mech_value in params['seven_mechanisms'].items():
                print(f"      {mech_key:.<28} {mech_value}")
        if 'basin_statistics' in params:
            print(f"    Basin Statistics:")
            for basin_key, basin_value in params['basin_statistics'].items():
                if isinstance(basin_value, dict):
                    print(f"      {basin_key}:")
                    for sub_key, sub_value in basin_value.items():
                        if isinstance(sub_value, float):
                            print(f"        {sub_key:.<26} {sub_value:>15.4f}")
                        else:
                            print(f"        {sub_key:.<26} {sub_value:>15}")
                else:
                    print(f"      {basin_key:.<28} {basin_value:>15}")
        if 'overrides' in params and params.get('overrides_active', False):
            print(f"    Active Overrides:")
            for override_key, override_value in params['overrides'].items():
                if isinstance(override_value, float):
                    print(f"      {override_key:.<28} {override_value:>15.4f}")
                else:
                    print(f"      {override_key:.<28} {override_value:>15}")
    print()
    
    print("🧠 BRAIN STATISTICS")
    print("-" * 80)
    brain = report['brain_statistics']
    print(f"  Brain file exists: {brain['brain_file_exists']}")
    if brain['brain_file_exists']:
        print(f"  Brain file size: {brain['brain_file_size_kb']} KB ({brain.get('brain_file_size_bytes', 0)} bytes)")
        print(f"  Words learned: {brain['words_learned']}")
        print(f"  Bytes per word: {brain.get('bytes_per_word', 0.0):.2f}")
        if brain['polarity_distribution']:
            print(f"\n  Polarity distribution:")
            for label, count in brain['polarity_distribution'].items():
                ratio = count / brain['words_learned'] if brain['words_learned'] > 0 else 0.0
                print(f"    {label:.<20} {count:>10} ({ratio*100:>5.1f}%)")
        if brain.get('polarity_statistics'):
            print(f"\n  Polarity strength statistics:")
            for label, stats in brain['polarity_statistics'].items():
                print(f"    {label}:")
                print(f"      Count: {stats['count']}")
                print(f"      Avg strength: {stats['avg_strength']:.4f}")
                print(f"      Max strength: {stats['max_strength']:.4f}")
                print(f"      Min strength: {stats['min_strength']:.4f}")
                print(f"      Strong words (>0.6): {stats['strong_words_count']}")
        if brain.get('word_samples'):
            print(f"\n  Strongest words by class:")
            for label, samples in brain['word_samples'].items():
                if samples:
                    words_str = ', '.join([f"{w['word']}({w['strength']:.2f})" for w in samples[:3]])
                    print(f"    {label:.<15} {words_str}")
        if 'sample_words' in brain:
            print(f"\n  Sample words: {', '.join(brain['sample_words'][:5])}...")
    print()
    
    print("📝 CODE METRICS")
    print("-" * 80)
    code = report['code_metrics']
    print(f"  Total files: {code['total_files']}")
    print(f"  Total lines: {code['total_lines']}")
    print()
    print("  File breakdown:")
    for filename, metrics in sorted(code['files'].items()):
        print(f"    {filename:.<30} {metrics['total_lines']:>5} lines ({metrics['code_lines']} code)")
    print()
    
    print("🔢 PHYSICS PARAMETERS")
    print("-" * 80)
    physics = report['architecture']['physics']
    for key, value in physics.items():
        print(f"  {key:.<30} {value}")
    print()
    
    print("🌐 THERMODYNAMIC LOOP")
    print("-" * 80)
    loop = report['architecture']['thermodynamic_loop']
    print(f"  Closed: {loop['closed']}")
    print(f"  Auto-Physics: {loop['auto_physics']}")
    print(f"  Self-Organizing: {loop['self_organizing']}")
    print(f"  Cycle: {loop['cycle']}")
    print()
    
    print("⚖️  THREE LAWS OF AUTO-PHYSICS")
    print("-" * 80)
    laws = report['architecture']['three_laws']
    for law_key, law_value in laws.items():
        print(f"  {law_key:.<30} {law_value}")
    print()
    
    print("📈 PREDICTION DISTRIBUTION")
    print("-" * 80)
    key_nums = report['key_numbers']
    if key_nums['total_predictions'] > 0:
        print(f"  Total predictions: {key_nums['total_predictions']}")
        print(f"  Entailment: {key_nums['entailment_predictions']} ({key_nums['entailment_ratio']*100:.1f}%)")
        print(f"  Contradiction: {key_nums['contradiction_predictions']} ({key_nums['contradiction_ratio']*100:.1f}%)")
        print(f"  Neutral: {key_nums['neutral_predictions']} ({key_nums['neutral_ratio']*100:.1f}%)")
        print(f"  Class imbalance: {key_nums['class_imbalance']:.4f}")
    else:
        print("  No predictions tracked yet (system not trained)")
    print()
    
    print("🌡️  THERMODYNAMIC STATE")
    print("-" * 80)
    print(f"  System temperature: {key_nums['system_temperature']:.4f}")
    print(f"  Current entropy: {key_nums['current_entropy']:.4f}")
    print(f"  Entropy base: {key_nums['entropy_base']:.4f}")
    print(f"  Entropy scale: {key_nums['entropy_scale_factor']:.4f}")
    print(f"  Frozen threshold: {key_nums['frozen_threshold']:.4f}")
    print(f"  Overheated threshold: {key_nums['overheated_threshold']:.4f}")
    print()
    
    print("🤖 AUTONOMOUS MEANING ENGINE (AME)")
    print("-" * 80)
    print(f"  Turbulence base: {key_nums.get('turbulence_base', 0.0):.4f}")
    print(f"  Turbulence scale: {key_nums.get('turbulence_scale', 0.0):.4f}")
    print(f"  City dominance threshold: {key_nums.get('city_dominance_threshold', 0.0):.4f}")
    print(f"  Split threshold: {key_nums.get('split_threshold', 0.0):.4f}")
    print(f"  Curvature threshold: {key_nums.get('curvature_threshold', 0.0):.4f}")
    print(f"  Hysteresis alpha: {key_nums.get('hysteresis_alpha', 0.0):.4f}")
    print(f"  Alignment strength: {key_nums.get('alignment_strength', 0.0):.4f}")
    print(f"  Total AME assignments: {key_nums.get('ame_total_assignments', 0)}")
    print(f"  Active basins: {key_nums.get('ame_active_basins', 3)}")
    print()
    
    print("⚙️  AUTO-PHYSICS OVERRIDES")
    print("-" * 80)
    if key_nums.get('overrides_active', False):
        print(f"  ✓ Overrides active: YES")
        if key_nums.get('override_entropy_scale') is not None:
            print(f"    Entropy scale override: {key_nums['override_entropy_scale']:.4f}")
        if key_nums.get('override_repulsion_strength') is not None:
            print(f"    Repulsion strength override: {key_nums['override_repulsion_strength']:.4f}")
        if key_nums.get('override_turbulence_scale') is not None:
            print(f"    Turbulence scale override: {key_nums['override_turbulence_scale']:.4f}")
    else:
        print(f"  Overrides active: NO (using defaults)")
    print()
    
    print("🌍 GEOMETRY-DISCOVERED CLUSTERS")
    print("-" * 80)
    cluster_stats = report.get('cluster_statistics', {})
    if cluster_stats.get('clusters_exist', False):
        total_entries = cluster_stats.get('total_entries', 0)
        print(f"  Total entries: {total_entries}")
        if cluster_stats.get('basin_statistics'):
            for basin_name, basin_data in cluster_stats['basin_statistics'].items():
                count = basin_data.get('count', 0)
                ratio = basin_data.get('ratio', 0.0)
                avg_conf = basin_data.get('avg_confidence', 0.0)
                print(f"  {basin_name}: {count} entries ({ratio*100:.1f}%), avg confidence: {avg_conf:.4f}")
    else:
        print("  No cluster data available (run unsupervised training first)")
    print()
    
    print("=" * 80)
    print("Report generated successfully!")
    print("=" * 80)


def main():
    """Generate and save report."""
    report = generate_full_report()
    
    # Print to console
    print_report(report)
    
    # Save JSON
    report_path = Path(__file__).parent / 'numerical_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Report saved to: {report_path}")
    print()


if __name__ == '__main__':
    main()

