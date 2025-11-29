# 📊 Livnium NLI v4: Numerical Report

**Planet Architecture - Complete Metrics & Statistics**

---

## 🔢 Key Numbers Summary

| Metric | Value |
|--------|-------|
| **Total Layers** | 8 |
| **Cold Basin Depth** | 1.0000 |
| **Far Basin Depth** | 1.0000 |
| **Reinforcement Rate** | 0.3000 |
| **Decay Rate** | 0.0100 |
| **Basin Capacity** | 200.0000 |
| **City Gravity** | 0.7000 |
| **City Threshold** | 0.1500 |
| **Words Learned** | 7,900 |
| **Brain Size** | 314.84 KB |
| **Total Code Lines** | 1,310 |
| **Python Files** | 11 |

---

## 🏗️ Architecture Structure

### 8-Layer Stack

1. **Layer 0**: Pure Resonance (bedrock)
2. **Layer 1**: Cold & Distance Curvature
3. **Layer 2**: Cold & Far Basins
4. **Layer 3**: The City (Valley)
5. **Layer 4**: Meta Routing
6. **Layer 5**: Temporal Stability (Moksha)
7. **Layer 6**: Semantic Memory
8. **Layer 7**: Decision

---

## 🌍 Planet Metaphor Parameters

### Entailment = Cold Region
- **Dense air**: High density computation
- **Stable**: Low variance, high stability
- **Pulls inward**: Strong attraction force

### Contradiction = Far Lands
- **High distance**: Edge of semantic continent
- **Different climate**: Opposite pole
- **Thin atmosphere**: Separation force

### Neutral = The City
- **Mixed temperatures**: Balance point
- **Crowded**: All roads cross here
- **Real curvature**: Gravitational mass (0.7)

---

## ⚙️ Layer Parameters

### Layer 1: Cold & Distance Curvature
- **History Window**: 10
- **Cold Density**: `max(0.0, resonance) * stability`
- **Distance**: `max(0.0, -resonance) + 0.3 * edge_distance`

### Layer 2: Cold & Far Basins
- **Reinforcement Rate**: 0.3000
- **Decay Rate**: 0.0100
- **Capacity**: 200.0000
- **Cold Basin Depth**: 1.0000
- **Far Basin Depth**: 1.0000

### Layer 3: The City (Valley)
- **City Threshold**: 0.1500 (15% overlap triggers city)
- **City Gravity**: 0.7000 (gravitational constant)
- **Attraction Ratio Threshold**: 0.3500 (35% for medium pull)

---

## 🧠 Brain Statistics

### Learned Knowledge
- **Total Words**: 7,900
- **Brain File Size**: 314.84 KB
- **Storage Format**: Pickle (polarity vectors)

### Polarity Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| **Entailment** | 1,784 | 22.6% |
| **Contradiction** | 2,796 | 35.4% |
| **Neutral** | 3,320 | 42.0% |

**Insight**: Neutral has the highest word count, showing the city is well-populated with balanced words.

---

## 📝 Code Metrics

### File Structure
| File | Lines | Code Lines |
|------|-------|------------|
| `train_v4.py` | 357 | 264 |
| `layer3_valley.py` | 172 | 117 |
| `layer2_basin.py` | 151 | 102 |
| `layer1_curvature.py` | 121 | 73 |
| `layered_classifier.py` | 115 | 78 |
| `layer5_temporal_stability.py` | 81 | 59 |
| `layer6_semantic_memory.py` | 81 | 63 |
| `layer7_decision.py` | 81 | 62 |
| `layer4_meta_routing.py` | 63 | 47 |
| `layer0_resonance.py` | 49 | 34 |
| `__init__.py` | 39 | 34 |

**Total**: 1,310 lines across 11 files

---

## 🔢 Physics Parameters

### Cold Density Computation
```
cold_density = max(0.0, resonance) * stability
```
- Positive resonance → cold region
- Stability = 1.0 - variance (higher = more stable)

### Distance Computation
```
distance = max(0.0, -resonance) + 0.3 * edge_distance
```
- Negative resonance → far lands
- Edge distance captures "edge of continent"

### City Formation
```
city_forms = (attraction_ratio < 0.15) and (max_attraction > 0.05)
```
- When cold and far are within 15% → city appears
- Minimum attraction required: 0.05

### City Gravity
```
city_pull = overlap_strength * 0.7 * (min_attraction + 0.1)
```
- Overlap strength: 1.0 when cold ≈ far
- Gravity constant: 0.7
- Minimum pull: 0.1 ensures city always has some mass

---

## 📈 Learning Dynamics

### Basin Reinforcement
- **Growth Rate**: 0.3 per correct example
- **Logistic Growth**: Slows as depth approaches capacity (200)
- **Decay**: 0.01 natural decay for opposite basin

### Basin Capacity
- **Maximum Depth**: 200.0
- **Prevents**: Infinite growth
- **Stabilizes**: System converges to stable attractors

---

## 🎯 Key Insights

1. **City Gravity (0.7) > Medium Gravity (0.6)**: The city has stronger pull than before
2. **Neutral Words (42%) > Contradiction (35%) > Entailment (23%)**: The city is well-populated
3. **Basin Capacity (200)**: Prevents runaway learning
4. **City Threshold (0.15)**: Tight threshold ensures city only forms when forces truly balance
5. **7,900 Words Learned**: Substantial semantic knowledge base

---

## 📊 System Health

✅ **Architecture**: 8 layers, clean separation  
✅ **Learning**: 7,900 words, balanced distribution  
✅ **Physics**: Cold/distance/city metaphor implemented  
✅ **Code**: 1,310 lines, well-structured  
✅ **Brain**: 314.84 KB, actively learning  

---

**Report Generated**: `numerical_report.json`  
**Last Updated**: Current session  
**System**: Livnium NLI v4 - Planet Architecture

