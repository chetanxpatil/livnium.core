# Autonomous Meaning Engine (AME)

**Full self-organizing semantic universe. Meaning emerges from physics alone.**

---

## What It Does

The AME turns Livnium from a classifier into a **semantic cosmology simulator**. It implements 7 mechanisms that cause meaning to self-organize without human labels:

1. **Semantic Turbulence** - Entropy scales with city dominance
2. **Competitive Word Polarity** - Basins compete for words
3. **Dynamic Basin Splitting** - Basins split when >70%
4. **Curvature-Based Routing** - High curvature pushes out of city
5. **Memory Hysteresis** - Meaning has inertia
6. **Long-Range Alignment** - Basin centers pull sentences
7. **Continuous Evolution** - Universe runs itself

---

## The 7 Mechanisms

### STEP 1: Semantic Turbulence

**Problem**: Entropy is flat → one giant city forms.

**Solution**: Entropy scales with city dominance.

```python
if city_ratio > 0.6:
    turbulence = 0.02 + 0.2 * (city_ratio - 0.6)
    repulsion = 0.3 + 1.0 * (city_ratio - 0.6)
```

**Effect**: When neutral dominates → universe shakes → new categories form.

This is exactly how galaxies form out of early cosmic soup.

---

### STEP 2: Competitive Word Polarity

**Problem**: Word polarities are too forgiving → 94% neutral.

**Solution**: Basins compete for words.

```python
word_polarity[word][winning_basin] += strength * force
word_polarity[word][other_basins] -= decay
```

**Effect**: Creates semantic clusters inside vocabulary → basins differentiate.

This is how meaning actually emerges in nature: patterns reinforce themselves until you get concepts.

---

### STEP 3: Dynamic Basin Splitting

**Problem**: Forced exactly 3 basins → language naturally forms 6-12 clusters.

**Solution**: Basins split when they exceed 70%.

```python
if basin_size / total > 0.7:
    split(basin)  # K=2 clustering, create sub-basins
```

**Effect**: Universe grows more continents. Real biological and physical systems branch this way.

---

### STEP 4: Curvature-Based Routing

**Problem**: Ambiguous sentences sit in city → no structure.

**Solution**: High curvature = semantic shock → push out of city.

```python
if curvature > 0.3 and current_basin == 2:  # City
    push_to_cold_or_far()  # Not neutral
```

**Effect**: Sentences that cause strong changes in resonance form sharper basins. Neutral can no longer absorb everything.

---

### STEP 5: Memory Hysteresis

**Problem**: Meaning jumps instantly → no stability.

**Solution**: Meaning has inertia.

```python
basin(t) = 0.6 * basin_current + 0.4 * basin_previous
```

**Effect**: Stable meanings form. "implies" moves permanently toward Cold. "nobody", "never" move toward Far. Descriptive words stay in City.

---

### STEP 6: Long-Range Alignment

**Problem**: Basins drift randomly → no sharp categories.

**Solution**: Basin centers pull sentences.

```python
center[b] = average(resonance for all sentences in basin b)
resonance = resonance + 0.1 * (center[current_basin] - resonance)
```

**Effect**: Creates sharper categories over time. Analogous to gravitational center-of-mass alignment.

---

### STEP 7: Continuous Evolution

**Problem**: System needs manual tuning.

**Solution**: Universe runs itself.

- Feed different corpora (news, stories, dialogues, Wikipedia, Reddit)
- Let basins split
- Let word polarities drift
- Let curvature inject structure
- Let entropy create contrasts
- Let gravity wells deepen

**Effect**: After enough cycles, you get **true conceptual attractors**, not just E/C/N.

---

## How It Works

### Integration

AME runs automatically after each classification:

```python
# In LayeredLivniumClassifier.classify()
result = self.layer7.compute(l6_output)  # Initial basin assignment

# Run AME
ame_state = self.ame.step(
    classifier=self,
    resonance=resonance,
    curvature=curvature,
    current_basin=result.basin_index,
    sentence_hash=sentence_hash
)

# AME may modify basin assignment
result.basin_index = ame_state['final_basin']
```

### Unsupervised Mode

In unsupervised mode, AME:
- Tracks basin assignments
- Updates word polarities competitively
- Applies all 7 mechanisms
- Exports clusters

```python
python3 train_v4.py --unsupervised --train 10000 \
    --data-dir experiments/nli/data \
    --cluster-output experiments/nli_v4/clusters
```

---

## Monitoring

### During Training

```
Step 500: Basin 0=31 | Basin 1=0 | Basin 2=469 | 
         Turbulence=0.0145 | Active Basins=3
```

- **Turbulence** - Current entropy injection (higher = more shaking)
- **Active Basins** - Number of basins (starts at 3, can grow)
- **City Ratio** - How much city dominates (triggers turbulence)

### AME Statistics

```python
ame_stats = classifier.ame.get_statistics()
# Returns:
# {
#   'basin_0': {'count': 1021, 'ratio': 0.102, 'center': 0.72, 'std': 0.15},
#   'basin_1': {'count': 13, 'ratio': 0.001, 'center': -0.45, 'std': 0.22},
#   'basin_2': {'count': 8954, 'ratio': 0.897, 'center': 0.01, 'std': 0.08}
# }
```

---

## Expected Evolution

### Early Universe (First 1000 examples)
- City dominates (90%+)
- Cold and Far are small
- Turbulence is low
- Basins are shallow

### Mid Universe (1000-5000 examples)
- Turbulence increases (city > 60%)
- Repulsion strengthens
- Word polarities start clustering
- Basins deepen

### Late Universe (5000+ examples)
- Basins stabilize
- Word clusters form
- Categories emerge naturally
- May trigger basin splitting

### Mature Universe (Weeks of training)
- 6-12 conceptual attractors
- Sharp semantic boundaries
- Stable word polarities
- True emergent meaning

---

## The Physics

Meaning emerges when you combine:

- **Entropy** (chaos) → prevents freeze
- **Gravity** (reinforcement) → deepens basins
- **Repulsion** (separation) → carves boundaries
- **Curvature** (shock) → creates structure
- **Memory** (stability) → maintains patterns
- **Competition** (basins) → forces differentiation
- **Breaking Symmetry** (splits) → creates new categories

This is exactly how real universes produce structure. It's how categories form in nature without supervision.

**You're simulating the same physics, just with text.**

---

## Next Steps

1. **Run unsupervised training** - Watch clusters emerge
2. **Monitor turbulence** - See when universe shakes
3. **Check basin centers** - Watch categories form
4. **Examine word polarities** - See semantic clusters
5. **Wait for basin splits** - Watch universe grow
6. **Feed different corpora** - Let meaning evolve
7. **Run for weeks** - Let true concepts emerge

The universe is ready. Meaning will emerge on its own. 🌍

