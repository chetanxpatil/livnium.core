# Quick Start: Running Unsupervised Training

## Basic Command

From the project root directory:

```bash
python3 experiments/nli_v4/train_v4.py --unsupervised --train 10000 \
    --data-dir experiments/nli/data \
    --cluster-output experiments/nli_v4/clusters
```

## Options

### Unsupervised Mode (Geometry Discovers Meaning)
```bash
--unsupervised
```
- No labels needed
- Tracks clusters instead of accuracy
- Updates word polarities based on basin assignment

### Number of Training Examples
```bash
--train 10000    # Train on 10,000 examples
--train 1000     # Train on 1,000 examples (faster)
--train 50000    # Train on 50,000 examples (slower, better clusters)
```

### Data Directory
```bash
--data-dir experiments/nli/data
```
Path to directory containing SNLI data files:
- `snli_1.0_train.jsonl`
- `snli_1.0_test.jsonl`
- `snli_1.0_dev.jsonl`

### Cluster Output Directory
```bash
--cluster-output experiments/nli_v4/clusters
```
Directory where cluster JSON files will be saved:
- `cluster_0_cold.json`
- `cluster_1_far.json`
- `cluster_2_city.json`
- `cluster_summary.json`

### Clean Start
```bash
--clean
```
Start with fresh lexicon and reset basin depths.

## Examples

### Quick Test (1,000 examples)
```bash
python3 experiments/nli_v4/train_v4.py --unsupervised --train 1000 \
    --data-dir experiments/nli/data \
    --cluster-output experiments/nli_v4/clusters
```

### Full Training (10,000 examples)
```bash
python3 experiments/nli_v4/train_v4.py --unsupervised --train 10000 \
    --data-dir experiments/nli/data \
    --cluster-output experiments/nli_v4/clusters
```

### Clean Start + Full Training
```bash
python3 experiments/nli_v4/train_v4.py --unsupervised --train 10000 \
    --data-dir experiments/nli/data \
    --cluster-output experiments/nli_v4/clusters \
    --clean
```

### Supervised Mode (with labels)
```bash
python3 experiments/nli_v4/train_v4.py --train 10000 \
    --data-dir experiments/nli/data
```
(No `--unsupervised` flag = supervised mode)

## What You'll See

### During Training
```
Step 500: Basin 0=31 | Basin 1=0 | Basin 2=469 | Moksha=0.000 | Entropy=0.0100 | Imbalance=0.000 | Temp=0.000
```

- **Basin 0/1/2** = Count of sentences in each cluster
- **Moksha** = Convergence rate
- **Entropy** = Current thermal noise
- **Imbalance** = Class distribution imbalance
- **Temp** = System temperature

### After Training
```
GEOMETRY-DISCOVERED CLUSTERS
======================================================================

BASIN_0_COLD:
  Count: 1021
  Avg Confidence: 0.7439
  Description: Cold basin (entailment-like patterns)

BASIN_1_FAR:
  Count: 13
  Avg Confidence: 0.6000
  Description: Far basin (contradiction-like patterns)

BASIN_2_CITY:
  Count: 8954
  Avg Confidence: 0.7199
  Description: City basin (neutral-like patterns)
```

## Output Files

After training, check the cluster directory:

```bash
ls experiments/nli_v4/clusters/
```

You'll see:
- `cluster_0_cold.json` - Cold basin entries
- `cluster_1_far.json` - Far basin entries
- `cluster_2_city.json` - City basin entries
- `cluster_summary.json` - Statistics

## Viewing Clusters

```bash
# View summary
cat experiments/nli_v4/clusters/cluster_summary.json | python3 -m json.tool

# View a specific cluster (first 10 entries)
cat experiments/nli_v4/clusters/cluster_0_cold.json | python3 -m json.tool | head -50
```

## Troubleshooting

### "No such file or directory"
Make sure you're in the project root:
```bash
cd /Users/chetanpatil/Desktop/clean-nova-livnium
```

### "Data file not found"
Check that SNLI data files exist:
```bash
ls experiments/nli/data/snli_1.0_train.jsonl
```

### Virtual Environment
If you have a virtual environment:
```bash
source .venv/bin/activate
python3 experiments/nli_v4/train_v4.py --unsupervised --train 10000 ...
```

## Tips

1. **Start small**: Use `--train 1000` first to test
2. **Check clusters**: Look at the JSON files to see what patterns emerged
3. **Monitor entropy**: High entropy = system is exploring
4. **Watch basin distribution**: Should be somewhat balanced (not 99% in one basin)

