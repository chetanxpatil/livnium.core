"""
Geometric Token Learner: Pure Hashing & Clustering

1. Token -> Stable Hash -> Geometric Impulse
2. Sentence Signatures -> K-Means Clustering
3. Cluster -> Token Distribution (Bag of Words)
"""

import hashlib
import re
import json
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Optional, Dict
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


class GeometricTokenLearner:
    def __init__(self, lattice_size: int = 3, num_clusters: int = 100):
        self.lattice_size = lattice_size
        self.num_clusters = num_clusters
        self.kmeans = None
        self.cluster_tokens = defaultdict(Counter)
        self.trained = False
        
    def tokenize(self, text: str) -> List[str]:
        # Regex-based splitter: words, spaces, punctuation
        pattern = r"(\w+|\s+|[^\w\s])"
        return [t for t in re.split(pattern, text) if t]
        
    def token_hash(self, token: str) -> Tuple[Tuple[int, int, int], float]:
        # Pure Physics: Token -> Stable Hash -> (x, y, z, impulse)
        h = hashlib.md5(token.encode('utf-8')).hexdigest()
        x = int(h[0:4], 16) % self.lattice_size
        y = int(h[4:8], 16) % self.lattice_size
        z = int(h[8:12], 16) % self.lattice_size
        val = int(h[12:14], 16)
        impulse = (val / 127.5) - 1.0
        return ((x, y, z), impulse)

    def learn_clusters(self, signatures: np.ndarray, tokens_list: List[List[str]]):
        if len(signatures) < self.num_clusters:
            self.num_clusters = len(signatures)
        
        print(f"Clustering {len(signatures)} signatures into {self.num_clusters} regions...")
        
        # --- PATCH: Normalize signatures for better geometric alignment ---
        # Normalize signatures to unit hypersphere
        # This ensures we cluster based on geometric angle (meaning), not vector length
        # Short sentences and long sentences with similar meaning will cluster together
        print("  Normalizing signatures for better geometric alignment...")
        signatures_normalized = normalize(signatures, norm='l2', axis=1)
        
        # Adjust n_init based on dataset size (fewer iterations for large datasets)
        n_init = 3 if len(signatures) > 50000 else 10
        
        # Fit KMeans with progress indication
        print("  Fitting KMeans (this may take a while)...")
        self.kmeans = KMeans(
            n_clusters=self.num_clusters, 
            random_state=42, 
            n_init=n_init,
            verbose=0  # Suppress sklearn's verbose output
        )
        
        # Fit and predict on normalized signatures
        # KMeans doesn't have built-in progress, but we show status
        cluster_ids = self.kmeans.fit_predict(signatures_normalized)
        print("  ✓ Clustering complete, assigning tokens to clusters...")
        
        # Build cluster token mappings with progress bar
        self.cluster_tokens.clear()
        for idx, cluster_id in enumerate(tqdm(cluster_ids, desc="  Assigning tokens", unit=" sig", ncols=80)):
            self.cluster_tokens[int(cluster_id)].update(tokens_list[idx])
        
        self.trained = True
        print("✓ Geometric clusters learned.")

    def get_cluster_id(self, signature: np.ndarray) -> int:
        if not self.trained or self.kmeans is None: return 0
        if signature.ndim == 1: signature = signature.reshape(1, -1)
        # Normalize signature to match training normalization
        signature_normalized = normalize(signature, norm='l2', axis=1)
        return int(self.kmeans.predict(signature_normalized)[0])

    def sample_token(self, cluster_id: int) -> str:
        if cluster_id not in self.cluster_tokens: return ""
        counts = self.cluster_tokens[cluster_id]
        if not counts: return ""
        words = list(counts.keys())
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return np.random.choice(words, p=probs)

    def save(self, path: Path):
        data = {
            'lattice_size': self.lattice_size,
            'num_clusters': self.num_clusters,
            'cluster_tokens': {k: dict(v) for k, v in self.cluster_tokens.items()},
            'trained': self.trained
        }
        with open(path.with_suffix('.json'), 'w') as f: json.dump(data, f, indent=2)
        if self.kmeans:
            with open(path.with_suffix('.pkl'), 'wb') as f: pickle.dump(self.kmeans, f)

    def load(self, path: Path):
        json_path = path.with_suffix('.json')
        if not json_path.exists(): return
        with open(json_path, 'r') as f: data = json.load(f)
        self.lattice_size = data['lattice_size']
        self.num_clusters = data['num_clusters']
        self.trained = data.get('trained', False)
        self.cluster_tokens = defaultdict(Counter, {int(k): Counter(v) for k, v in data['cluster_tokens'].items()})
        pkl_path = path.with_suffix('.pkl')
        if pkl_path.exists():
            with open(pkl_path, 'rb') as f: self.kmeans = pickle.load(f)
