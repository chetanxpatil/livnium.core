"""
Train Text-to-Geometry System on DailyDialogue Dataset

This script:
1. Loads dialogues from DailyDialogue test.csv
2. Extracts sentences from each dialogue
3. Injects them into geometry and extracts meaning signatures
4. Builds a signature database for similarity search
5. Learns patterns in the geometric space
"""

import sys
import csv
import json
import ast
import ssl
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# Add project root to path (go up two levels: training -> nova -> project_root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nova.core.text_to_geometry import TextToGeometry
from nova.core.geometric_token_learner import GeometricTokenLearner


def parse_dialogue(dialog_str: str) -> List[str]:
    """
    Parse dialogue string into list of sentences.
    
    Handles multiple formats:
    - Format 1: "['sentence1' 'sentence2' ...]"
    - Format 2: "'sentence1' 'sentence2' 'sentence3'"
    - Format 3: "sentence1 sentence2 sentence3"
    """
    if not dialog_str or not dialog_str.strip():
        return []
    
    dialog_str = dialog_str.strip()
    sentences = []
    
    # Try format 1: Python list format "['sentence1' 'sentence2' ...]"
    try:
        # Remove outer quotes if present
        if dialog_str.startswith('"') and dialog_str.endswith('"'):
            dialog_str = dialog_str[1:-1]
        
        # Try parsing as Python list
        dialog_list = ast.literal_eval(dialog_str)
        
        if isinstance(dialog_list, list):
            for item in dialog_list:
                if isinstance(item, str) and item.strip():
                    sentences.append(item.strip())
            if sentences:
                return sentences
    except:
        pass
    
    # Try format 2: Single-quoted sentences "'sentence1' 'sentence2'"
    import re
    matches = re.findall(r"'([^']+)'", dialog_str)
    if matches:
        sentences = [m.strip() for m in matches if m.strip()]
        if sentences:
            return sentences
    
    # Try format 3: Space-separated (fallback)
    # Split by common sentence boundaries
    parts = re.split(r'[.!?]\s+', dialog_str)
    sentences = [p.strip() for p in parts if p.strip() and len(p.strip()) > 3]
    
    # If still no sentences, return the whole string as one sentence
    if not sentences and dialog_str:
        sentences = [dialog_str]
    
    return sentences


def load_cornell_movie_dataset(max_dialogues: Optional[int] = None, 
                               corpus_path: Optional[Path] = None) -> List[Dict]:
    """
    Load Cornell Movie-Dialogs Corpus using ConvoKit.
    
    Dataset: 220,579 conversational exchanges between 10,292 pairs of movie characters
    in 617 movies. 304,713 utterances, 83,097 conversations.
    
    Args:
        max_dialogues: Maximum number of conversations to load
        corpus_path: Optional path to existing corpus (if already downloaded)
    
    Returns:
        List of dialogue dictionaries with 'sentences' key
    """
    try:
        from convokit import Corpus, download
        import ssl
        import urllib.request
    except ImportError as e:
        print("❌ ConvoKit not installed or missing dependencies!")
        print(f"   Error: {e}")
        print()
        print("   To install ConvoKit (without xformers):")
        print("   pip install convokit --no-deps")
        print("   pip install pandas numpy scipy scikit-learn nltk spacy dill msgpack-numpy")
        print()
        print("   Or install with minimal dependencies:")
        print("   pip install convokit")
        print("   (xformers build failure is OK - it's optional)")
        return []
    
    print("Loading Cornell Movie-Dialogs Corpus...")
    
    # Check default corpus location first
    default_corpus_path = Path.home() / ".convokit" / "saved-corpora" / "movie-corpus"
    
    # Load corpus
    if corpus_path and corpus_path.exists():
        print(f"Loading from existing corpus: {corpus_path}")
        corpus = Corpus(filename=str(corpus_path))
    elif default_corpus_path.exists():
        print(f"Loading from cached corpus: {default_corpus_path}")
        corpus = Corpus(filename=str(default_corpus_path))
    else:
        print("Downloading Cornell Movie-Dialogs Corpus (this may take a while)...")
        try:
            # Try normal download first
            corpus = Corpus(filename=download("movie-corpus"))
            print("✓ Download complete")
        except (urllib.error.URLError, ssl.SSLError) as e:
            # SSL certificate issue - try with unverified context
            print(f"⚠ SSL certificate issue: {e}")
            print("   Attempting download with SSL verification disabled...")
            print("   (This is safe for public datasets)")
            
            # Create unverified SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Monkey-patch urllib to use unverified context
            original_urlopen = urllib.request.urlopen
            def urlopen_with_ssl_fix(*args, **kwargs):
                if 'context' not in kwargs:
                    kwargs['context'] = ssl_context
                return original_urlopen(*args, **kwargs)
            urllib.request.urlopen = urlopen_with_ssl_fix
            
            try:
                corpus = Corpus(filename=download("movie-corpus"))
                print("✓ Download complete")
            finally:
                # Restore original
                urllib.request.urlopen = original_urlopen
    
    print(f"Corpus stats:")
    print(f"  Speakers: {len(corpus.speakers)}")
    print(f"  Utterances: {len(corpus.utterances)}")
    print(f"  Conversations: {len(corpus.conversations)}")
    print()
    
    # Extract dialogues
    dialogues = []
    conversation_ids = list(corpus.conversations.keys())
    
    if max_dialogues:
        conversation_ids = conversation_ids[:max_dialogues]
    
    print(f"Extracting {len(conversation_ids)} conversations...")
    
    for i, conv_id in enumerate(conversation_ids):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(conversation_ids)} conversations...")
        
        conversation = corpus.get_conversation(conv_id)
        utterances = conversation.get_utterance_ids()
        
        # Extract sentences from utterances
        sentences = []
        for utt_id in utterances:
            utterance = corpus.get_utterance(utt_id)
            text = utterance.text.strip()
            if text and len(text) > 3:  # Filter very short utterances
                sentences.append(text)
        
        if len(sentences) >= 2:  # Only include dialogues with at least 2 turns
            dialogues.append({
                'id': i,
                'conversation_id': conv_id,
                'sentences': sentences,
                'num_turns': len(sentences),
                'movie': conversation.meta.get('movie_name', 'Unknown'),
                'movie_idx': conversation.meta.get('movie_idx', None)
            })
    
    print(f"✓ Loaded {len(dialogues)} dialogues")
    print(f"Total sentences: {sum(len(d['sentences']) for d in dialogues)}")
    return dialogues


def load_empathetic_dialogues_dataset(max_dialogues: Optional[int] = None, 
                                      corpus_path: Optional[Path] = None) -> List[Dict]:
    """
    Load EmpatheticDialogues dataset from Convokit.
    
    Args:
        max_dialogues: Maximum number of dialogues to load
        corpus_path: Optional path to existing corpus
    
    Returns:
        List of dialogue dictionaries
    """
    try:
        from convokit import Corpus, download
    except ImportError as e:
        print("❌ ConvoKit not installed or missing dependencies!")
        print(f"   Error: {e}")
        print()
        print("   To install ConvoKit (without xformers):")
        print("   pip install convokit --no-deps")
        print("   pip install pandas numpy scipy scikit-learn nltk spacy dill msgpack-numpy")
        print()
        print("   Or install with minimal dependencies:")
        print("   pip install convokit")
        print("   (xformers build failure is OK - it's optional)")
        return []
    
    print("Loading EmpatheticDialogues...")
    
    # Check default corpus location first
    # ConvoKit may use different naming conventions
    possible_paths = [
        Path.home() / ".convokit" / "saved-corpora" / "empathetic-dialogues",
        Path.home() / ".convokit" / "saved-corpora" / "empathetic_dialogues",
        Path.home() / ".convokit" / "saved-corpora" / "empathy",
    ]
    
    # Load corpus
    if corpus_path and corpus_path.exists():
        print(f"Loading from existing corpus: {corpus_path}")
        corpus = Corpus(filename=str(corpus_path))
    else:
        # Try cached paths
        cached_path = None
        for path in possible_paths:
            if path.exists():
                cached_path = path
                break
        
        if cached_path:
            print(f"Loading from cached corpus: {cached_path}")
            corpus = Corpus(filename=str(cached_path))
        else:
            print("Downloading EmpatheticDialogues (this may take a while)...")
            print("⚠ Note: ConvoKit may not have this dataset pre-configured.")
            print("   Trying alternative dataset names...")
            
            # Try different possible dataset names
            dataset_names = ["empathetic-dialogues", "empathetic_dialogues", "empathy", "empathetic"]
            corpus = None
            
            for dataset_name in dataset_names:
                try:
                    print(f"  Trying '{dataset_name}'...")
                    corpus = Corpus(filename=download(dataset_name))
                    print(f"✓ Download complete using '{dataset_name}'")
                    break
                except KeyError:
                    continue
                except (urllib.error.URLError, ssl.SSLError) as e:
                    # SSL certificate issue - try with unverified context
                    print(f"⚠ SSL certificate issue: {e}")
                    print("   Attempting download with SSL verification disabled...")
                    print("   (This is safe for public datasets)")
                    
                    # Create unverified SSL context
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE
                    
                    # Monkey-patch urllib to use unverified context
                    original_urlopen = urllib.request.urlopen
                    def urlopen_with_ssl_fix(*args, **kwargs):
                        if 'context' not in kwargs:
                            kwargs['context'] = ssl_context
                        return original_urlopen(*args, **kwargs)
                    urllib.request.urlopen = urlopen_with_ssl_fix
                    
                    try:
                        corpus = Corpus(filename=download(dataset_name))
                        print(f"✓ Download complete using '{dataset_name}'")
                        break
                    except KeyError:
                        continue
                    finally:
                        # Restore original
                        urllib.request.urlopen = original_urlopen
            
            if corpus is None:
                print("❌ Could not download EmpatheticDialogues from ConvoKit.")
                print("   This dataset may not be available in ConvoKit's default repository.")
                print("   Please download it manually and provide --corpus-path")
                print("   Or use a different dataset: --dataset cornell or --dataset nova")
                return []
    
    print(f"Corpus stats:")
    print(f"  Speakers: {len(corpus.speakers)}")
    print(f"  Utterances: {len(corpus.utterances)}")
    print(f"  Conversations: {len(corpus.conversations)}")
    print()
    
    # Extract dialogues
    dialogues = []
    conversation_ids = list(corpus.conversations.keys())
    
    if max_dialogues:
        conversation_ids = conversation_ids[:max_dialogues]
    
    print(f"Extracting {len(conversation_ids)} conversations...")
    
    for i, conv_id in enumerate(conversation_ids):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(conversation_ids)} conversations...")
        
        conversation = corpus.get_conversation(conv_id)
        utterances = conversation.get_utterance_ids()
        
        # Extract sentences from utterances
        sentences = []
        for utt_id in utterances:
            utterance = corpus.get_utterance(utt_id)
            text = utterance.text.strip()
            if text and len(text) > 3:  # Filter very short utterances
                sentences.append(text)
        
        # Get emotion/situation metadata if available
        emotion = conversation.meta.get('emotion', None)
        situation = conversation.meta.get('situation', None)
        
        if len(sentences) >= 2:  # Only include dialogues with at least 2 turns
            dialogues.append({
                'id': i,
                'conversation_id': conv_id,
                'sentences': sentences,
                'num_turns': len(sentences),
                'emotion': emotion,
                'situation': situation
            })
    
    print(f"✓ Loaded {len(dialogues)} dialogues")
    print(f"Total sentences: {sum(len(d['sentences']) for d in dialogues)}")
    return dialogues


def load_dailydialogue_dataset(csv_path: Path, max_dialogues: Optional[int] = None) -> List[Dict]:
    """
    Load DailyDialogue dataset from CSV.
    
    Returns:
        List of dialogue dictionaries with 'dialog', 'act', 'emotion' keys
    """
    dialogues = []
    
    print(f"Loading DailyDialogue dataset from {csv_path}...")
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_dialogues and i >= max_dialogues:
                break
            
            dialog_str = row.get('dialog', '')
            act_str = row.get('act', '')
            emotion_str = row.get('emotion', '')
            
            # Parse dialogue into sentences
            sentences = parse_dialogue(dialog_str)
            
            # Parse act and emotion (they're space-separated lists)
            try:
                acts = [int(x) for x in act_str.strip('[]').split()] if act_str else []
                emotions = [int(x) for x in emotion_str.strip('[]').split()] if emotion_str else []
            except:
                acts = []
                emotions = []
            
            dialogues.append({
                'id': i,
                'dialog': dialog_str,
                'sentences': sentences,
                'acts': acts,
                'emotions': emotions,
                'num_turns': len(sentences)
            })
    
    print(f"Loaded {len(dialogues)} dialogues")
    print(f"Total sentences: {sum(len(d['sentences']) for d in dialogues)}")
    return dialogues


def extract_signatures(dialogues: List[Dict], 
                      interface,
                      collapse_steps: int = 15,
                      verbose: bool = False) -> Tuple[List[Dict], List[np.ndarray], List[List[str]]]:
    """
    Extract geometric signatures for all sentences in dialogues.
    
    Returns:
        Tuple of (signature dictionaries, signature arrays, token lists)
    """
    signatures = []
    signature_arrays = []
    token_lists = []
    
    print(f"\nExtracting geometric signatures...")
    print(f"  Collapse steps: {collapse_steps}")
    print()
    
    for dialogue in tqdm(dialogues, desc="Processing dialogues"):
        dialogue_id = dialogue['id']
        sentences = dialogue['sentences']
        
        for turn_idx, sentence in enumerate(sentences):
            # NO FILTERING - include all sentences (even very short ones)
            if len(sentence.strip()) < 1:  # Only skip completely empty
                continue
            
            try:
                # Tokenize sentence
                tokens = interface.learner.tokenize(sentence)
                if not tokens:
                    if verbose and len(signatures) < 5:
                        print(f"  Warning: No tokens for sentence: '{sentence[:50]}...'")
                    continue
                
                # Extract signature (non-recursive only)
                signature = interface.get_meaning_signature(sentence, collapse_steps=collapse_steps)
                
                # Get metadata (handle both DailyDialogue and Cornell formats)
                act = None
                emotion = None
                if 'acts' in dialogue and turn_idx < len(dialogue['acts']):
                    act = dialogue['acts'][turn_idx]
                if 'emotions' in dialogue and turn_idx < len(dialogue['emotions']):
                    emotion = dialogue['emotions'][turn_idx]
                
                signatures.append({
                    'dialogue_id': dialogue_id,
                    'turn_idx': turn_idx,
                    'sentence': sentence,
                    'signature': signature.tolist(),  # Convert to list for JSON
                    'signature_stats': {
                        'mean': float(np.mean(signature)),
                        'std': float(np.std(signature)),
                        'sum': float(np.sum(signature)),
                        'min': float(np.min(signature)),
                        'max': float(np.max(signature))
                    },
                    'act': act,
                    'emotion': emotion,
                    'length': len(sentence)
                })
                
                # Store for clustering
                signature_arrays.append(signature)
                token_lists.append(tokens)
                
                # Reset geometry for next sentence
                interface.reset_geometry()
                
            except Exception as e:
                if verbose or len(signatures) < 5:
                    print(f"Error processing sentence '{sentence[:50]}...': {e}")
                    import traceback
                    traceback.print_exc()
                continue
    
    print(f"\n✓ Extracted {len(signatures)} signatures")
    return signatures, np.array(signature_arrays), token_lists


def extract_learned_patterns(signatures: List[Dict], output_path: Path):
    """
    Extract and save ONLY learned patterns (no original sentences).
    
    Like NLI's GlobalLexicon - only saves learned patterns:
    - word_patterns: signature region -> words
    - word_sequences: learned n-grams
    - vocabulary: all learned words
    """
    print(f"\nExtracting learned patterns (no sentences stored)...")
    
    from collections import defaultdict, Counter
    
    word_patterns = {}  # signature region -> common words
    word_sequences = defaultdict(list)  # learned word sequences
    vocabulary = set()
    
    # Learn patterns from signatures (same as SentenceDecoder.learn_from_data)
    for sig_data in signatures:
        sentence = sig_data['sentence']
        signature = np.array(sig_data['signature'])
        
        # Extract words
        words = sentence.lower().split()
        vocabulary.update(words)
        
        # Learn: which signature regions correspond to which words
        num_regions = min(10, len(signature))
        region_size = len(signature) // num_regions
        
        for i, word in enumerate(words):
            # NO FILTERING - include all words (even short ones)
            # Map word to signature region
            region_idx = min(i % num_regions, num_regions - 1)
            region_start = region_idx * region_size
            region_end = region_start + region_size
            region_sig = signature[region_start:region_end]
            
            # Store word with its signature region pattern
            region_key = tuple(np.round(region_sig, 2))
            if region_key not in word_patterns:
                word_patterns[region_key] = []
            word_patterns[region_key].append(word)
        
        # Learn sentence structures (word patterns) - NO MINIMUM LENGTH
        if len(words) >= 1:  # Changed from 3 to 1 - include all sequences
            # Store 2-word and 3-word sequences
            for i in range(len(words) - 1):
                if i + 1 < len(words):
                    seq = tuple(words[i:i+2])
                    word_sequences[words[i]].append(seq)
                if i + 2 < len(words):
                    seq = tuple(words[i:i+3])
                    word_sequences[words[i]].append(seq)
    
    # NO FILTERING - keep ALL words per pattern (not just top 5)
    # Just deduplicate while preserving order
    for region_key in word_patterns:
        words = word_patterns[region_key]
        # Remove duplicates but keep all words (not just top 5)
        seen = set()
        unique_words = []
        for word in words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)
        word_patterns[region_key] = unique_words
    
    # Convert to JSON-serializable format
    # Convert tuple keys to lists for JSON (convert numpy types to plain floats)
    word_patterns_json = {}
    for key, words in word_patterns.items():
        # Convert tuple to list of plain floats (not numpy types)
        key_list = [float(x) for x in key]
        word_patterns_json[str(key_list)] = words
    
    # Convert word_sequences to lists
    word_sequences_json = {}
    for word, seqs in word_sequences.items():
        word_sequences_json[word] = [list(seq) for seq in seqs]
    
    # Save only learned patterns (NO sentences)
    patterns = {
        'word_patterns': word_patterns_json,  # signature region -> words
        'word_sequences': word_sequences_json,  # learned n-grams
        'vocabulary': list(vocabulary),  # all learned words
        'metadata': {
            'total_patterns': len(word_patterns),
            'total_sequences': sum(len(seqs) for seqs in word_sequences.values()),
            'vocabulary_size': len(vocabulary),
            'training_signatures': len(signatures)
        }
    }
    
    # Save to JSON
    print(f"Saving learned patterns to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=2)
    
    print(f"✓ Patterns saved: {len(word_patterns)} word patterns, {len(vocabulary)} words")
    return patterns


def analyze_patterns(signatures: List[Dict], output_path: Optional[Path] = None):
    """
    Analyze patterns in the geometric signatures.
    """
    print(f"\nAnalyzing patterns...")
    
    # Group by act
    act_groups = defaultdict(list)
    for sig in signatures:
        if sig['act'] is not None:
            act_groups[sig['act']].append(sig)
    
    # Group by emotion
    emotion_groups = defaultdict(list)
    for sig in signatures:
        if sig['emotion'] is not None:
            emotion_groups[sig['emotion']].append(sig)
    
    # Compute statistics
    patterns = {
        'by_act': {},
        'by_emotion': {},
        'overall': {
            'mean_signature': np.mean([sig['signature'] for sig in signatures], axis=0).tolist(),
            'std_signature': np.std([sig['signature'] for sig in signatures], axis=0).tolist()
        }
    }
    
    # Act patterns
    for act, sigs in act_groups.items():
        act_signatures = np.array([sig['signature'] for sig in sigs])
        patterns['by_act'][act] = {
            'count': len(sigs),
            'mean_signature': np.mean(act_signatures, axis=0).tolist(),
            'std_signature': np.std(act_signatures, axis=0).tolist()
        }
    
    # Emotion patterns
    for emotion, sigs in emotion_groups.items():
        emotion_signatures = np.array([sig['signature'] for sig in sigs])
        patterns['by_emotion'][emotion] = {
            'count': len(sigs),
            'mean_signature': np.mean(emotion_signatures, axis=0).tolist(),
            'std_signature': np.std(emotion_signatures, axis=0).tolist()
        }
    
    # Print summary
    print(f"\nPattern Summary:")
    print(f"  Total signatures: {len(signatures)}")
    print(f"  Unique acts: {len(act_groups)}")
    print(f"  Unique emotions: {len(emotion_groups)}")
    print(f"\nAct distribution:")
    for act in sorted(act_groups.keys()):
        print(f"    Act {act}: {len(act_groups[act])} sentences")
    print(f"\nEmotion distribution:")
    for emotion in sorted(emotion_groups.keys()):
        print(f"    Emotion {emotion}: {len(emotion_groups[emotion])} sentences")
    
    # Save patterns
    if output_path:
        print(f"\nSaving patterns to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        print(f"✓ Patterns saved")
    
    return patterns


def train_model(dialogues: List[Dict],
                interface,
                collapse_steps: int = 15,
                output_dir: Path = Path("nova/model"),
                max_dialogues: Optional[int] = None,
                dataset_name: str = "Dataset",
                num_clusters: int = 50):
    """
    Main training function.
    """
    print("=" * 70)
    print(f"Training Text-to-Geometry System on {dataset_name}")
    print("=" * 70)
    print()
    
    # Limit dialogues if specified
    if max_dialogues:
        dialogues = dialogues[:max_dialogues]
        print(f"Processing first {max_dialogues} dialogues")
    
    # Extract signatures, signature arrays, and token lists
    signatures, signature_arrays, token_lists = extract_signatures(dialogues, interface, collapse_steps=collapse_steps)
    
    if not signatures:
        print("No signatures extracted. Exiting.")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract and save ONLY learned patterns (no sentences)
    patterns_path = output_dir / "learned_patterns.json"
    patterns = extract_learned_patterns(signatures, patterns_path)
    
    # NEW: Learn geometric tokens using MD5-based clustering
    print("\n" + "=" * 70)
    print("Learning Geometric Tokens (MD5 Hash-Based Clustering)")
    print("=" * 70)
    print()
    
    # Use the learner from the interface
    # Get lattice size (non-recursive only)
    lattice_size = interface.geometry.lattice_size
    
    # Update learner to match interface's lattice size
    interface.learner.lattice_size = lattice_size
    interface.learner.num_clusters = num_clusters
    
    # Learn clusters from signatures and tokens
    interface.learner.learn_clusters(signature_arrays, token_lists)
    
    # Save clusters
    cluster_path = output_dir / "geometric_clusters"
    interface.learner.save(cluster_path)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  Learned patterns: {patterns_path}")
    print(f"  Geometric clusters: {cluster_path.with_suffix('.json')}")
    print(f"  KMeans model: {cluster_path.with_suffix('.pkl')}")
    print(f"\nTotal patterns learned: {patterns['metadata']['total_patterns']}")
    print(f"Vocabulary size: {patterns['metadata']['vocabulary_size']}")
    print(f"Geometric clusters: {interface.learner.num_clusters}")
    print(f"\n✓ No sentences stored - only learned patterns and clusters")
    print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train text-to-geometry on dialogue datasets")
    parser.add_argument("--dataset", type=str, choices=["nova", "cornell", "empathetic"], 
                       default="nova",
                       help="Dataset to use: 'nova', 'cornell' (Movie-Dialogs), or 'empathetic' (EmpatheticDialogues)")
    parser.add_argument("--csv", type=str, default="nova/data/daily_test.csv",
                       help="Path to CSV file (for nova dataset or any CSV in DailyDialogue format)")
    parser.add_argument("--corpus-path", type=str, default=None,
                       help="Path to existing Cornell corpus (optional, will download if not provided)")
    parser.add_argument("--max-dialogues", type=int, default=None,
                       help="Maximum number of dialogues to process")
    parser.add_argument("--collapse-steps", type=int, default=15,
                       help="Number of collapse steps per sentence")
    parser.add_argument("--lattice-size", type=int, default=3,
                       help="Geometry lattice size (3, 5, 7, ...)")
    parser.add_argument("--impulse-scale", type=float, default=0.1,
                       help="Character impulse scale")
    parser.add_argument("--output-dir", type=str, default="nova/model",
                       help="Output directory for models")
    parser.add_argument("--num-clusters", type=int, default=50,
                       help="Number of geometric clusters (K)")
    
    args = parser.parse_args()
    
    # Load dataset based on choice
    if args.dataset == "cornell":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        dialogues = load_cornell_movie_dataset(
            max_dialogues=args.max_dialogues,
            corpus_path=corpus_path
        )
        dataset_name = "Cornell Movie-Dialogs Corpus"
    elif args.dataset == "empathetic":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        dialogues = load_empathetic_dialogues_dataset(
            max_dialogues=args.max_dialogues,
            corpus_path=corpus_path
        )
        dataset_name = "EmpatheticDialogues"
    else:  # nova
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found: {csv_path}")
            return
        dialogues = load_dailydialogue_dataset(csv_path, max_dialogues=args.max_dialogues)
        dataset_name = "Nova"
    
    if not dialogues:
        print("No dialogues loaded. Exiting.")
        return
    
    # Create interface (non-recursive only)
    print("=" * 70)
    print("Using REGULAR Geometry (Flat Collapse)")
    print("=" * 70)
    print()
    
    interface = TextToGeometry(
        lattice_size=args.lattice_size,
        impulse_scale=args.impulse_scale,
        num_clusters=args.num_clusters
    )
    
    # Train
    train_model(
        dialogues=dialogues,
        interface=interface,
        collapse_steps=args.collapse_steps,
        output_dir=Path(args.output_dir),
        max_dialogues=args.max_dialogues,
        dataset_name=dataset_name,
        num_clusters=args.num_clusters
    )


if __name__ == "__main__":
    main()

