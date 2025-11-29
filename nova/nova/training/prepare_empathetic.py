"""
Prepare EmpatheticDialogues Dataset for Livnium Training

Downloads from Hugging Face Datasets Hub or provides manual download instructions.
"""

import csv
import json
import urllib.request
import os
from pathlib import Path
from collections import defaultdict


def download_from_huggingface_hub():
    """Try to download from Hugging Face Hub using direct file access."""
    print(">>> Attempting to download from Hugging Face Hub...")
    
    # Hugging Face Hub direct file URLs
    base_url = "https://huggingface.co/datasets/empathetic_dialogues/resolve/main"
    files = {
        'train': f"{base_url}/train.csv",
        'validation': f"{base_url}/validation.csv",
        'test': f"{base_url}/test.csv"
    }
    
    data = {}
    for split, url in files.items():
        try:
            print(f"  Trying {split}...")
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode('utf-8')
                lines = content.strip().split('\n')
                if lines and len(lines) > 1:
                    data[split] = lines[1:]  # Skip header
                    print(f"  ✓ Downloaded {len(data[split])} lines from {split}")
        except Exception as e:
            print(f"  ⚠ {split} failed: {str(e)[:100]}")
            data[split] = []
    
    return data


def parse_empathetic_csv(lines):
    """Parse EmpatheticDialogues CSV format."""
    conversations = defaultdict(list)
    
    for line in lines:
        if not line.strip():
            continue
        
        # EmpatheticDialogues CSV format (comma-separated):
        # conv_id,utterance_idx,context,utterance,prompt,speaker_idx,utterance_emotion,utterance_emotion_word
        # Need to handle quoted fields that may contain commas
        try:
            # Use CSV reader for proper parsing
            import io
            reader = csv.reader(io.StringIO(line))
            parts = next(reader)
            
            if len(parts) < 4:
                continue
            
            conv_id = parts[0].strip()
            idx_str = parts[1].strip()
            utterance = parts[3].strip().replace('\n', ' ').replace('\r', ' ').strip()
            
            try:
                idx = int(idx_str) if idx_str else 0
            except ValueError:
                idx = 0
            
            if conv_id and utterance and len(utterance) > 1:
                conversations[conv_id].append((idx, utterance))
        except Exception:
            continue
    
    return conversations


def prepare_data():
    """Download and prepare EmpatheticDialogues as CSV."""
    print("=" * 70)
    print("Preparing EmpatheticDialogues Dataset")
    print("=" * 70)
    print()
    
    # Try downloading from Hugging Face Hub
    data = download_from_huggingface_hub()
    
    if not any(data.values()):
        print()
        print("❌ Automatic download failed.")
        print()
        print("=" * 70)
        print("Manual Download Instructions")
        print("=" * 70)
        print()
        print("1. Download EmpatheticDialogues manually:")
        print("   https://github.com/facebookresearch/EmpatheticDialogues")
        print()
        print("2. Extract the CSV files (train.csv, valid.csv, test.csv)")
        print()
        print("3. Place them in: nova/data/empathetic_data/")
        print()
        print("4. Run this script again - it will detect local files")
        print()
        
        # Check for local files
        local_dir = Path("nova/data/empathetic_data")
        if local_dir.exists():
            print(">>> Checking for local files...")
            local_files = {
                'train': local_dir / 'train.csv',
                'validation': local_dir / 'valid.csv',
                'test': local_dir / 'test.csv'
            }
            
            for split, filepath in local_files.items():
                if filepath.exists():
                    print(f"  ✓ Found {filepath}")
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        data[split] = lines[1:] if len(lines) > 1 else []
                else:
                    # Try alternative names
                    alt_names = ['validation.csv', 'val.csv'] if split == 'validation' else []
                    for alt_name in alt_names:
                        alt_path = local_dir / alt_name
                        if alt_path.exists():
                            print(f"  ✓ Found {alt_path}")
                            with open(alt_path, 'r', encoding='utf-8') as f:
                                lines = f.readlines()
                                data[split] = lines[1:] if len(lines) > 1 else []
                            break
    
    if not any(data.values()):
        print("❌ No data available. Please download manually and try again.")
        return
    
    # Parse all splits
    print()
    print(">>> Parsing conversations...")
    all_conversations = defaultdict(list)
    
    for split, lines in data.items():
        if not lines:
            continue
        conversations = parse_empathetic_csv(lines)
        for conv_id, turns in conversations.items():
            all_conversations[conv_id].extend(turns)
        print(f"  Processed {split}: {len(conversations)} conversations")
    
    print(f"  Total unique conversations: {len(all_conversations)}")
    
    if not all_conversations:
        print("❌ No conversations parsed. Check CSV format.")
        return
    
    # Format as DailyDialogue CSV
    output_file = Path("nova/data/empathetic_train.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print()
    print(f">>> Writing to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # DailyDialogue format: dialog, act, emotion
        writer.writerow(['dialog', 'act', 'emotion'])
        
        count = 0
        total_sentences = 0
        
        for conv_id, turns in all_conversations.items():
            # Sort by utterance index
            turns.sort(key=lambda x: x[0])
            
            # Extract sentences
            sentences = [text for _, text in turns if text.strip()]
            
            if len(sentences) >= 2:  # Only include dialogues with at least 2 turns
                # Format as DailyDialogue: "['sentence1' 'sentence2' ...]"
                dialog_str = " ".join([f"'{s}'" for s in sentences])
                
                # Write: dialog, act (empty), emotion (empty)
                writer.writerow([dialog_str, '', ''])
                count += 1
                total_sentences += len(sentences)
    
    print()
    print("=" * 70)
    print("✓ Success!")
    print("=" * 70)
    print(f"  Saved {count} dialogues")
    print(f"  Total sentences: {total_sentences}")
    print(f"  File: {output_file}")
    print()
    print("Next step: Train with:")
    print(f"  python3 nova/training/train_text_to_geometry.py \\")
    print(f"    --csv {output_file} \\")
    print(f"    --dataset nova \\")
    print(f"    --max-dialogues 10000 \\")
    print(f"    --lattice-size 5 \\")
    print(f"    --collapse-steps 20 \\")
    print(f"    --num-clusters 1000")
    print()


if __name__ == "__main__":
    prepare_data()
