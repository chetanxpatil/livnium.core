"""
Convert emotion-emotion_69k.csv to DailyDialogue format
"""

import csv
import re
from pathlib import Path
from collections import defaultdict


def parse_empathetic_dialogue(dialog_text):
    """Extract Customer and Agent turns from empathetic_dialogues field."""
    if not dialog_text or not dialog_text.strip():
        return []
    
    sentences = []
    
    # The format appears to be: "Customer :text\nAgent :text"
    # Split by newlines first
    lines = dialog_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if it starts with "Customer :" or "Agent :"
        if line.startswith("Customer :") or line.startswith("Customer:"):
            text = line.replace("Customer :", "").replace("Customer:", "").strip()
            if text:
                sentences.append(text)
        elif line.startswith("Agent :") or line.startswith("Agent:"):
            text = line.replace("Agent :", "").replace("Agent:", "").strip()
            if text:
                sentences.append(text)
        else:
            # Might be continuation of previous line or standalone
            if line:
                sentences.append(line)
    
    return sentences


def convert_csv(input_file, output_file):
    """Convert emotion-emotion_69k.csv to DailyDialogue format."""
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    print()
    
    conversations = defaultdict(list)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        print(f"Header: {header}")
        print()
        
        for row_idx, row in enumerate(reader):
            if len(row) < 4:
                continue
            
            try:
                # Column indices:
                # 0: index
                # 1: Situation
                # 2: emotion
                # 3: empathetic_dialogues
                # 4: labels
                
                situation = row[1] if len(row) > 1 else ""
                emotion = row[2] if len(row) > 2 else ""
                dialog_text = row[3] if len(row) > 3 else ""
                
                if not dialog_text:
                    continue
                
                # Parse the dialogue
                sentences = parse_empathetic_dialogue(dialog_text)
                
                if len(sentences) >= 1:
                    # Use situation + row index as conversation ID to group related turns
                    # But we want to keep each dialogue turn separate for now
                    # Actually, let's group by situation to form full conversations
                    conv_id = situation[:100] if situation else f"conv_{row_idx // 10}"
                    if conv_id not in conversations:
                        conversations[conv_id] = []
                    conversations[conv_id].extend(sentences)
                
            except Exception as e:
                if row_idx < 10:
                    print(f"  Warning on row {row_idx}: {e}")
                continue
    
    print(f"Found {len(conversations)} unique conversations")
    print()
    print("Writing DailyDialogue format...")
    
    # Write in DailyDialogue format
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['dialog', 'act', 'emotion'])
        
        count = 0
        total_sentences = 0
        
        for conv_id, sentences in conversations.items():
            if len(sentences) >= 2:
                # Format as DailyDialogue: "['sentence1' 'sentence2' ...]"
                dialog_str = " ".join([f"'{s}'" for s in sentences])
                writer.writerow([dialog_str, '', ''])
                count += 1
                total_sentences += len(sentences)
    
    print()
    print("=" * 70)
    print("✓ Conversion Complete!")
    print("=" * 70)
    print(f"  Conversations: {count}")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Output file: {output_file}")
    print()
    print("Train with:")
    print(f"  python3 nova/training/train_text_to_geometry.py \\")
    print(f"    --csv {output_file} \\")
    print(f"    --dataset nova \\")
    print(f"    --max-dialogues 10000 \\")
    print(f"    --lattice-size 5 \\")
    print(f"    --collapse-steps 20 \\")
    print(f"    --num-clusters 1000")


if __name__ == "__main__":
    input_file = Path("nova/data/emotion-emotion_69k.csv")
    output_file = Path("nova/data/empathetic_train.csv")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        exit(1)
    
    convert_csv(input_file, output_file)

