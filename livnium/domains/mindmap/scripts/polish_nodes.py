#!/usr/bin/env python3
"""
Polish Mindmap Nodes using LLM (Groq or Local).

Reads mindmap.json, rewrites node text to be more "human-like", and saves back.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")
USE_GROQ = bool(GROQ_API_KEY)
USE_MINIMAX = bool(MINIMAX_API_KEY)

# Paths
SCRIPT_DIR = Path(__file__).parent
# Assuming mindmap.json is in tools/visualize relative to this script
MINDMAP_PATH = SCRIPT_DIR.parent.parent / "tools" / "visualize" / "mindmap.json"

def _polish_with_groq(text: str) -> str:
    """Polish text using Groq API."""
    try:
        from groq import Groq
    except ImportError:
        print("Error: 'groq' library not found. Run 'pip install groq'")
        sys.exit(1)

    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""Rewrite this specific data description into a single, natural human sentence. 
    Explain what it is simply. Avoid "This node represents..." or stats. 
    Make it sound like a knowledgeable narrator.

    Raw Text: "{text}"

    Human Rewrite:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq Error: {e}")
        return text

def _polish_with_llamacpp(text: str) -> str:
    """Polish text using local Llama.cpp."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("Error: 'llama-cpp-python' not found. Run 'pip install llama-cpp-python'")
        sys.exit(1)
        
    # Static model loading
    if not hasattr(_polish_with_llamacpp, "_model"):
        print(f"Loading local model: {LLM_MODEL_PATH}...")
        _polish_with_llamacpp._model = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=512,
            verbose=False
        )
    
    model = _polish_with_llamacpp._model
    
    prompt = f"""Instruction: Rewrite this technical description into one simple human sentence.
    Input: "{text}"
    Output:"""
    
    output = model(
        prompt, 
        max_tokens=60, 
        stop=["\n", "Input:"], 
        echo=False
    )
    return output['choices'][0]['text'].strip()

def _polish_with_minimax(text: str) -> str:
    """Polish text using MiniMax API (OpenAI-compatible)."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: 'openai' library not found. Run 'pip install openai'")
        sys.exit(1)

    client = OpenAI(
        api_key=MINIMAX_API_KEY,
        base_url="https://api.minimax.io/v1"
    )

    prompt = f"""Rewrite this specific data description into a single, natural human sentence.
    Explain what it is simply. Avoid "This node represents..." or stats.
    Make it sound like a knowledgeable narrator.

    Raw Text: "{text}"

    Human Rewrite:"""

    try:
        response = client.chat.completions.create(
            model="MiniMax-M2.5",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"MiniMax Error: {e}")
        return text


def polish_node(node: Dict[str, Any]) -> bool:
    """Polish a single node. Returns True if changed."""
    original_text = node.get("text", "")

    # Skip if empty or already short/human-like (heuristic)
    if not original_text or len(original_text) < 10:
        return False

    print(f"\nPolishing: {node['id']}")
    print(f"  Old: {original_text[:50]}...")

    new_text = original_text

    if USE_GROQ:
        new_text = _polish_with_groq(original_text)
        # Rate limit protection for free tier
        time.sleep(0.5)
    elif USE_MINIMAX:
        new_text = _polish_with_minimax(original_text)
        time.sleep(0.3)
    elif LLM_MODEL_PATH:
        new_text = _polish_with_llamacpp(original_text)
    else:
        print("  [!] No LLM configured (Set GROQ_API_KEY, MINIMAX_API_KEY, or LLM_MODEL_PATH)")
        return False
        
    if new_text and new_text != original_text:
        node["text"] = new_text
        print(f"  New: {new_text}")
        return True
    
    return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polish mindmap nodes with AI")
    parser.add_argument("--limit", "-n", type=str, default="5", help="Number of nodes to polish (or 'all')")
    args = parser.parse_args()

    if not MINDMAP_PATH.exists():
        print(f"Error: {MINDMAP_PATH} not found.")
        return

    print(f"Loading {MINDMAP_PATH}...")
    with open(MINDMAP_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    print(f"Found {len(nodes)} nodes.")
    
    # Detect backend
    if USE_GROQ:
        print("Using backend: Groq API")
    elif USE_MINIMAX:
        print("Using backend: MiniMax API")
    elif LLM_MODEL_PATH:
        print(f"Using backend: Local Llama ({Path(LLM_MODEL_PATH).name})")
    else:
        print("Error: No LLM backend detected.")
        print("Please export GROQ_API_KEY='...' OR MINIMAX_API_KEY='...' OR LLM_MODEL_PATH='path/to/model.gguf'")
        return

    count = 0
    try:
        limit = args.limit.lower()
        if limit == 'all':
             target_nodes = nodes
        else:
             try:
                 num = int(limit)
                 target_nodes = nodes[:num]
             except ValueError:
                 print(f"Invalid limit: {limit}")
                 return

        print(f"Polishing {len(target_nodes)} nodes...")
        
        for node in target_nodes:
            if polish_node(node):
                count += 1
                
        print(f"\nPolished {count} nodes.")
        
        if count > 0:
            print("Saving updates...")
            with open(MINDMAP_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("Done! Refresh viewer.html to see changes.")
            
    except KeyboardInterrupt:
        print("\nStopped by user. Saving progress...")
        if count > 0:
            with open(MINDMAP_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print("Saved.")

if __name__ == "__main__":
    main()
