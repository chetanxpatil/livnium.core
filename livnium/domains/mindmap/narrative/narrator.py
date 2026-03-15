"""
Fast Local Narrator: Convert Basins to Human-Readable Sentences

90% heuristic, 10% optional language polish.
Read-only observer that does not modify geometry or edges.

Design:
- Lock basin size: max 10 nodes per narration
- Extract: top node by mass + top 3 neighbors by alignment
- Generate: simple sentence template (no AI required)
- Optional: polish with local small model (Phi-3-mini, Qwen2.5-3B via llama.cpp)
"""

import os
from typing import List, Dict, Any, Optional
from ..ingestion.ingest import ThoughtNode


# Feature flags
USE_LLM_NARRATOR = os.getenv("USE_LLM_NARRATOR", "False").lower() == "true"
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "")  # Path to local model (llama.cpp format)
MAX_NODES_PER_NARRATION = 10  # Lock basin size for narration


def summarize_basin(
    basin_id: str,
    nodes: List[ThoughtNode],
    edges: List[Dict[str, Any]],
    basin_node_ids: List[str]
) -> Dict[str, Any]:
    """
    Fast heuristic summarization of a basin.
    
    Strategy:
    1. Lock to top 10 nodes by mass
    2. Extract top node (center) + top 3 neighbors by alignment
    3. Generate simple sentence template
    4. Optional: polish with local LLM
    
    Args:
        basin_id: Basin identifier
        nodes: All nodes
        edges: All edges
        basin_node_ids: Node IDs in this basin
        
    Returns:
        Dictionary with core_idea, supporting_ideas, interpretation
    """
    # Filter to basin nodes
    basin_nodes = [n for n in nodes if n.id in basin_node_ids]
    
    if not basin_nodes:
        return {
            "core_idea": "Unknown",
            "supporting_ideas": [],
            "interpretation": "Empty basin.",
            "stats": {}
        }
    
    # Lock basin size: sort by mass, take top N
    basin_nodes.sort(key=lambda n: n.mass, reverse=True)
    top_nodes = basin_nodes[:MAX_NODES_PER_NARRATION]
    
    # Central node (highest mass)
    central_node = top_nodes[0]
    
    # Extract core idea (first sentence, cleaned)
    core_text = _extract_first_sentence(central_node.text)
    core_idea = core_text if core_text else "Unclear concept"
    
    # Find top 3 neighbors by alignment to central node
    node_map = {n.id: n for n in nodes}
    central_edges = [
        e for e in edges
        if (e["source"] == central_node.id or e["target"] == central_node.id)
        and (e["source"] in [n.id for n in top_nodes] and e["target"] in [n.id for n in top_nodes])
    ]
    central_edges.sort(key=lambda e: e.get("alignment", 0), reverse=True)
    
    # Get top 3 supporting nodes
    supporting_ideas = []
    seen_ids = {central_node.id}
    for edge in central_edges[:6]:  # Check more to get 3 good ones
        other_id = edge["target"] if edge["source"] == central_node.id else edge["source"]
        if other_id not in seen_ids:
            other_node = node_map.get(other_id)
            if other_node:
                text = _extract_first_sentence(other_node.text)
                if text and len(text) > 10:
                    supporting_ideas.append(text)
                    seen_ids.add(other_id)
                    if len(supporting_ideas) >= 3:
                        break
    
    # Compute simple stats
    basin_edges = [e for e in edges 
                   if e["source"] in [n.id for n in top_nodes] 
                   and e["target"] in [n.id for n in top_nodes]]
    
    if basin_edges:
        avg_alignment = sum(e.get("alignment", 0) for e in basin_edges) / len(basin_edges)
        avg_tension = sum(e.get("tension", 0) for e in basin_edges) / len(basin_edges)
    else:
        avg_alignment = 0.0
        avg_tension = 0.0
    
    # Generate interpretation (simple template, no AI)
    interpretation = _generate_simple_interpretation(
        core_idea,
        supporting_ideas,
        len(basin_nodes),
        avg_alignment,
        avg_tension
    )
    
    # Optional: polish with LLM (local or cloud)
    if USE_LLM_NARRATOR and (LLM_MODEL_PATH or os.getenv("GROQ_API_KEY") or os.getenv("MINIMAX_API_KEY")):
        try:
            print(f"  → Polishing basin '{basin_id}' with LLM...")
            interpretation = _polish_with_llm(interpretation, core_idea, supporting_ideas)
            print(f"  → LLM polish complete")
        except Exception as e:
            # Fall back to heuristic if LLM fails
            print(f"  ⚠ LLM polish failed (using heuristic): {e}")
            pass
    
    return {
        "core_idea": core_idea,
        "supporting_ideas": supporting_ideas[:3],  # Lock to 3
        "interpretation": interpretation,
        "stats": {
            "size": len(basin_nodes),
            "avg_alignment": avg_alignment,
            "avg_tension": avg_tension,
            "central_node_id": central_node.id
        }
    }


def _extract_first_sentence(text: str, max_length: int = 80) -> str:
    """Extract and clean first sentence from text."""
    if not text:
        return ""
    
    # Remove markdown headers
    text = text.replace('#', '').strip()
    
    # Get first sentence
    sentences = text.split('.')
    if sentences:
        first = sentences[0].strip()
        # Clean up
        first = first.replace('\n', ' ').replace('  ', ' ').strip()
        # Truncate if needed
        if len(first) > max_length:
            first = first[:max_length-3] + "..."
        return first
    
    # Fallback: first 80 chars
    text = text.replace('\n', ' ').strip()
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    return text


def _generate_simple_interpretation(
    core_idea: str,
    supporting_ideas: List[str],
    size: int,
    avg_alignment: float,
    avg_tension: float
) -> str:
    """
    Generate interpretation focused on intent and purpose, not geometry.
    
    Answers three questions:
    1. What problem is this cluster about?
    2. What tension does it resolve or expose?
    3. Why does it exist separately?
    
    No AI required - just pattern matching and templates.
    """
    core_lower = core_idea.lower()
    
    # Determine what kind of tension/conflict exists
    if avg_tension > 0.3:
        tension_desc = "unresolved tension"
        conflict_hint = "conflicting perspectives or competing requirements"
    elif avg_tension > 0.2:
        tension_desc = "moderate tension"
        conflict_hint = "some internal disagreement or tradeoffs"
    else:
        tension_desc = "low tension"
        conflict_hint = "aligned understanding"
    
    # Determine shared role/purpose
    if len(supporting_ideas) >= 2:
        # Try to infer shared role from supporting ideas
        supporting_text = ", ".join(supporting_ideas[:2])
        shared_role = f"they address {supporting_text}"
    elif len(supporting_ideas) == 1:
        shared_role = f"it connects to {supporting_ideas[0].lower()}"
    else:
        shared_role = "it stands alone"
    
    # Build interpretation focused on intent
    # Structure: "This cluster is about X. It connects Y because Z. The tension suggests W."
    
    if len(supporting_ideas) >= 2:
        interpretation = f"This cluster is about {core_lower}. It connects {supporting_ideas[0].lower()} and {supporting_ideas[1].lower()} because they share a common role or address related problems. The {tension_desc} suggests {conflict_hint}."
    elif len(supporting_ideas) == 1:
        interpretation = f"This cluster is about {core_lower}. {shared_role}, indicating they work together on a shared problem. The {tension_desc} suggests {conflict_hint}."
    else:
        interpretation = f"This cluster is about {core_lower}. {shared_role}. The {tension_desc} suggests {conflict_hint}."
    
    # Ensure reasonable length
    if len(interpretation) > 250:
        interpretation = interpretation[:247] + "..."
    
    return interpretation


def _polish_with_llm(
    interpretation: str,
    core_idea: str,
    supporting_ideas: List[str]
) -> str:
    """
    Optional: polish interpretation with LLM (10% language polish).

    Supports:
    - llama.cpp (local, via llama-cpp-python)
    - Groq API (cloud, very fast, free tier)
    - MiniMax API (cloud, fast, 204K context, OpenAI-compatible)

    Falls back gracefully if LLM not available.
    """
    # Try llama.cpp first (local, fastest)
    if LLM_MODEL_PATH and os.path.exists(LLM_MODEL_PATH):
        try:
            return _polish_with_llamacpp(interpretation, core_idea, supporting_ideas)
        except Exception as e:
            # Fall through to other options
            pass
    
    # Try Groq API (cloud, very fast, free tier)
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if groq_api_key:
        try:
            return _polish_with_groq(interpretation, core_idea, supporting_ideas)
        except Exception as e:
            # Fall through to other options
            pass

    # Try MiniMax API (cloud, fast, OpenAI-compatible)
    minimax_api_key = os.getenv("MINIMAX_API_KEY", "")
    if minimax_api_key:
        try:
            return _polish_with_minimax(interpretation, core_idea, supporting_ideas)
        except Exception as e:
            # Fall through to heuristic
            pass

    # Fallback: return heuristic version
    return interpretation


def _polish_with_llamacpp(
    interpretation: str,
    core_idea: str,
    supporting_ideas: List[str]
) -> str:
    """
    Polish with llama.cpp (local model via llama-cpp-python).
    
    Requires: pip install llama-cpp-python
    Model: Phi-3-mini, Qwen2.5-3B, or Mistral-7B (quantized .gguf format)
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    
    # Load model (lazy, cache it)
    if not hasattr(_polish_with_llamacpp, '_model'):
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {LLM_MODEL_PATH}")
        print(f"    Loading LLM model from {LLM_MODEL_PATH}...")
        _polish_with_llamacpp._model = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=512,  # Small context for speed
            n_threads=4,  # Adjust for your CPU
            verbose=False
        )
        print(f"    Model loaded successfully")
    
    model = _polish_with_llamacpp._model
    
    # Build prompt focused on intent, not geometry (≤120 tokens)
    prompt = f"""Rewrite to explain intent, not statistics:

"{interpretation}"

Focus on: what problem this solves, what tension it addresses, why it exists separately.
One clear sentence. No numbers or technical terms."""
    
    # Generate (fast, short response - ≤120 tokens total)
    try:
        response = model(
            prompt,
            max_tokens=80,  # Short response for speed
            temperature=0.4,  # Slightly higher for more natural language
            stop=["\n\n", "---", "\n"],
            echo=False
        )
        
        polished = response['choices'][0]['text'].strip()
    except Exception as e:
        raise RuntimeError(f"LLM generation failed: {e}")
    
    # Clean up - take first sentence only
    polished = polished.split('.')[0].strip()
    if polished and not polished.endswith('.'):
        polished += '.'
    
    # Validate output
    if not polished or len(polished) < 15:
        return interpretation  # Fallback if output is bad
    
    return polished


def _polish_with_groq(
    interpretation: str,
    core_idea: str,
    supporting_ideas: List[str]
) -> str:
    """
    Polish with Groq API (cloud, very fast, free tier).
    
    Requires: export GROQ_API_KEY=your_key
    Models: llama-3-8b-instant, mixtral-8x7b-32768
    """
    try:
        from groq import Groq
    except ImportError:
        raise ImportError("groq not installed. Install with: pip install groq")
    
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    
    client = Groq(api_key=api_key)
    
    # Build prompt focused on intent, not geometry (≤120 tokens)
    prompt = f"""Rewrite to explain intent, not statistics:

"{interpretation}"

Focus on: what problem this solves, what tension it addresses, why it exists separately.
One clear sentence. No numbers or technical terms."""
    
    # Call Groq API (very fast)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Fast, free tier model
        messages=[
            {"role": "system", "content": "You explain intent and purpose, not statistics. Focus on what problems clusters solve and what tensions they address."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=80,  # Short response
        temperature=0.4  # Slightly higher for natural language
    )
    
    polished = response.choices[0].message.content.strip()
    
    # Clean up - take first sentence only
    polished = polished.split('.')[0].strip()
    if polished and not polished.endswith('.'):
        polished += '.'
    
    # Validate output
    if not polished or len(polished) < 15:
        return interpretation  # Fallback if output is bad
    
    return polished


def _polish_with_minimax(
    interpretation: str,
    core_idea: str,
    supporting_ideas: List[str]
) -> str:
    """
    Polish with MiniMax API (cloud, fast, OpenAI-compatible).

    Requires: pip install openai
    Set: export MINIMAX_API_KEY=your_key
    Models: MiniMax-M2.5, MiniMax-M2.5-highspeed (204K context)
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Install with: pip install openai")

    api_key = os.getenv("MINIMAX_API_KEY", "")
    if not api_key:
        raise ValueError("MINIMAX_API_KEY not set")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.minimax.io/v1"
    )

    # Build prompt focused on intent, not geometry (≤120 tokens)
    prompt = f"""Rewrite to explain intent, not statistics:

"{interpretation}"

Focus on: what problem this solves, what tension it addresses, why it exists separately.
One clear sentence. No numbers or technical terms."""

    # Call MiniMax API (OpenAI-compatible)
    response = client.chat.completions.create(
        model="MiniMax-M2.5",
        messages=[
            {"role": "system", "content": "You explain intent and purpose, not statistics. Focus on what problems clusters solve and what tensions they address."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=80,  # Short response
        temperature=0.4  # Slightly higher for natural language
    )

    polished = response.choices[0].message.content.strip()

    # Clean up - take first sentence only
    polished = polished.split('.')[0].strip()
    if polished and not polished.endswith('.'):
        polished += '.'

    # Validate output
    if not polished or len(polished) < 15:
        return interpretation  # Fallback if output is bad

    return polished


def format_basin_summary(summary: Dict[str, Any]) -> str:
    """
    Format basin summary into readable output.
    
    Output format:
    ```
    Basin Summary:
    Core idea: <concept>
    Supporting ideas: <list>
    Interpretation: <one short sentence>
    ```
    """
    lines = [
        "Basin Summary:",
        f"Core idea: {summary['core_idea']}",
    ]
    
    if summary['supporting_ideas']:
        lines.append(f"Supporting ideas: {', '.join(summary['supporting_ideas'])}")
    else:
        lines.append("Supporting ideas: None")
    
    lines.append(f"Interpretation: {summary['interpretation']}")
    
    return "\n".join(lines)
