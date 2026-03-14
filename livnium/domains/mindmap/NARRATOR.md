# Basin Narrator: Fast, Free, Usable Right Now

## Architecture

```
Geometry (Livnium)  →  Basin (max 10 nodes)
                           ↓
                   Heuristic summary (90%)
                           ↓
               Optional LLM polish (10%)
```

**90% heuristic, 10% language polish, 0% LLM control**

## Design Principles

1. **Lock basin size**: Max 10 nodes per narration (for speed)
2. **Extract**: Top node by mass + top 3 neighbors by alignment
3. **Generate**: Simple sentence template (no AI required)
4. **Optional**: Polish with local small model (Phi-3-mini, Qwen2.5-3B)

## Fast + Free Options

### 🥇 Local, Fastest, Zero Cost

**llama.cpp + small models:**
- `Phi-3-mini (3.8B)` - Recommended
- `Qwen2.5-3B`
- `Mistral-7B (quantized)`

Speed: **10-50ms per short prompt**

### 🥈 Cloud, Very Fast, Free Tier

**Groq (LPU inference):**
- Models: LLaMA-3-8B, Mixtral
- Speed: Very fast
- Free tier exists but limited
- Risk: Rate limits, API dependency

### 🥉 MiniMax API (Cloud, Fast, 204K Context)

**MiniMax:**
- Models: MiniMax-M2.5, MiniMax-M2.5-highspeed
- Speed: Fast (OpenAI-compatible API)
- 204K context window
- Competitive pricing

### Google Gemini (Free)

- Can handle long context
- Slower than Groq/local
- Fine for occasional explanations

## Usage

### Default (Heuristic Only)

```python
from livnium.domains.mindmap import fast_summarize_basin

summary = fast_summarize_basin(
    basin_id="basin_0",
    nodes=all_nodes,
    edges=all_edges,
    basin_node_ids=["node_1", "node_2", ...]
)
```

### With Optional LLM Polish

**Option 1: Local Model (llama.cpp)**
```bash
# Install llama-cpp-python
pip install llama-cpp-python

# Download a model (e.g., Phi-3-mini)
# From: https://huggingface.co/models?library=gguf

# Use it
export USE_LLM_NARRATOR=True
export LLM_MODEL_PATH=/path/to/phi-3-mini.gguf

python3 -m livnium.domains.mindmap.demo
```

**Option 2: Groq API (Cloud, Very Fast, Free Tier)**
```bash
# Install groq
pip install groq

# Get API key from https://console.groq.com/
export USE_LLM_NARRATOR=True
export GROQ_API_KEY=your_api_key_here

python3 -m livnium.domains.mindmap.demo
```

## Key Insight

> **Big companies summarize text.  
> You summarize structure.**

Meaning in Livnium is **spatial** (forces, attractors), not textual.

The narrator's job is **not to understand everything** - it's to **name what already stabilized**.

## Output Format

```
Basin Summary:
Core idea: <concept>
Supporting ideas: <list of 3>
Interpretation: <one short sentence>
```

## Performance

- **Heuristic path**: <10ms per basin
- **With local LLM**: 10-50ms per basin
- **Read-only**: Does not modify geometry or edges

## LLM Integration (Implemented)

The narrator supports two LLM backends:

### 1. llama.cpp (Local, Fastest)

**Setup:**
```bash
pip install llama-cpp-python
```

**Download Models:**
- Phi-3-mini: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
- Qwen2.5-3B: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- Mistral-7B: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

**Usage:**
```bash
export USE_LLM_NARRATOR=True
export LLM_MODEL_PATH=/path/to/model.gguf
```

**Performance:** 10-50ms per basin (depends on model size and CPU)

### 2. Groq API (Cloud, Very Fast, Free Tier)

**Setup:**
```bash
pip install groq
```

**Get API Key:**
- Sign up at https://console.groq.com/
- Free tier: Very generous rate limits

**Usage:**
```bash
export USE_LLM_NARRATOR=True
export GROQ_API_KEY=your_key_here
```

**Performance:** 50-200ms per basin (network latency)

### 3. MiniMax API (Cloud, Fast, 204K Context)

**Setup:**
```bash
pip install openai
```

**Get API Key:**
- Sign up at https://platform.minimaxi.com/
- Create an API key

**Usage:**
```bash
export USE_LLM_NARRATOR=True
export MINIMAX_API_KEY=your_key_here
```

**Performance:** 50-200ms per basin (network latency)

### Fallback Behavior

If LLM is not available or fails, the narrator automatically falls back to the heuristic path. No errors, just graceful degradation.

The narrator tries providers in this order: llama.cpp (local) → Groq → MiniMax → heuristic fallback.

