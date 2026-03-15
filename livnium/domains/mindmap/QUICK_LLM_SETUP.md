# Quick LLM Setup for Basin Narrator

## Option 1: Download Model (Recommended)

Run the helper script:

```bash
./livnium/domains/mindmap/download_model.sh
```

This will:
1. Let you choose a model (Phi-3-mini recommended)
2. Download it to `~/.livnium_models/`
3. Show you the exact commands to use it

## Option 2: Manual Download

1. **Download a model** from HuggingFace (GGUF format):
   - Phi-3-mini: https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-GGUF
   - Qwen2.5-3B: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
   - Mistral-7B: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF

2. **Set environment variables:**
   ```bash
   export USE_LLM_NARRATOR=True
   export LLM_MODEL_PATH=/path/to/your/model.gguf
   ```

3. **Regenerate:**
   ```bash
   ./livnium/domains/mindmap/regenerate.sh
   ```

## Option 3: Use Groq API (No Download Needed)

1. **Install groq:**
   ```bash
   pip install groq
   ```

2. **Get API key:**
   - Sign up at https://console.groq.com/
   - Free tier is very generous

3. **Set environment variables:**
   ```bash
   export USE_LLM_NARRATOR=True
   export GROQ_API_KEY=your_key_here
   ```

4. **Regenerate:**
   ```bash
   ./livnium/domains/mindmap/regenerate.sh
   ```

## Option 4: Use MiniMax API (Fast, 204K Context)

1. **Install openai:**
   ```bash
   pip install openai
   ```

2. **Get API key:**
   - Sign up at https://platform.minimaxi.com/
   - Create an API key

3. **Set environment variables:**
   ```bash
   export USE_LLM_NARRATOR=True
   export MINIMAX_API_KEY=your_key_here
   ```

4. **Regenerate:**
   ```bash
   ./livnium/domains/mindmap/regenerate.sh
   ```

## Test Without LLM (Heuristic Only)

Just run without setting `USE_LLM_NARRATOR`:

```bash
./livnium/domains/mindmap/regenerate.sh
```

The narrator will use fast heuristics (works immediately, no setup needed).

## Performance

- **Heuristic only**: <10ms per basin
- **Local LLM (Phi-3-mini)**: 10-50ms per basin
- **Groq API**: 50-200ms per basin
- **MiniMax API**: 50-200ms per basin

All options work great - choose based on your preference!

