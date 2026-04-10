# Graphify Local LLM Setup

This repo has been updated so `graphify` can run locally and use:

- **Ollama** via `--semantic-backend ollama`
- **torch / local Hugging Face models** via `--semantic-backend torch`
- **AST-only mode** via `--semantic-backend none`

---

## What changed

The following Python files were added or updated to support local scanning and local semantic extraction:

- `graphify/pipeline.py` — standalone local pipeline
- `graphify/semantic_local.py` — local semantic extraction for Ollama / torch / heuristic fallback
- `graphify/__main__.py` — `graphify .` and `graphify run ...` now work locally
- `graphify/watch.py` — local watch mode can refresh semantic edges
- `graphify/__init__.py` — exports the new helpers
- `tests/test_local_llm.py` — regression tests for the local backend flow

---

## Local usage

### 1) Activate the repo virtual environment

```bash
cd /Users/matebestek/Development/graphify
source .venv/bin/activate
```

---

## Run graphify locally

### AST-only mode

```bash
graphify . --semantic-backend none --no-viz
```

This writes:

- `graphify-out/graph.json`
- `graphify-out/GRAPH_REPORT.md`

---

## Use with Ollama

Start Ollama and pull a model first:

```bash
ollama serve
ollama pull llama3.2
```

Then run:

```bash
graphify . --semantic-backend ollama --ollama-model llama3.2
```

For image-heavy corpora, use a vision-capable model such as:

```bash
graphify . --semantic-backend ollama --ollama-model llava
```

If Ollama is not running, `graphify` now falls back cleanly instead of crashing.

---

## Use with torch / local Hugging Face models

Install the optional local-model packages:

```bash
pip install transformers sentence-transformers
```

Then run with a local or downloaded model:

```bash
graphify . --semantic-backend torch --local-model /path/to/your/model
```

Or with a model name:

```bash
graphify . --semantic-backend torch --local-model sentence-transformers/all-MiniLM-L6-v2

graphify . --semantic-backend torch --local-model google/gemma-4-31B-it

Example:
graphify ../../OneDrive\ -\ Onkološki\ inštitut\ Ljubljana/Documents/Obsidian/obsidian/work\ log --semantic-backend torch --local-model google/gemma-4-31B-it

```

If full text generation is not available, the code falls back to local semantic similarity / heuristic extraction.

---

## Watch mode

### Ollama watch mode

```bash
graphify . --watch --semantic-backend ollama --ollama-model llama3.2
```

### Torch watch mode

```bash
graphify . --watch --semantic-backend torch --local-model sentence-transformers/all-MiniLM-L6-v2
```

---

## Optional outputs

```bash
graphify . --semantic-backend ollama --wiki
graphify . --semantic-backend torch --graphml --svg
graphify . --semantic-backend none --obsidian
```

---

## Verification completed

The following commands were run successfully in this repo:

```bash
./.venv/bin/python -m pytest tests/test_local_llm.py -q
```

Result:

```text
3 passed in 0.52s
```

```bash
./.venv/bin/python -m pytest tests/test_install.py tests/test_pipeline.py tests/test_watch.py -q
```

Result:

```text
54 passed in 3.66s
```

```bash
./.venv/bin/graphify . --semantic-backend none --no-viz
```

Result:

```text
Corpus: 110 files · ~113,776 words
  code:     81 files
  docs:     28 files
  papers:   1 files
  images:   0 files
Semantic backend: none -> none

Wrote:
  - /Users/matebestek/Development/graphify/graphify-out/graph.json
  - /Users/matebestek/Development/graphify/graphify-out/GRAPH_REPORT.md
```

---

## Important note about `graphify .`

If `graphify .` from your shell still fails, your shell is likely picking up an older global install instead of the repo virtual environment.

Use either:

```bash
source .venv/bin/activate
graphify . --semantic-backend none --no-viz
```

or:

```bash
./.venv/bin/graphify . --semantic-backend none --no-viz
```

To check which one your shell is using:

```bash
which graphify
```

---

## Recommended quick start

```bash
cd /Users/matebestek/Development/graphify
source .venv/bin/activate
ollama serve
ollama pull llama3.2
graphify . --semantic-backend ollama --ollama-model llama3.2
```
