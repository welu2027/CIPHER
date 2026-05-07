# CIPHER
### Calibrated Introspection via Partially Hidden Environment Rules

CIPHER is a procedurally-generated benchmark designed to test whether language models actually know what they know - and what they don't. Every instance is a tiny invented world with its own causal rules, but some of those rules are deliberately hidden. The model has to figure out how much it can trust its own understanding, rank which gaps matter most, probe the system if it wants, commit to a plan, and then honestly assess how robust that plan is.

The whole point is that no model can memorize its way through this. Every world uses made-up vocabulary - invented entity names, invented property words, invented causal language - generated fresh from abstract math. If a model scores well, it's because it genuinely reasoned under uncertainty, not because it pattern-matched on something from training.

## What's in this dataset

```
cipher/
  __init__.py
  world.py          state representation, rules, action engine
  generator.py      procedural instance generator (seeded, fully deterministic)
  simulator.py      executes a model's plan against the hidden rules
  scorer.py         computes all four scoring dimensions
  schema.py         validates and parses model JSON output
  prompt.py         builds the natural-language prompt for each instance
  flavor.py         procedural vocabulary layer (invented terms per instance)
  optimal.py        beam-search oracle for computing normalized scores
data/
  instances.jsonl   1,000 pre-generated instances (seed=2026, with oracle bounds)
scripts/
  generate_dataset.py   regenerate the benchmark at any seed/size
```

The `data/instances.jsonl` file has everything needed to run evaluations without regenerating. Each line is one instance with a `prompt` field (what the model sees) and a `hidden` field (ground truth used for scoring - not shown to the model).

## How scoring works

Each model response is scored on four dimensions, all normalized to [0, 1]:

| Dimension | What it's measuring |
|-----------|---------------------|
| **Objective** | How good is the final plan vs. the oracle beam search? |
| **Calibration** | Brier score on the model's stated confidence in its own knowledge |
| **Attention** | Does the model rank the important unknowns above the unimportant ones? |
| **Executive** | Plan structure: named risks, alternative plans, probe strategy |

The composite is a weighted average. One thing worth noting: no simple strategy wins all four dimensions at once. A model that always plans greedily gets a great objective score but zero attention and poor calibration. A model that hedges everywhere gets decent calibration but a bad objective. A model that genuinely reasons about what it doesn't know - and acts accordingly - is the one that scores well across the board.

## Quick Start

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run a stub baseline** (no API key needed)
```bash
python3 scripts/evaluate.py --data data/instances.jsonl --model stub-greedy --out results.json
```

Available built-in models: `stub-noop`, `stub-random`, `stub-greedy`, `stub-cautious`, `stub-probe-heavy`.

**Run a real model**

Set your API key, then pass the model flag:
```bash
# Claude
export ANTHROPIC_API_KEY=your_key
python3 scripts/evaluate.py --data data/instances.jsonl --model claude --out results.json

# Gemini
export GOOGLE_API_KEY=your_key
python3 scripts/evaluate.py --data data/instances.jsonl --model gemini --out results.json
```

Use `--limit N` to run on a subset (e.g. `--limit 50`) before a full run.

**Adding a custom model**

Register a function in the `AGENTS` dict in `scripts/evaluate.py`:
```python
def my_agent(inst: Instance) -> dict:
    prompt = inst.prompt  # the natural-language prompt string
    # call your model, return a dict matching the schema in cipher/schema.py
    ...

AGENTS["my-model"] = my_agent
```

## Regenerating the dataset

The included `data/instances.jsonl` is ready to use, but if you want to regenerate it at a different seed or size:

```bash
python3 scripts/generate_dataset.py --n 1000 --out data/instances.jsonl --seed 2026 --oracle
```

The `--oracle` flag pre-computes the best and worst achievable objectives for each instance (used to normalize scores).
