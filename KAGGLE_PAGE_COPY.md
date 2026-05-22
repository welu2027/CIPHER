# CIPHER

**Calibrated Introspection via Partially Hidden Environment Rules**

Developed by the Feng Lab at Stevens Institute of Technology, CIPHER evaluates whether LLMs know what they don't know and act on it. Models are placed in procedurally generated causal worlds with invented vocabulary (preventing memorization), where some governing rules are completely omitted from the prompt. A model must plan toward a goal, assess its own confidence in each visible rule, rank which hidden rules pose the greatest risk, and submit a contingency plan that holds under adversarial conditions. 1,000 instances across three difficulty tiers (easy: 1 hidden rule, medium: 2, hard: 3).

**Motivation:** Standard calibration benchmarks only test whether a model says "I don't know." CIPHER tests whether a model can reason about the shape of its own ignorance and act on it. Our key finding: objective and executive scores are anti-correlated at r = -0.81 across frontier models, showing that planning ability and contingency quality are distinct capabilities that do not scale together.

Please see our [research paper](#) for further details.

---

## Scoring

Models are ranked on a composite score (0-1) across four dimensions:

| Dimension | Weight | What it measures |
|---|---|---|
| Objective | 35% | Plan quality vs. an oracle solver |
| Calibration | 25% | Accuracy of self-reported confidence (Brier score) |
| Attention | 20% | Rank correlation on hidden rule importance |
| Executive | 20% | Whether its contingency plan outperforms its primary plan under adversarial conditions |

The composite score is a weighted mean: `0.35 * objective + 0.25 * calibration + 0.20 * attention + 0.20 * executive`.

---

## Setup

### Install Dependencies

```python
import subprocess, sys

def _pip(*packages):
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *packages], check=True)

_pip("kaggle-benchmarks", "pandas", "matplotlib", "numpy")
print("Dependencies installed.")
```

### Setup Clients

The dataset is attached at `/kaggle/input/datasets/wenhaolu49/cipher`. Add your model API keys as Kaggle Secrets before running.

**WARNING:** This notebook will execute real API calls. Check usage limits before running.

```python
import os, sys, json
import pandas as pd
import kaggle_benchmarks as kbench
from cipher.generator import Instance
from cipher.scorer import score_response
from cipher.schema import validate_response

_KAGGLE_ROOT = "/kaggle/input/datasets/wenhaolu49/cipher"
print("Available models:", list(kbench.llms.keys()))
```

---

## Load Instances

```python
DATA_PATH = os.path.join(_KAGGLE_ROOT, "data", "instances.jsonl")

with open(DATA_PATH) as f:
    ALL_RECORDS = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(ALL_RECORDS)} instances")
# Output: Loaded 1000 instances
```

Each instance contains:
- `prompt` - the full task prompt shown to the model (visible rules only)
- `difficulty` - `easy`, `medium`, or `hard`
- `hidden` - ground truth including hidden rules, oracle scores, and metacognition labels (used for scoring only)

---

## Response Generation

Models must return strict JSON with four fields: `metacog_assessment` (confidence per rule component), `critical_unknowns_ranked` (hidden rules by estimated impact), `exploratory_actions` (optional probes), `final_plan` (action sequence), and `self_judgment` (robustness score + named risks + contingency plan).

```python
@kbench.task(name="cipher_single_instance_scorer_v2", store_task=False)
def cipher_single_instance_scorer(llm, prompt: str, record_json: str) -> dict:
    try:
        reply = llm.prompt(prompt)
        rec   = json.loads(record_json)
        scored = _score(str(reply), rec)
        scored["raw_response"] = str(reply)
        return scored
    except Exception:
        return dict(composite=0.0, objective=0.0, calibration=0.0, attention=0.0, executive=0.0)
```

---

## Batch Evaluation

Set `QUALITATIVE_N` to an integer (e.g. `10`) for a fast test run, or `None` for the full 1,000-instance evaluation.

```python
QUALITATIVE_N = None  # set to 10 for a quick test

@kbench.task(name="cipher_eval_full")
def cipher_metacognition_evaluation(llm):
    records_to_run = ALL_RECORDS[:QUALITATIVE_N] if QUALITATIVE_N else ALL_RECORDS
    evaluation_df = pd.DataFrame([{
        "prompt":      rec["prompt"],
        "record_json": json.dumps(rec),
    } for rec in records_to_run])

    with kbench.client.enable_cache():
        runs = cipher_single_instance_scorer.evaluate(
            llm=[llm],
            evaluation_data=evaluation_df,
            n_jobs=8,
            remove_run_files=True,
        )

    results_df = runs.as_dataframe()
    return results_df

%choose cipher_eval_full
cipher_metacognition_evaluation.run(kbench.llm)
```

Sample output:

```
=== CIPHER Results (n=1000) ===
Dimension      Score
----------------------
Composite      0.684
Objective      0.566
Calibration    0.902
Attention      0.672
Executive      0.630

=== By Difficulty ===
Difficulty  Composite  Objective  Calibration  Attention  Executive
--------------------------------------------------------------------
easy            0.679      0.576        0.915      0.615      0.627
medium          0.683      0.550        0.901      0.682      0.642
hard            0.692      0.589        0.891      0.707      0.609
```

---

## Closing Remarks

Thank you for running CIPHER. For independently validated scores across all frontier models, check the official leaderboard. Questions, comments, or issues? Share your thoughts in the discussion forum.
