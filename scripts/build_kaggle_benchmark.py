"""Kaggle Benchmarks adapter for CIPHER.

CIPHER: Calibrated Introspection via Partially Hidden Environment Rules

HOW TO USE
==========
1. Create a new Kaggle Benchmark Task Notebook.
2. Attach `data/instances.jsonl` as a Dataset input (upload the file to a new
   Kaggle Dataset first, then add it to the notebook).
3. Zip the repo and upload as a Dataset, then add it to the notebook.
   Adjust `sys.path` accordingly in the cell below.
4. Run all cells. The `@kbench.task` decorator + `.run()` calls will cause
   kaggle-benchmarks to auto-generate task files and run files under the
   hood. After execution, go to
   https://www.kaggle.com/benchmarks/tasks/new to finalize and publish the
   benchmark named "cipher".

Local preview
-------------
This file can also be run locally to sanity-check the task logic before
porting to a notebook:

    pip install kaggle-benchmarks
    python3 scripts/build_kaggle_benchmark.py --preview

"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the ignoranceforge package importable whether this is run locally or
# dropped into a Kaggle notebook with the repo attached as a Dataset.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, "..")))
# In a Kaggle notebook with the repo uploaded under /kaggle/input/ignoranceforge:
sys.path.insert(0, "/kaggle/input/ignoranceforge")

from ignoranceforge.generator import Instance
from ignoranceforge.world import (
    World, State, EntityState, Rule, Trigger, Effect, Action,
)
from ignoranceforge.schema import validate_response
from ignoranceforge.scorer import score_response
from ignoranceforge import build_prompt


# ---------------------------------------------------------------------------
# Rehydrate Instances from JSONL (mirrors scripts/evaluate.py helper)
# ---------------------------------------------------------------------------

def _instance_from_record(rec):
    hidden = rec["hidden"]
    hidden_rules_lookup = {h["rule_name"]: set(h["hidden"])
                           for h in hidden.get("hidden_fields", [])}
    rules = []
    for r in hidden["rules"]:
        hides = hidden_rules_lookup.get(r["name"], set())
        t_raw, e_raw = r["trigger"], r["effect"]
        trigger = Trigger(
            kind=t_raw["kind"], i=t_raw["i"], j=t_raw.get("j", -1),
            k=t_raw.get("k", 0),
            hidden_kind="trigger_kind" in hides,
            hidden_k="trigger_k" in hides,
        )
        effect = Effect(
            kind=e_raw["kind"], target=e_raw["target"],
            delta=e_raw.get("delta", 0), source=e_raw.get("source", -1),
            hidden_kind="effect_kind" in hides,
            hidden_delta="effect_delta" in hides,
        )
        rules.append(Rule(name=r["name"], trigger=trigger, effect=effect))
    initial = State(tuple(EntityState(phase=e["phase"], flux=e["flux"])
                          for e in hidden["initial_state"]))
    world = World(initial=initial, rules=tuple(rules),
                  horizon=hidden["horizon"])
    return Instance(
        id=rec["id"], seed=rec["seed"], difficulty=rec["difficulty"],
        world=world, public_rule_descriptions=[],
        hidden_fields=hidden.get("hidden_fields", []),
        metacog_ground_truth=hidden["metacog_ground_truth"],
        true_unknown_ranking=hidden["true_unknown_ranking"],
        oracle_objective=hidden.get("oracle_best"),
    )


def _load_instances(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# The single parametrized task. Each JSONL row becomes one .run() call; the
# kaggle-benchmarks runtime records each run as a separate task result.
# ---------------------------------------------------------------------------

def _score_model_output(raw_text, rec):
    """Parse JSON out of the model's reply and return (composite, breakdown)."""
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start == -1 or end == -1:
        return 0.0, {"error": "no_json_found"}
    try:
        raw = json.loads(raw_text[start:end + 1])
    except json.JSONDecodeError as e:
        return 0.0, {"error": f"invalid_json: {e}"}

    inst = _instance_from_record(rec)
    resp = validate_response(raw)
    best = rec["hidden"].get("oracle_best")
    worst = rec["hidden"].get("oracle_worst")
    breakdown = score_response(resp, inst, best_obj=best, worst_obj=worst)
    return breakdown.composite, breakdown.to_dict()


def register_and_run(instances, limit=None):
    """Define the CIPHER kbench task and run it once per instance.

    This MUST execute inside a Kaggle Benchmark Task notebook for the benchmark
    to be registered with Kaggle's backend. Locally, it will just score against
    a stub LLM for preview purposes.
    """
    import kaggle_benchmarks as kbench
    from kaggle_benchmarks.actors.llms import GoogleGenAI
    from google import genai

    @kbench.task(name="cipher_metacog")
    def cipher_task(llm, instance_id: str, prompt: str,
                    record_json: str):
        """A single CIPHER instance. The LLM must return strict JSON per the
        schema shown in the prompt. Scoring combines objective, calibration,
        attention, and executive sub-scores to measure epistemic self-awareness."""
        response = llm.prompt(prompt)
        rec = json.loads(record_json)
        composite, breakdown = _score_model_output(str(response), rec)
        kbench.assertions.assert_true(
            composite >= 0.5,
            f"Composite {composite:.3f} < 0.5 for {instance_id}. "
            f"Breakdown: {breakdown}",
        )

    shared_llm = GoogleGenAI(
        client=genai.Client(api_key=os.environ["MODEL_PROXY_API_KEY"]),
        model="gemini-2.0-flash",
    )
    run_records = instances if limit is None else instances[:limit]
    for rec in run_records:
        cipher_task.run(
            llm=shared_llm,
            instance_id=rec["id"],
            prompt=rec["prompt"],
            record_json=json.dumps(rec),
        )
    print(f"CIPHER: registered + ran {len(run_records)} tasks.")


# ---------------------------------------------------------------------------
# Local preview: score stubs without touching the kbench SDK
# ---------------------------------------------------------------------------

def preview(instances, limit=5):
    from scripts.evaluate import stub_greedy  # reuse existing stub
    print(f"--- Local preview ({limit} instances, stub-greedy) ---")
    for rec in instances[:limit]:
        inst = _instance_from_record(rec)
        raw = stub_greedy(inst)
        resp = validate_response(raw)
        best = rec["hidden"].get("oracle_best")
        worst = rec["hidden"].get("oracle_worst")
        bd = score_response(resp, inst, best_obj=best, worst_obj=worst)
        print(f"  {rec['id']}  composite={bd.composite:.3f}  "
              f"obj={bd.objective:.2f}  cal={bd.calibration:.2f}  "
              f"att={bd.attention:.2f}  exe={bd.executive:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/instances.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--preview", action="store_true",
                    help="Score stubs locally without invoking kbench.")
    args = ap.parse_args()

    instances = _load_instances(args.data)
    print(f"Loaded {len(instances)} instances from {args.data}")

    if args.preview:
        preview(instances, limit=args.limit or 5)
        return

    register_and_run(instances, limit=args.limit)


if __name__ == "__main__":
    main()
