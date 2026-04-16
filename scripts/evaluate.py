"""Run a model (or a stub baseline) over the benchmark and compute scores.

The `--model` flag selects which agent to use:
  stub-random   -> emits random-but-valid JSON (lower-bound baseline)
  stub-noop     -> emits an empty final_plan (degenerate floor baseline)
  stub-greedy   -> runs a local beam search assuming no hidden rules

Real model hookups (Claude/Gemini/etc) can be added as new entries in the
AGENTS registry below; each agent is a function (instance_record) -> dict.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env if present so HF_TOKEN etc. are available without manual export.
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

from cipher import generate_instance  # noqa
from cipher.generator import Instance
from cipher.world import (
    Action, World, State, EntityState, Rule, Trigger, Effect,
)
from cipher.simulator import run_plan
from cipher.schema import validate_response
from cipher.scorer import score_response
from cipher.optimal import oracle_score


def _instance_from_record(rec: Dict[str, Any]) -> Instance:
    """Rehydrate an Instance from a generated JSONL record. Rules are rebuilt
    from the `hidden.rules` ground truth; the public prompt already hides
    what should be hidden, so the World can carry full semantics safely."""
    hidden = rec["hidden"]
    rules = []
    hidden_rules_lookup = {h["rule_name"]: set(h["hidden"])
                           for h in hidden.get("hidden_fields", [])}
    for r in hidden["rules"]:
        name = r["name"]
        hides = hidden_rules_lookup.get(name, set())
        t_raw = r["trigger"]
        e_raw = r["effect"]
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
        rules.append(Rule(name=name, trigger=trigger, effect=effect))

    initial = State(tuple(EntityState(phase=e["phase"], flux=e["flux"])
                          for e in hidden["initial_state"]))
    world = World(initial=initial, rules=tuple(rules), horizon=hidden["horizon"])
    return Instance(
        id=rec["id"], seed=rec["seed"], difficulty=rec["difficulty"],
        world=world, public_rule_descriptions=[], hidden_fields=hidden.get("hidden_fields", []),
        metacog_ground_truth=hidden["metacog_ground_truth"],
        true_unknown_ranking=hidden["true_unknown_ranking"],
        oracle_objective=hidden.get("oracle_best"),
    )


# ---------- Agents (stubs) ----------

def _all_claims_for(inst: Instance) -> List[Dict[str, Any]]:
    claims = []
    for gt in inst.metacog_ground_truth:
        claims.append({
            "rule_name": gt["rule_name"],
            "component": gt["component"],
            "known": True,   # stub baseline: naively claims everything is known
            "confidence": 0.5,
        })
    return claims


def stub_noop(inst: Instance) -> Dict[str, Any]:
    return {
        "metacog_assessment": _all_claims_for(inst),
        "critical_unknowns_ranked": [],
        "exploratory_actions": [],
        "final_plan": [{"kind": "wait"}],
        "self_judgment": {"robustness_score": 50, "risks_identified": [],
                          "alternative_if_unknown_X": {}},
    }


def stub_random(inst: Instance) -> Dict[str, Any]:
    rng = random.Random(inst.seed ^ 0xdeadbeef)
    n = inst.world.initial.n
    kinds = ["pulse", "damp", "shift", "unshift", "observe", "wait"]
    def rand_act():
        k = rng.choice(kinds)
        return {"kind": k, "i": rng.randrange(n)}
    probes = [rand_act() for _ in range(rng.randint(0, 2))]
    plan = [rand_act() for _ in range(rng.randint(1, 5))]
    return {
        "metacog_assessment": [
            {"rule_name": gt["rule_name"], "component": gt["component"],
             "known": rng.random() > 0.5, "confidence": rng.random()}
            for gt in inst.metacog_ground_truth
        ],
        "critical_unknowns_ranked": list(inst.true_unknown_ranking[::-1]),  # reversed on purpose
        "exploratory_actions": probes,
        "final_plan": plan,
        "self_judgment": {"robustness_score": 40, "risks_identified": ["rng"],
                          "alternative_if_unknown_X": {"unknown": "", "plan": [rand_act()]}},
    }


def stub_greedy(inst: Instance) -> Dict[str, Any]:
    """Assumes rules are as-stated and runs the oracle search on THAT world.
    This is a decent upper-bound-when-lucky baseline: it will do well when
    hidden parts don't matter much and poorly when they do."""
    _, best_plan = oracle_score(inst.world)
    plan_objs = [{"kind": a.kind, "i": a.i, "j": a.j} for a in best_plan]
    # claim everything is known with high confidence (naive; this is what
    # under-confident metacognition looks like)
    mc = [{"rule_name": gt["rule_name"], "component": gt["component"],
           "known": True, "confidence": 0.9}
          for gt in inst.metacog_ground_truth]
    return {
        "metacog_assessment": mc,
        "critical_unknowns_ranked": [],
        "exploratory_actions": [],
        "final_plan": plan_objs,
        "self_judgment": {"robustness_score": 70,
                          "risks_identified": [],
                          "alternative_if_unknown_X": {}},
    }


def claude_agent(inst: Instance) -> Dict[str, Any]:
    """Real LLM agent using the Anthropic API. Requires ANTHROPIC_API_KEY
    in the environment and `pip install anthropic`. Model id is read from
    IF_CLAUDE_MODEL (default: claude-opus-4-6)."""
    import anthropic
    from cipher import build_prompt
    client = anthropic.Anthropic()
    prompt = build_prompt(inst)
    model = os.environ.get("IF_CLAUDE_MODEL", "claude-opus-4-6")
    msg = client.messages.create(
        model=model, max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    # be forgiving: extract the outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {"metacog_assessment": [], "critical_unknowns_ranked": [],
                "exploratory_actions": [], "final_plan": [],
                "self_judgment": {"robustness_score": 0, "risks_identified": [],
                                  "alternative_if_unknown_X": {}}}
    return json.loads(text[start:end + 1])


def hf_agent(inst: Instance) -> Dict[str, Any]:
    """LLM agent via Hugging Face Inference Providers. Requires HF_TOKEN in
    env (loaded from .env if present). Model id from HF_MODEL. Optional
    HF_PROVIDER pins a specific provider; blank => HF auto-route."""
    from huggingface_hub import InferenceClient
    from cipher import build_prompt
    token = os.environ["HF_TOKEN"]
    model = os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
    provider = os.environ.get("HF_PROVIDER") or "auto"
    client = InferenceClient(model=model, token=token, provider=provider)
    prompt = build_prompt(inst)
    resp = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048, temperature=0.0,
    )
    text = resp.choices[0].message.content
    start = text.find("{"); end = text.rfind("}")
    if start == -1 or end == -1:
        return {"metacog_assessment": [], "critical_unknowns_ranked": [],
                "exploratory_actions": [], "final_plan": [],
                "self_judgment": {"robustness_score": 0, "risks_identified": [],
                                  "alternative_if_unknown_X": {}}}
    return json.loads(text[start:end + 1])


def gemini_agent(inst: Instance) -> Dict[str, Any]:
    """Real LLM agent using Google Generative AI. Requires GOOGLE_API_KEY
    and `pip install google-generativeai`. Model id from IF_GEMINI_MODEL."""
    import google.generativeai as genai
    from cipher import build_prompt
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(os.environ.get("IF_GEMINI_MODEL", "gemini-2.5-pro"))
    resp = model.generate_content(build_prompt(inst))
    text = resp.text
    start = text.find("{"); end = text.rfind("}")
    return json.loads(text[start:end + 1]) if start != -1 else {}


AGENTS = {
    "stub-noop": stub_noop,
    "stub-random": stub_random,
    "stub-greedy": stub_greedy,
    "claude": claude_agent,
    "gemini": gemini_agent,
    "hf": hf_agent,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="stub-greedy", choices=list(AGENTS))
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--leaderboard", default="leaderboard.json",
                    help="Aggregate file; each run appends one row.")
    ap.add_argument("--label", default=None,
                    help="Optional label for this run (e.g. HF_MODEL string).")
    args = ap.parse_args()

    agent = AGENTS[args.model]
    results = []

    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]

    try:
        from tqdm import tqdm
        iterator = tqdm(records, desc=f"eval[{args.model}]", unit="inst")
    except ImportError:
        iterator = records
        print(f"(install tqdm for progress bars: pip install tqdm)", file=sys.stderr)

    for rec in iterator:
        inst = _instance_from_record(rec)
        raw = agent(inst)
        resp = validate_response(raw)
        best = rec["hidden"].get("oracle_best")
        worst = rec["hidden"].get("oracle_worst")
        breakdown = score_response(resp, inst, best_obj=best, worst_obj=worst)
        results.append({"id": inst.id, "difficulty": inst.difficulty,
                        **breakdown.to_dict()})

    avg = lambda key: sum(r[key] for r in results) / len(results)
    summary = {
        "model": args.model,
        "n": len(results),
        "mean_composite": avg("composite"),
        "mean_objective": avg("objective"),
        "mean_calibration": avg("calibration"),
        "mean_attention": avg("attention"),
        "mean_executive": avg("executive"),
    }
    out = {"summary": summary, "per_instance": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    # --- Append to cross-run leaderboard ---
    import datetime
    label = args.label or (
        os.environ.get("HF_MODEL") if args.model == "hf"
        else os.environ.get("IF_CLAUDE_MODEL") if args.model == "claude"
        else os.environ.get("IF_GEMINI_MODEL") if args.model == "gemini"
        else args.model
    )
    row = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "agent": args.model,
        "label": label,
        "data": os.path.basename(args.data),
        "results_file": args.out,
        **summary,
    }
    lb_path = args.leaderboard
    board = []
    if os.path.exists(lb_path):
        try:
            board = json.load(open(lb_path))
            if not isinstance(board, list):
                board = []
        except Exception:
            board = []
    board.append(row)
    board.sort(key=lambda r: r.get("mean_composite", 0), reverse=True)
    with open(lb_path, "w") as f:
        json.dump(board, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nAppended to {lb_path} ({len(board)} rows total).")
    print("\n--- Leaderboard (by composite) ---")
    print(f"{'agent':<14}{'label':<45}{'n':>5}{'comp':>8}{'obj':>8}{'cal':>8}{'att':>8}{'exe':>8}")
    for r in board:
        print(f"{r['agent']:<14}{(r.get('label') or '-')[:44]:<45}"
              f"{r['n']:>5}{r['mean_composite']:>8.3f}"
              f"{r['mean_objective']:>8.3f}{r['mean_calibration']:>8.3f}"
              f"{r['mean_attention']:>8.3f}{r['mean_executive']:>8.3f}")


if __name__ == "__main__":
    main()
