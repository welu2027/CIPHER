"""LLM baselines for CIPHER using OpenAI API.

Three baselines, each run on all 1000 instances:

  oracle  — full-info: model sees ALL rules (no hidden laws). Upper-bounds
             the objective dimension. Shows the information gap.

  cot     — chain-of-thought: standard hidden-rule prompt + CoT instruction
             prepended to the system turn. Same JSON output format.

  react   — agentic loop: model can call probe_world(actions) to execute
             actions against the real world state and observe resulting entity
             values before committing to a final plan.

Output: one JSON file per baseline, in analysis/models/ format so that
clean_and_rebuild.py and offline_analysis.py can consume them.

Usage:
  # Set OPENAI_API_KEY in env or .env file, then:
  python scripts/llm_baselines.py --baseline oracle --model gpt-4o
  python scripts/llm_baselines.py --baseline cot    --model gpt-4o
  python scripts/llm_baselines.py --baseline react  --model gpt-4o

  # Limit to N instances for testing:
  python scripts/llm_baselines.py --baseline oracle --model gpt-4o --limit 10
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import replace
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _v = _line.split("=", 1)
        _k = _k.strip().lstrip("$").removeprefix("env:")  # handle $env:KEY syntax
        _v = _v.strip().strip('"').strip("'")
        os.environ.setdefault(_k, _v)

from cipher.generator import Instance, generate_instance
from cipher.prompt import build_prompt, SCHEMA_BLOCK
from cipher.schema import validate_response
from cipher.scorer import score_response
from cipher.flavor import pick_flavor, describe_rule
from cipher.world import Action, State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_instances(data_path: str, limit: int | None) -> List[Dict]:
    with open(data_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    if limit:
        records = records[:limit]
    return records


def _rehydrate(rec: Dict) -> Instance:
    """Reconstruct a full Instance from instances.jsonl record (seed + difficulty)."""
    return generate_instance(rec["seed"], rec["difficulty"])


def _extract_json(text: str) -> Dict:
    """Pull the outermost JSON object from a model reply."""
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _openai_chat(client, model: str, messages: List[Dict], tools=None,
                 temperature: float = 0.0, max_tokens: int = 3000) -> Any:
    """Single OpenAI chat completion with simple retry on rate-limit."""
    kwargs = dict(model=model, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            name = type(e).__name__
            if "RateLimitError" in name or "rate" in str(e).lower():
                wait = 2 ** attempt
                print(f"  rate-limit, retry in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("OpenAI API failed after 5 retries")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_oracle_prompt(inst: Instance) -> str:
    """Standard prompt but with ALL rules visible (no hidden laws)."""
    # Swap to an all-visible instance so build_prompt shows all rules.
    n_rules = len(inst.world.rules)
    oracle_inst = replace(
        inst,
        visible_rule_indices=list(range(n_rules)),
        hidden_rule_indices=[],
    )
    return build_prompt(oracle_inst)


COT_PREFIX = (
    "Before outputting your JSON response, reason carefully inside a "
    "<think>...</think> block. Work through: (1) what the visible rules "
    "imply about entity dynamics, (2) what the hidden rules could plausibly "
    "do given the system state, (3) which action sequences are most likely "
    "to raise the objective, and (4) what your alternative plan should be. "
    "After closing </think>, output ONLY the JSON with no other text.\n\n"
)

def build_cot_prompt(inst: Instance) -> str:
    return COT_PREFIX + build_prompt(inst)


REACT_SYSTEM = (
    "You are an agent exploring an unknown physical system. "
    "Before committing to a final plan you may call probe_world() "
    "to simulate actions and observe the resulting entity states. "
    "Each call costs actions from your horizon budget. "
    "When you are ready, output the final JSON response (and nothing else)."
)

PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "probe_world",
        "description": (
            "Execute a sequence of actions starting from the current simulation "
            "state and return the resulting entity values. Each action costs one "
            "step of your horizon budget."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "description": "List of action objects to execute.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string",
                                     "enum": ["pulse","damp","shift","unshift",
                                              "align","observe","wait"]},
                            "i": {"type": "integer", "description": "primary entity index"},
                            "j": {"type": "integer", "description": "secondary entity index (align only)"},
                        },
                        "required": ["kind"],
                    },
                }
            },
            "required": ["actions"],
        },
    },
}


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def run_oracle(client, model: str, inst: Instance) -> Dict:
    prompt = build_oracle_prompt(inst)
    resp   = _openai_chat(client, model, [{"role": "user", "content": prompt}])
    return _extract_json(resp.choices[0].message.content or "")


def run_cot(client, model: str, inst: Instance) -> Dict:
    prompt = build_cot_prompt(inst)
    resp   = _openai_chat(client, model,
                          [{"role": "user", "content": prompt}],
                          max_tokens=6000)
    text   = resp.choices[0].message.content or ""
    # Strip <think>...</think> block before parsing JSON
    if "<think>" in text and "</think>" in text:
        text = text[text.index("</think>") + len("</think>"):]
    return _extract_json(text)


def run_react(client, model: str, inst: Instance) -> Dict:
    """ReAct loop: model probes the world via tool calls, then outputs JSON."""
    horizon  = inst.world.horizon
    sim_state: State = inst.world.initial
    budget_used      = 0
    MAX_PROBE_CALLS  = 3

    messages = [
        {"role": "system", "content": REACT_SYSTEM},
        {"role": "user",   "content": build_prompt(inst)},
    ]

    for _ in range(MAX_PROBE_CALLS + 1):
        remaining = horizon - budget_used
        if remaining <= 0:
            break

        resp = _openai_chat(client, model, messages,
                            tools=[PROBE_TOOL], max_tokens=4000)
        choice = resp.choices[0]
        msg    = choice.message

        # Model wants to probe
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            # Append the assistant message with tool_calls (required by OpenAI API)
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                try:
                    args    = json.loads(tc.function.arguments)
                    actions = args.get("actions", [])
                except (json.JSONDecodeError, AttributeError):
                    actions = []

                # Execute each action on the running sim state
                entity_snapshots = []
                for act_obj in actions:
                    if budget_used >= horizon:
                        break
                    kind = act_obj.get("kind", "wait")
                    i    = int(act_obj.get("i", 0))
                    j    = int(act_obj.get("j", -1))
                    action = Action(kind=kind, i=i, j=j)
                    sim_state = action.apply(sim_state)
                    sim_state = inst.world.step(sim_state)
                    budget_used += 1

                for idx, e in enumerate(sim_state.entities):
                    entity_snapshots.append({
                        "entity": idx, "phase": e.phase, "flux": e.flux
                    })

                tool_result = {
                    "actions_executed": len(actions),
                    "budget_remaining": horizon - budget_used,
                    "entity_states": entity_snapshots,
                }
                messages.append({
                    "role":        "tool",
                    "tool_call_id": tc.id,
                    "content":     json.dumps(tool_result),
                })
        else:
            # Model is done — extract JSON from content
            return _extract_json(msg.content or "")

    # Final call without tools to force JSON output
    messages.append({
        "role": "user",
        "content": (
            f"You have used {budget_used} actions probing. "
            f"{horizon - budget_used} actions remain for your final plan. "
            "Now output your JSON response."
        ),
    })
    final = _openai_chat(client, model, messages, max_tokens=3000)
    return _extract_json(final.choices[0].message.content or "")


# ---------------------------------------------------------------------------
# Scoring & subrun format
# ---------------------------------------------------------------------------

def _score_to_subrun(dr: Dict) -> Dict:
    """Wrap a dictResult in the kbench subrun envelope format."""
    return {"results": [{"dictResult": dr}]}


def run_baseline(baseline: str, model: str, data_path: str,
                 out_path: str, limit: int | None) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client  = OpenAI(api_key=api_key)
    records = _load_instances(data_path, limit)
    n       = len(records)
    print(f"Running baseline='{baseline}' model='{model}' on {n} instances...")

    agent_fn = {"oracle": run_oracle, "cot": run_cot, "react": run_react}[baseline]

    subruns    = []
    n_fail     = 0
    score_sums = {"composite": 0.0, "objective": 0.0, "calibration": 0.0,
                  "attention": 0.0, "executive": 0.0}

    for idx, rec in enumerate(records):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  {idx+1}/{n}  failures={n_fail}", flush=True)

        inst = _rehydrate(rec)

        try:
            raw  = agent_fn(client, model, inst)
            resp = validate_response(raw)
        except Exception as e:
            print(f"  [!] instance {idx} error: {e}", flush=True)
            # Store a _ZERO-style failure subrun (no best_objective key)
            subruns.append(_score_to_subrun({
                "composite": 0.0, "objective": 0.0, "calibration": 0.0,
                "attention": 0.0, "executive": 0.0,
                "parse_errors": 99,
            }))
            n_fail += 1
            continue

        bd = score_response(resp, inst)
        dr = bd.to_dict()
        dr["difficulty"] = rec["difficulty"]

        for dim in score_sums:
            score_sums[dim] += dr[dim]

        subruns.append(_score_to_subrun(dr))

    n_valid = n - n_fail
    print(f"\nDone. {n_valid}/{n} valid, {n_fail} failures.")
    if n_valid:
        print("Scores (valid only):")
        for dim, total in score_sums.items():
            print(f"  {dim:<14} {total/n_valid:.4f}")

    out = {"baseline": baseline, "model": model, "n_instances": n, "subruns": subruns}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\nSaved {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_OUTFILES = {
    "oracle": "oracle_{model}.json",
    "cot":    "cot_{model}.json",
    "react":  "react_{model}.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["oracle", "cot", "react"])
    ap.add_argument("--model",    default="gpt-4o",
                    help="OpenAI model string, e.g. gpt-4o, gpt-4o-mini, o3")
    ap.add_argument("--data",     default=None,
                    help="Path to instances.jsonl (auto-detected if omitted)")
    ap.add_argument("--out",      default=None,
                    help="Output JSON path (default: analysis/models/<baseline>_<model>.json)")
    ap.add_argument("--limit",    type=int, default=None,
                    help="Only run first N instances (for testing)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)

    if args.data is None:
        for cand in [os.path.join(repo_root, "data", "instances.jsonl"),
                     os.path.join(repo_root, "..", "data", "instances.jsonl")]:
            if os.path.exists(cand):
                args.data = cand
                break
    if args.data is None or not os.path.exists(args.data):
        print("ERROR: could not find instances.jsonl. Pass --data explicitly.", file=sys.stderr)
        sys.exit(1)

    if args.out is None:
        safe_model = args.model.replace("/", "_").replace("-", "_")
        fname      = _OUTFILES[args.baseline].format(model=safe_model)
        args.out   = os.path.join(repo_root, "analysis", "models", fname)

    run_baseline(args.baseline, args.model, args.data, args.out, args.limit)


if __name__ == "__main__":
    main()
