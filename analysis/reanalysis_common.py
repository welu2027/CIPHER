"""Shared loaders for the ARR re-analysis (analysis/run_reanalysis.py).

Everything here works offline from saved artifacts:
  data/instances.jsonl       - 1,000 instances with hidden ground truth
  results/models/*.json      - Kaggle Benchmarks exports, one subrun per saved
                               model response (prompt + raw reply + stored scores)

No LLM API is ever called.

Instance matching: each subrun is matched to its instance by exact prompt text,
NOT by position. Kaggle subruns are not stored in dataset order, so positional
alignment (as used by analysis/clean_and_rebuild.py) mislabels difficulty for
~35-55% of subruns.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)

from cipher.generator import Instance  # noqa: E402
from cipher.world import World, State, EntityState, Rule, Trigger, Effect  # noqa: E402
from cipher.schema import validate_response, ParsedResponse  # noqa: E402
from cipher.scorer import score_response  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "instances.jsonl")
MODELS_DIR = os.path.join(ROOT, "results", "models")
BASELINES_DIR = os.path.join(ROOT, "results", "baselines")
OUT_DIR = os.path.join(HERE, "out")
CACHE_DIR = os.path.join(OUT_DIR, "cache")

DIMS = ["composite", "objective", "calibration", "attention", "executive"]
WEIGHTS = {"objective": 0.35, "calibration": 0.25, "attention": 0.20, "executive": 0.20}

# Display name -> (file, family). Order = paper Table 5 order is NOT assumed.
MODELS: Dict[str, tuple] = {
    "Claude Opus 4.6":          ("claude_opus_46.json", "Claude"),
    "Claude Opus 4.5":          ("claude_opus_45.json", "Claude"),
    "Claude Sonnet 4.6":        ("claude_sonnet_46.json", "Claude"),
    "Claude Opus 4.7":          ("claude_opus_47.json", "Claude"),
    "Claude 4.5 Haiku":         ("claude_haiku_45.json", "Claude"),
    "Gemini 2.5 Pro":           ("gemini_25_pro.json", "Gemini"),
    "Gemini 2.5 Flash":         ("gemini_25_flash.json", "Gemini"),
    "Gemini 3 Flash Preview":   ("gemini_flash_preview.json", "Gemini"),
    "Gemini 3.1 Pro Preview":   ("gemini_pro_preview.json", "Gemini"),
    "GPT-5.4 Nano":             ("gpt_54_nano.json", "GPT"),
    "GPT-5.4 mini":             ("gpt_54_mini.json", "GPT"),
    "GPT-5.4":                  ("gpt_54.json", "GPT"),
    "GPT-5.5":                  ("gpt_55.json", "GPT"),
    "Qwen 3 Next 80B Instruct": ("qwen_3_next_80b_instruct.json", "Qwen"),
    "DeepSeek V3.2":            ("deepseek_v3_2.json", "DeepSeek"),
    "Gemma 4 31B":              ("gemma_4_31b.json", "Gemma"),
}

STUBS: Dict[str, str] = {
    "stub-noop": "noop.json",
    "stub-greedy": "greedy.json",
    "stub-random": "random.json",
    "stub-cautious": "cautious.json",
    "stub-probe-heavy": "probe_heavy.json",
}


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

def load_records() -> List[Dict[str, Any]]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def instance_from_record(rec: Dict[str, Any]) -> Instance:
    """Identical to the rehydration in notebooks/kaggle_benchmark.ipynb."""
    h = rec["hidden"]
    rules = []
    for r in h["rules"]:
        t, e = r["trigger"], r["effect"]
        rules.append(Rule(
            name=r["name"],
            trigger=Trigger(kind=t["kind"], i=t["i"], j=t.get("j", -1), k=t.get("k", 0)),
            effect=Effect(kind=e["kind"], target=e["target"],
                          delta=e.get("delta", 0), source=e.get("source", -1)),
        ))
    initial = State(tuple(EntityState(phase=s["phase"], flux=s["flux"]) for s in h["initial_state"]))
    return Instance(
        id=rec["id"], seed=rec["seed"], difficulty=rec["difficulty"],
        world=World(initial=initial, rules=tuple(rules), horizon=h["horizon"]),
        visible_rule_indices=h["visible_rule_indices"],
        hidden_rule_indices=h["hidden_rule_indices"],
        public_rule_descriptions=[],
        hidden_fields=h.get("hidden_fields", []),
        metacog_ground_truth=h["metacog_ground_truth"],
        true_unknown_ranking=h["true_unknown_ranking"],
        oracle_objective=h.get("oracle_best"),
    )


# ---------------------------------------------------------------------------
# Response parsing - verbatim port of the notebook's _score() front end
# ---------------------------------------------------------------------------

def extract_outermost_json(text: str) -> str:
    start = text.find("{")
    if start == -1:
        return ""
    depth, in_string, escape = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def parse_json_tolerant(text: str):
    cleaned = re.sub(r'//[^\n]*', '', text)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(cleaned.replace("'", '"'))


def parse_reply(text: str) -> tuple[Optional[ParsedResponse], Optional[dict], str]:
    """Return (parsed, raw_json, status). status in {ok, no_json, json_error}."""
    blob = extract_outermost_json(text)
    if not blob:
        return None, None, "no_json"
    try:
        raw = parse_json_tolerant(blob)
    except Exception:
        return None, None, "json_error"
    if not isinstance(raw, dict):
        return None, None, "json_error"
    try:
        return validate_response(raw), raw, "ok"
    except Exception:
        return None, raw, "schema_exception"


# ---------------------------------------------------------------------------
# Saved model responses
# ---------------------------------------------------------------------------

@dataclass
class Saved:
    model: str
    instance_id: str
    reply: str
    stored: Dict[str, Any]      # dictResult as exported by Kaggle (scores at run time)
    prompt_mojibake: bool       # prompt text was cp1252-mangled UTF-8


def _texts(subrun) -> tuple[str, str]:
    contents = subrun["conversations"][0]["requests"][0]["contents"]
    user = "".join(p.get("text", "") for c in contents if c["role"].endswith("USER") for p in c["parts"])
    asst = "".join(p.get("text", "") for c in contents if not c["role"].endswith("USER") for p in c["parts"])
    return user, asst


def _demojibake(t: str) -> Optional[str]:
    try:
        return t.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None


def load_model_file(name: str, prompt_to_id: Dict[str, str]) -> Dict[str, Any]:
    fname, _ = MODELS[name]
    with open(os.path.join(MODELS_DIR, fname), encoding="utf-8") as f:
        d = json.load(f)
    saved: Dict[str, Saved] = {}
    unmatched = 0
    dupes = 0
    for s in d.get("subruns", []):
        if not s.get("conversations"):
            unmatched += 1
            continue
        user, asst = _texts(s)
        moj = False
        iid = prompt_to_id.get(user)
        if iid is None:
            fixed = _demojibake(user)
            iid = prompt_to_id.get(fixed) if fixed else None
            moj = iid is not None
        if iid is None:
            unmatched += 1
            continue
        if iid in saved:
            dupes += 1
        saved[iid] = Saved(name, iid, asst, s["results"][0].get("dictResult", {}), moj)
    agg = None
    for r in d.get("results", []):
        if "numericResult" in r:
            agg = r["numericResult"].get("value")
    return {
        "saved": saved, "unmatched": unmatched, "dupes": dupes,
        "n_subruns": len(d.get("subruns", [])),
        "kaggle_aggregate": agg,
        "slug": d.get("modelVersion", {}).get("slug"),
        "task_name": d.get("taskVersion", {}).get("name"),
        "start": d.get("startTime"),
    }


def load_all_models(use_cache: bool = True) -> Dict[str, Dict[str, Any]]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, "models.pkl")
    if use_cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    records = load_records()
    p2id = {r["prompt"]: r["id"] for r in records}
    out = {name: load_model_file(name, p2id) for name in MODELS}
    with open(cache, "wb") as f:
        pickle.dump(out, f)
    return out


def load_stub(name: str) -> Dict[str, Any]:
    with open(os.path.join(BASELINES_DIR, STUBS[name]), encoding="utf-8") as f:
        return json.load(f)


def ensure_out(*parts: str) -> str:
    p = os.path.join(OUT_DIR, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        open(path, "w").close()
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
