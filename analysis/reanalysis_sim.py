"""Simulation helpers for the executive re-analyses (2.1-2.3, 2.7).

All functions mirror cipher/scorer.py exactly unless the name says otherwise;
`decompose_executive` is checked against the stored executive score.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Callable, Dict, List, Optional

from reanalysis_common import ROOT

sys.path.insert(0, ROOT)
from cipher.generator import Instance  # noqa: E402
from cipher.schema import ParsedResponse, validate_response  # noqa: E402
from cipher.world import Action, Effect, Rule, World  # noqa: E402

H = 7  # horizon; asserted per instance below


def combined(probes: List[Action], plan: List[Action], budget: int) -> List[Action]:
    """Probes then plan, capped at the horizon (scorer/simulator convention)."""
    p = probes[:budget]
    return (p + plan[:max(0, budget - len(p))])[:budget]


def norm(raw: float, best: float, worst: float) -> float:
    span = best - worst
    if span <= 0:
        return 1.0
    return max(0.0, min(1.0, (raw - worst) / span))


def objective_of(world: World, actions: List[Action], obj_fn: Optional[Callable] = None) -> int:
    s = world.execute(actions)
    return (obj_fn or world.objective)(s)


def hidden_entities(inst: Instance) -> set:
    """Entities the scorer counts as 'involved in a hidden rule' (trigger.i, effect.target)."""
    out = set()
    for idx in inst.hidden_rule_indices:
        r = inst.world.rules[idx]
        out.add(r.trigger.i)
        out.add(r.effect.target)
    return out


def zero_flux_world(inst: Instance) -> World:
    rules = list(inst.world.rules)
    for idx in inst.hidden_rule_indices:
        o = rules[idx]
        rules[idx] = Rule(o.name, o.trigger, Effect(kind="zero_flux", target=o.effect.target,
                                                    delta=0, source=o.effect.source))
    return World(initial=inst.world.initial, rules=tuple(rules), horizon=inst.world.horizon)


def world_with_hidden_effects(inst: Instance, effects: Dict[int, Effect]) -> World:
    rules = list(inst.world.rules)
    for idx, eff in effects.items():
        o = rules[idx]
        rules[idx] = Rule(o.name, o.trigger, eff)
    return World(initial=inst.world.initial, rules=tuple(rules), horizon=inst.world.horizon)


def contingency_bucket_score(alt_obj: float, final_obj: float, span: float,
                             has_alt: bool, alt_differs: bool) -> float:
    """Scorer's B rule, parameterised on the world the two plans were run in."""
    if not has_alt:
        return 0.0
    if alt_obj > final_obj:
        return min(1.0, 0.5 + (alt_obj - final_obj) / span)
    if alt_obj == final_obj and alt_differs:
        return 0.3
    return 0.1


def bucket_label(b: float) -> str:
    if b == 0.0:
        return "0"
    if abs(b - 0.1) < 1e-12:
        return "0.1"
    if abs(b - 0.3) < 1e-12:
        return "0.3"
    return "0.5-1.0"


_HLABEL = re.compile(r"\bH\s*(\d+)\b", re.IGNORECASE)


def names_hidden_rule(resp: ParsedResponse, inst: Instance) -> bool:
    """Contingency target names a valid hidden-law label (H0..H{n-1}) or a hidden rule's R-name."""
    s = resp.self_judgment.alternative_unknown or ""
    n_hidden = len(inst.hidden_rule_indices)
    for m in _HLABEL.finditer(s):
        if int(m.group(1)) < n_hidden:
            return True
    hidden_names = {inst.world.rules[i].name for i in inst.hidden_rule_indices}
    return any(re.search(rf"\b{re.escape(nm)}\b", s) for nm in hidden_names)


def decompose_executive(resp: ParsedResponse, inst: Instance, best: int, worst: int) -> Dict:
    assert inst.world.horizon == H
    budget = inst.world.horizon
    span = max(1, best - worst)
    probes = resp.exploratory_actions
    hid = hidden_entities(inst)

    a_has = bool(probes)
    a_hidden = a_has and any(a.kind == "observe" and a.i in hid for a in probes)
    a_budget = a_has and len(probes) <= budget // 2
    A = min(1.0, 0.4 * a_has + 0.4 * a_hidden + 0.2 * a_budget)

    adv = zero_flux_world(inst)
    final_c = combined(probes, resp.final_plan, budget)
    alt = resp.self_judgment.alternative_plan
    alt_c = combined(probes, alt, budget) if alt else final_c
    f_adv = objective_of(adv, final_c)
    a_adv = objective_of(adv, alt_c)
    alt_differs = [x.__dict__ for x in alt] != [x.__dict__ for x in resp.final_plan]
    B = contingency_bucket_score(a_adv, f_adv, span, bool(alt), alt_differs)

    full = inst.world
    obj_actual = norm(objective_of(full, final_c), best, worst)
    obj_probe_free = norm(objective_of(full, resp.final_plan[:budget]), best, worst)

    return {
        "A": A, "A_has_probe": a_has, "A_hidden_entity": a_hidden, "A_budget_le3": a_budget,
        "B": B, "B_bucket": bucket_label(B), "has_contingency": bool(alt),
        "contingency_names_hidden": names_hidden_rule(resp, inst),
        "n_probes": len(probes), "n_probes_effective": len(probes[:budget]),
        "n_final": len(resp.final_plan),
        "final_truncated": len(probes[:budget]) + len(resp.final_plan) > budget,
        "executive": 0.5 * A + 0.5 * B,
        "objective": obj_actual, "objective_probe_free": obj_probe_free,
        "final_adv_raw": f_adv, "alt_adv_raw": a_adv,
    }


def stub_responses() -> Dict[str, Callable]:
    """Deterministic stub agents from scripts/evaluate.py (no API calls)."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import evaluate  # noqa: E402
    return {name: evaluate.AGENTS[name] for name in
            ["stub-noop", "stub-greedy", "stub-random", "stub-cautious", "stub-probe-heavy"]}


def parse_stub(raw: dict) -> ParsedResponse:
    return validate_response(raw)
