"""Separate-probe-budget condition (ablation for the shared-action-budget confound).

In the original condition, exploratory probes and the final plan share one
H=7 action budget, and every action (including `observe`) advances the
dynamics. Here the two are decoupled:

  * Up to PROBE_CAP (=3) `observe` probes are taken BEFORE the plan starts.
    They do not change state, do not advance the dynamics (no rules fire),
    and do not count against the horizon.
  * The final plan gets the full horizon (7 actions) from the initial state.

Because probes cannot affect the trajectory, the final plan is scored against
the SAME oracle best/worst bounds as the original benchmark, and objective is
directly comparable across conditions. Non-`observe` exploratory actions are
not executed (they are counted in `invalid_probes`).

Calibration and attention are scored exactly as in cipher/scorer.py. Executive
keeps the same two components and point values; only the probe/plan budget
accounting changes:
  A (probe quality):   +0.4 any probe, +0.4 an observe on an entity in a hidden
                       rule, +0.2 if the number of probes submitted <= PROBE_CAP.
  B (contingency):     final plan vs contingency plan, each run from the
                       initial state in the zero_flux adversarial world;
                       same 0 / 0.1 / 0.3 / 0.5-1.0 buckets as the original.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from .generator import Instance
from .schema import ParsedResponse
from .scorer import _attention, _calibration
from .world import Action, Effect, Rule, World

PROBE_CAP = 3
SCORER_VERSION = "separate-budget-v1"

# --- prompt edits -----------------------------------------------------------
# Applied to the stored prompt text (data/instances.jsonl) so everything except
# the budget wording is byte-identical to what the original models saw.

_BUDGET_OLD = "Probes consume budget.\n"
_BUDGET_NEW = (
    "Exploratory probes do NOT consume this budget: you may additionally issue "
    "up to {cap} observe probes, which are taken before your plan starts, do not "
    "change the system, and do not advance it (no laws fire during probes). "
    "Your final plan then has the full {h}-action budget.\n"
)
_EXPL_OLD = '"exploratory_actions": [ <up to 5 action objects> ],'
_EXPL_NEW = '"exploratory_actions": [ <up to {cap} observe actions; NOT counted against the horizon> ],'
_PLAN_OLD = '"final_plan":          [ <action objects; total actions <= horizon> ],'
_PLAN_NEW = '"final_plan":          [ <action objects; at most {h} actions> ],'
_TOTAL_OLD = "actions total (exploratory + final plan combined)."
_TOTAL_NEW = "actions in your final plan."


def make_separate_budget_prompt(prompt: str, horizon: int = 7, cap: int = PROBE_CAP) -> str:
    edits = [
        (_TOTAL_OLD, _TOTAL_NEW),
        (_BUDGET_OLD, _BUDGET_NEW.format(cap=cap, h=horizon)),
        (_EXPL_OLD, _EXPL_NEW.format(cap=cap)),
        (_PLAN_OLD, _PLAN_NEW.format(h=horizon)),
    ]
    out = prompt
    for old, new in edits:
        n = out.count(old)
        if n != 1:
            raise ValueError(f"expected exactly one occurrence of {old!r}, found {n}")
        out = out.replace(old, new)
    return out


# --- scoring ----------------------------------------------------------------

@dataclass
class SeparateBudgetScore:
    objective: float
    calibration: float
    attention: float
    executive: float
    composite: float
    A: float
    B: float
    raw_objective: int
    best_objective: int
    worst_objective: int
    n_probes_submitted: int
    n_probes_used: int
    invalid_probes: int
    parse_errors: int
    scorer_version: str = SCORER_VERSION

    def to_dict(self) -> Dict:
        return asdict(self)


def _norm(raw: float, best: float, worst: float) -> float:
    span = best - worst
    if span <= 0:
        return 1.0
    return max(0.0, min(1.0, (raw - worst) / span))


def _hidden_entities(inst: Instance) -> set:
    out = set()
    for idx in inst.hidden_rule_indices:
        r = inst.world.rules[idx]
        out.add(r.trigger.i)
        out.add(r.effect.target)
    return out


def _adversarial_world(inst: Instance) -> World:
    rules = list(inst.world.rules)
    for idx in inst.hidden_rule_indices:
        o = rules[idx]
        rules[idx] = Rule(o.name, o.trigger, Effect(kind="zero_flux", target=o.effect.target,
                                                    delta=0, source=o.effect.source))
    return World(initial=inst.world.initial, rules=tuple(rules), horizon=inst.world.horizon)


def score_response_separate(resp: ParsedResponse, inst: Instance, best_obj: int, worst_obj: int,
                            weights: Dict[str, float] | None = None) -> SeparateBudgetScore:
    weights = weights or {"objective": 0.35, "calibration": 0.25, "attention": 0.2, "executive": 0.2}
    h = inst.world.horizon

    probes_all: List[Action] = resp.exploratory_actions
    observes = [a for a in probes_all if a.kind == "observe"]
    used = observes[:PROBE_CAP]
    invalid = len(probes_all) - len(observes)

    plan = resp.final_plan[:h]
    raw = inst.world.objective(inst.world.execute(plan))
    obj = _norm(raw, best_obj, worst_obj)

    hid = _hidden_entities(inst)
    A = 0.0
    if probes_all:
        A += 0.4
        if any(a.i in hid for a in used):
            A += 0.4
        if len(probes_all) <= PROBE_CAP:
            A += 0.2
    A = min(1.0, A)

    adv = _adversarial_world(inst)
    span = max(1, best_obj - worst_obj)
    alt = resp.self_judgment.alternative_plan
    if not alt:
        B = 0.0
    else:
        f_adv = adv.objective(adv.execute(plan))
        a_adv = adv.objective(adv.execute(alt[:h]))
        if a_adv > f_adv:
            B = min(1.0, 0.5 + (a_adv - f_adv) / span)
        elif a_adv == f_adv and [x.__dict__ for x in alt] != [x.__dict__ for x in resp.final_plan]:
            B = 0.3
        else:
            B = 0.1
    ex = 0.5 * A + 0.5 * B

    cal = _calibration(resp, inst)
    att = _attention(resp, inst)
    comp = (weights["objective"] * obj + weights["calibration"] * cal
            + weights["attention"] * att + weights["executive"] * ex)
    return SeparateBudgetScore(
        objective=obj, calibration=cal, attention=att, executive=ex, composite=comp,
        A=A, B=B, raw_objective=raw, best_objective=best_obj, worst_objective=worst_obj,
        n_probes_submitted=len(probes_all), n_probes_used=len(used), invalid_probes=invalid,
        parse_errors=len(resp.errors),
    )
