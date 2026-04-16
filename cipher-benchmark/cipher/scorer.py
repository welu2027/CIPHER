"""Multi-dimensional scoring.

Four sub-scores, all normalized to [0, 1]:

1. objective    — (agent_obj - worst) / (best - worst), clamped to [0, 1].
2. calibration  — 1 - mean Brier score over metacognitive claims.
3. attention    — Spearman-like rank overlap between the model's
                  critical_unknowns_ranked and the true impact ranking.
4. executive    — rule-based plan-quality heuristics.

Composite = weighted mean (default equal weights).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from .generator import Instance
from .schema import ParsedResponse
from .simulator import run_plan, run_actions
from .optimal import oracle_score
from .world import Action


@dataclass
class ScoreBreakdown:
    objective: float
    calibration: float
    attention: float
    executive: float
    composite: float
    raw_objective: int
    best_objective: int
    worst_objective: int
    parse_errors: int

    def to_dict(self) -> Dict:
        return asdict(self)


def _worst_objective(world, beam_width: int = 32) -> int:
    """Estimate worst-case objective by beam-searching on negated objective."""
    from .world import all_actions
    actions = all_actions(world.initial.n)
    beam = [(world.initial, [], world.objective(world.initial))]
    worst = world.objective(world.initial)
    for _ in range(world.horizon):
        cand = []
        for state, plan, _ in beam:
            for a in actions:
                s2 = a.apply(state)
                s2 = world.step(s2)
                cand.append((s2, plan + [a], world.objective(s2)))
        cand.sort(key=lambda x: x[2])
        beam = cand[:beam_width]
        if beam and beam[0][2] < worst:
            worst = beam[0][2]
    return worst


def _calibration(resp: ParsedResponse, inst: Instance) -> float:
    gt_index = {(g["rule_name"], g["component"]): g["true_known"]
                for g in inst.metacog_ground_truth}
    if not gt_index:
        return 1.0
    squared_errors: List[float] = []
    for claim in resp.metacog_assessment:
        key = (claim.rule_name, claim.component)
        if key not in gt_index:
            continue
        truth = 1.0 if gt_index[key] else 0.0
        # the stated probability that the component is known = confidence if
        # claim.known is True, else (1 - confidence).
        p_known = claim.confidence if claim.known else 1.0 - claim.confidence
        squared_errors.append((p_known - truth) ** 2)
    # missing claims are penalized as a Brier of 0.25 (max-uncertainty answer)
    answered = {(c.rule_name, c.component) for c in resp.metacog_assessment}
    missing = len([k for k in gt_index if k not in answered])
    squared_errors += [0.25] * missing
    if not squared_errors:
        return 1.0
    brier = sum(squared_errors) / len(squared_errors)
    return max(0.0, 1.0 - brier)  # Brier in [0,1]; 1 - brier maps to [0,1]


def _attention(resp: ParsedResponse, inst: Instance) -> float:
    truth = inst.true_unknown_ranking
    if not truth:
        return 1.0
    claimed = resp.critical_unknowns_ranked
    if not claimed:
        return 0.0
    # compute weighted rank correlation: reward for each correctly-ranked pair
    # normalized by number of pairs.
    truth_rank = {name: idx for idx, name in enumerate(truth)}
    # keep only claimed entries that appear in truth
    filtered = [c for c in claimed if c in truth_rank]
    if len(filtered) < 2:
        # partial credit if the top-1 unknown is correctly identified
        if filtered and filtered[0] == truth[0]:
            return 0.6
        return 0.2 if filtered else 0.0
    concordant = 0
    total = 0
    for a_idx in range(len(filtered)):
        for b_idx in range(a_idx + 1, len(filtered)):
            total += 1
            ta = truth_rank[filtered[a_idx]]
            tb = truth_rank[filtered[b_idx]]
            if ta < tb:
                concordant += 1
    return concordant / total if total else 0.0


def _executive(resp: ParsedResponse, inst: Instance) -> float:
    score = 0.0
    max_score = 4.0

    # (a) probes precede commitment AND at least one probe targets an entity
    #     mentioned in a hidden rule component
    hidden_rules = {rec["rule_name"] for rec in inst.hidden_fields}
    hidden_entity_indices = set()
    for r in inst.world.rules:
        if r.name in hidden_rules:
            hidden_entity_indices.add(r.trigger.i)
            hidden_entity_indices.add(r.effect.target)
    if resp.exploratory_actions:
        score += 0.5
        if any(a.kind == "observe" and a.i in hidden_entity_indices
               for a in resp.exploratory_actions):
            score += 0.5

    # (b) final plan is non-empty and respects the horizon
    total_actions = len(resp.exploratory_actions) + len(resp.final_plan)
    if resp.final_plan and total_actions <= inst.world.horizon:
        score += 1.0
    elif resp.final_plan:
        score += 0.3

    # (c) at least one risk is named AND it corresponds to an actual hidden
    #     component (named in the risks text as substring)
    if resp.self_judgment.risks_identified:
        score += 0.3
        hidden_tokens = set()
        for rec in inst.hidden_fields:
            hidden_tokens.add(rec["rule_name"])
            for h in rec["hidden"]:
                hidden_tokens.add(h)
        if any(any(tok in risk for tok in hidden_tokens)
               for risk in resp.self_judgment.risks_identified):
            score += 0.2

    # (d) alternative plan actually differs from the final plan (flexibility)
    if resp.self_judgment.alternative_plan:
        if [a.__dict__ for a in resp.self_judgment.alternative_plan] != \
                [a.__dict__ for a in resp.final_plan]:
            score += 1.0
        else:
            score += 0.3

    return min(1.0, score / max_score)


def score_response(resp: ParsedResponse, inst: Instance,
                   best_obj: int | None = None,
                   worst_obj: int | None = None,
                   weights: Dict[str, float] | None = None) -> ScoreBreakdown:
    weights = weights or {"objective": 0.35, "calibration": 0.25,
                          "attention": 0.2, "executive": 0.2}

    if best_obj is None:
        best_obj, _ = oracle_score(inst.world)
    if worst_obj is None:
        worst_obj = _worst_objective(inst.world)

    plan_result = run_plan(inst, resp.exploratory_actions, resp.final_plan)
    raw_obj = plan_result.objective
    span = best_obj - worst_obj
    if span <= 0:
        obj_norm = 1.0  # degenerate world
    else:
        obj_norm = max(0.0, min(1.0, (raw_obj - worst_obj) / span))

    calib = _calibration(resp, inst)
    attn = _attention(resp, inst)
    exec_ = _executive(resp, inst)

    composite = (weights["objective"] * obj_norm
                 + weights["calibration"] * calib
                 + weights["attention"] * attn
                 + weights["executive"] * exec_)

    return ScoreBreakdown(
        objective=obj_norm,
        calibration=calib,
        attention=attn,
        executive=exec_,
        composite=composite,
        raw_objective=raw_obj,
        best_objective=best_obj,
        worst_objective=worst_obj,
        parse_errors=len(resp.errors),
    )
