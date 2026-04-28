"""Multi-dimensional scoring.

Four sub-scores, all normalized to [0, 1]:

1. objective    — (agent_obj - worst) / (best - worst), clamped to [0, 1].
2. calibration  — 1 - mean Brier score over metacognitive claims.
3. attention    — pairwise concordance between the model's ranked list of
                  hidden laws (H0, H1, ...) and the true impact ranking.
4. executive    — simulation-based: does the alternative plan actually
                  outperform the final plan under an adversarial counterfactual?

Composite = weighted mean (default weights).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

from .generator import Instance
from .schema import ParsedResponse
from .simulator import run_plan, run_actions
from .optimal import oracle_score
from .world import Action, World


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
    """Pairwise concordance between the model's ranked hidden-law list and
    the true impact ranking. The model lists hidden laws by their prompt
    labels (H0, H1, ...) which map to hidden_rule_indices in order."""
    truth = inst.true_unknown_ranking  # list of rule names, most-impactful first
    if not truth:
        return 1.0

    # Build a mapping from prompt label (H0, H1, ...) to rule name.
    label_to_rule = {
        f"H{pos}": inst.world.rules[rule_idx].name
        for pos, rule_idx in enumerate(inst.hidden_rule_indices)
    }
    truth_rank = {name: idx for idx, name in enumerate(truth)}

    # Resolve the model's claimed ranking (accepts either Hx labels or rule names).
    claimed_rules: List[str] = []
    for item in resp.critical_unknowns_ranked:
        if item in label_to_rule:
            claimed_rules.append(label_to_rule[item])
        elif item in truth_rank:
            claimed_rules.append(item)
        # unrecognized entries are silently dropped

    filtered = [c for c in claimed_rules if c in truth_rank]
    if len(filtered) < 2:
        if filtered and filtered[0] == truth[0]:
            return 0.6
        return 0.2 if filtered else 0.0

    concordant = 0
    total = 0
    for a_idx in range(len(filtered)):
        for b_idx in range(a_idx + 1, len(filtered)):
            total += 1
            if truth_rank[filtered[a_idx]] < truth_rank[filtered[b_idx]]:
                concordant += 1
    return concordant / total if total else 0.0


def _executive(resp: ParsedResponse, inst: Instance,
               best_obj: int, worst_obj: int) -> float:
    """Simulation-based executive score.

    Two components, equally weighted:

    (A) Probe quality (0-1): did the model probe entities involved in hidden
        rules, and was the probe budget reasonable relative to the horizon?

    (B) Contingency quality (0-1): is the alternative plan actually better than
        the final plan under an adversarial counterfactual where ALL hidden rules
        fire destructively (i.e., zero_flux on all their target entities)?
        We simulate both plans on that adversarial world and compare.
    """
    span = max(1, best_obj - worst_obj)

    # --- Component A: probe quality ---
    hidden_entity_indices = set()
    for rule_idx in inst.hidden_rule_indices:
        r = inst.world.rules[rule_idx]
        hidden_entity_indices.add(r.trigger.i)
        hidden_entity_indices.add(r.effect.target)

    probe_score = 0.0
    if resp.exploratory_actions:
        # partial credit for any probes
        probe_score += 0.4
        # bonus if at least one probe observes an entity in a hidden rule
        if any(a.kind == "observe" and a.i in hidden_entity_indices
               for a in resp.exploratory_actions):
            probe_score += 0.4
        # bonus if probe budget is <= half the horizon (disciplined probing)
        budget_used = len(resp.exploratory_actions)
        if budget_used <= inst.world.horizon // 2:
            probe_score += 0.2
    probe_score = min(1.0, probe_score)

    # --- Component B: contingency quality ---
    # Build an adversarial world where each hidden rule becomes zero_flux on
    # its effect target — a worst-case interpretation.
    from .world import Rule, Trigger, Effect, State
    adv_rules = list(inst.world.rules)
    for rule_idx in inst.hidden_rule_indices:
        orig = inst.world.rules[rule_idx]
        # Adversarial rule: same trigger, but zero_flux on its effect target.
        adv_effect = Effect(kind="zero_flux", target=orig.effect.target,
                            delta=0, source=orig.effect.source)
        adv_rules[rule_idx] = Rule(name=orig.name, trigger=orig.trigger,
                                   effect=adv_effect)
    adv_world = World(initial=inst.world.initial,
                      rules=tuple(adv_rules), horizon=inst.world.horizon)

    budget = inst.world.horizon
    final_combined = (resp.exploratory_actions[:budget]
                      + resp.final_plan[:max(0, budget - len(resp.exploratory_actions[:budget]))])
    final_combined = final_combined[:budget]

    if resp.self_judgment.alternative_plan:
        alt_combined = (resp.exploratory_actions[:budget]
                        + resp.self_judgment.alternative_plan[:max(0, budget - len(resp.exploratory_actions[:budget]))])
        alt_combined = alt_combined[:budget]
    else:
        alt_combined = final_combined

    final_adv_obj = adv_world.objective(adv_world.execute(final_combined))
    alt_adv_obj = adv_world.objective(adv_world.execute(alt_combined))

    # Contingency score: how much better is the alternative under adversarial conditions?
    if not resp.self_judgment.alternative_plan:
        contingency_score = 0.0
    elif alt_adv_obj > final_adv_obj:
        # Alternative genuinely helps under adversarial conditions
        improvement = (alt_adv_obj - final_adv_obj) / span
        contingency_score = min(1.0, 0.5 + improvement)
    elif alt_adv_obj == final_adv_obj and [a.__dict__ for a in resp.self_judgment.alternative_plan] != \
            [a.__dict__ for a in resp.final_plan]:
        # Different plan, same outcome — partial credit for attempting
        contingency_score = 0.3
    else:
        contingency_score = 0.1

    return 0.5 * probe_score + 0.5 * contingency_score


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
    exec_ = _executive(resp, inst, best_obj=best_obj, worst_obj=worst_obj)

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
