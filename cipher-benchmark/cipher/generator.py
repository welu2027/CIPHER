"""Procedural instance generator.

An Instance packages a World, a public (masked) view of the rules, and
metadata used for scoring (ground-truth answers for metacognitive claims,
impact ranking for the hidden components, etc.).

Determinism: every instance is a pure function of (seed, difficulty).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Literal

from .world import (
    World, State, EntityState, Rule, Trigger, Effect, MODULUS,
)

Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class Instance:
    id: str
    seed: int
    difficulty: Difficulty
    world: World
    # rules_public: same rules as world.rules but with hidden fields masked in
    # their string descriptions. Structurally identical (so the simulator can
    # run), but the prompt builder will use .describe() with reveal=False on
    # masked fields.
    public_rule_descriptions: List[str]
    hidden_fields: List[Dict[str, Any]]  # per-rule record of what was hidden
    # ground truth for metacog scoring: items the model should correctly
    # identify as known/unknown, with the correct label
    metacog_ground_truth: List[Dict[str, Any]]
    # impact ranking of hidden components, most-impactful first
    true_unknown_ranking: List[str]
    # the oracle objective score under the full rules (computed lazily later)
    oracle_objective: int | None = None


TRIGGER_KINDS = ["phase_eq", "flux_eq", "phase_gt", "flux_gt",
                 "parity_odd", "parity_even", "phase_eq_phase"]
EFFECT_KINDS = ["flux_add", "phase_add", "align_phase", "swap_pf", "zero_flux"]


def _random_trigger(rng: random.Random, n: int) -> Trigger:
    kind = rng.choice(TRIGGER_KINDS)
    i = rng.randrange(n)
    j = rng.randrange(n)
    while j == i and n > 1:
        j = rng.randrange(n)
    k = rng.randrange(MODULUS)
    return Trigger(kind=kind, i=i, j=j, k=k)


def _random_effect(rng: random.Random, n: int) -> Effect:
    kind = rng.choice(EFFECT_KINDS)
    target = rng.randrange(n)
    source = rng.randrange(n)
    while source == target and n > 1:
        source = rng.randrange(n)
    delta = rng.choice([-2, -1, 1, 2, 3])
    return Effect(kind=kind, target=target, delta=delta, source=source)


def _diff_params(difficulty: Difficulty) -> Tuple[int, int, int]:
    """Returns (n_entities, n_rules, n_hidden_fields)."""
    if difficulty == "easy":
        return 3, 3, 1
    if difficulty == "medium":
        return 4, 4, 2
    return 4, 5, 3


def generate_instance(seed: int, difficulty: Difficulty = "medium") -> Instance:
    rng = random.Random(seed)
    n_entities, n_rules, n_hidden = _diff_params(difficulty)

    initial = State(tuple(
        EntityState(phase=rng.randrange(MODULUS), flux=rng.randrange(MODULUS))
        for _ in range(n_entities)
    ))

    rules: List[Rule] = []
    for r_idx in range(n_rules):
        rules.append(Rule(
            name=f"R{r_idx}",
            trigger=_random_trigger(rng, n_entities),
            effect=_random_effect(rng, n_entities),
        ))

    # Choose which fields to hide. Each hidden slot is a (rule_index, field).
    candidate_slots = []
    for idx in range(n_rules):
        candidate_slots.append((idx, "trigger_kind"))
        candidate_slots.append((idx, "trigger_k"))
        candidate_slots.append((idx, "effect_kind"))
        candidate_slots.append((idx, "effect_delta"))
    rng.shuffle(candidate_slots)
    hidden_slots = candidate_slots[:n_hidden]

    # Apply hiding by replacing rules with masked-metadata versions (structure
    # unchanged — simulator still uses full values; only .describe() hides).
    masked_rules: List[Rule] = []
    hidden_records: List[Dict[str, Any]] = []
    for idx, rule in enumerate(rules):
        t = rule.trigger
        e = rule.effect
        slot_info = {"rule_name": rule.name, "hidden": []}
        for (h_idx, field) in hidden_slots:
            if h_idx != idx:
                continue
            if field == "trigger_kind":
                t = Trigger(**{**t.__dict__, "hidden_kind": True})
                slot_info["hidden"].append("trigger_kind")
            elif field == "trigger_k":
                t = Trigger(**{**t.__dict__, "hidden_k": True})
                slot_info["hidden"].append("trigger_k")
            elif field == "effect_kind":
                e = Effect(**{**e.__dict__, "hidden_kind": True})
                slot_info["hidden"].append("effect_kind")
            elif field == "effect_delta":
                e = Effect(**{**e.__dict__, "hidden_delta": True})
                slot_info["hidden"].append("effect_delta")
        masked_rules.append(Rule(name=rule.name, trigger=t, effect=e))
        if slot_info["hidden"]:
            hidden_records.append(slot_info)

    world = World(initial=initial, rules=tuple(masked_rules), horizon=7)

    public_descriptions = [
        r.describe(reveal_trigger=False, reveal_effect=False) for r in masked_rules
    ]

    # Ground truth for metacog calibration. We construct a set of factual claims
    # the model could assert; the correct confidence is 1.0 for true claims and
    # 0.0 for false. The prompt will ask the model to rate its confidence that
    # specific rule components are known.
    metacog_gt: List[Dict[str, Any]] = []
    for idx, rule in enumerate(masked_rules):
        hidden = set()
        for rec in hidden_records:
            if rec["rule_name"] == rule.name:
                hidden = set(rec["hidden"])
        # 4 claims per rule: each claim = "I know <component>"
        for component in ["trigger_kind", "trigger_k", "effect_kind", "effect_delta"]:
            metacog_gt.append({
                "rule_name": rule.name,
                "component": component,
                "true_known": component not in hidden,
            })

    # Impact ranking: ablate each hidden component and measure objective delta
    # when the rule's effect is suppressed vs not. Rules whose effect matters
    # more (higher absolute delta) rank first.
    impact = []
    baseline_final = world.execute([])  # just let rules step once, no actions.
    baseline_obj = world.objective(baseline_final)
    for rec in hidden_records:
        rule_name = rec["rule_name"]
        # build world without this rule
        reduced = tuple(r for r in masked_rules if r.name != rule_name)
        reduced_world = World(initial=initial, rules=reduced, horizon=world.horizon)
        reduced_final = reduced_world.execute([])
        reduced_obj = reduced_world.objective(reduced_final)
        impact.append((rule_name, abs(baseline_obj - reduced_obj)))
    impact.sort(key=lambda x: -x[1])
    true_ranking = [f"{name}:{','.join(rec['hidden'])}"
                    for name, _ in impact
                    for rec in hidden_records if rec["rule_name"] == name]

    return Instance(
        id=f"IF-{difficulty}-{seed:08x}",
        seed=seed,
        difficulty=difficulty,
        world=world,
        public_rule_descriptions=public_descriptions,
        hidden_fields=hidden_records,
        metacog_ground_truth=metacog_gt,
        true_unknown_ranking=true_ranking,
        oracle_objective=None,
    )
