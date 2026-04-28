"""Procedural instance generator.

An Instance packages a World, a public (masked) view of the rules, and
metadata used for scoring (ground-truth answers for metacognitive claims,
impact ranking for the hidden components, etc.).

Determinism: every instance is a pure function of (seed, difficulty).

Hiding mechanism: hidden rules are OMITTED from the prompt entirely (not
shown with '?' placeholders). The model is told only that N additional laws
exist whose complete forms were not recovered. This forces genuine uncertainty
reasoning rather than `?`-token parsing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Literal, Set

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
    # Indices of rules that are fully visible in the prompt (0-based).
    visible_rule_indices: List[int]
    # Indices of rules that are completely omitted from the prompt.
    hidden_rule_indices: List[int]
    hidden_fields: List[Dict[str, Any]]  # per-rule record — kept for scoring compat
    # ground truth for metacog scoring: items the model should correctly
    # identify as known/unknown, with the correct label
    metacog_ground_truth: List[Dict[str, Any]]
    # impact ranking of hidden rules, most-impactful first (rule names)
    true_unknown_ranking: List[str]
    # the oracle objective score under the full rules (computed lazily later)
    oracle_objective: int | None = None
    # kept for backward compat with old callers that used public_rule_descriptions
    public_rule_descriptions: List[str] = field(default_factory=list)


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
    """Returns (n_entities, n_rules, n_hidden_rules).

    Hidden rules are omitted entirely from the prompt. The model is told only
    that N additional laws exist whose forms were not recovered.
    """
    if difficulty == "easy":
        return 3, 4, 1   # 3 visible + 1 hidden
    if difficulty == "medium":
        return 4, 5, 2   # 3 visible + 2 hidden
    return 4, 6, 3       # 3 visible + 3 hidden


def _oracle_plan(world: World, beam_width: int = 64) -> List:
    """Return the beam-search best plan for impact ranking."""
    from .world import all_actions
    actions = all_actions(world.initial.n)
    beam = [(world.initial, [], world.objective(world.initial))]
    best_obj = world.objective(world.initial)
    best_plan: List = []
    for _ in range(world.horizon):
        candidates = []
        for state, plan, _ in beam:
            for a in actions:
                s2 = a.apply(state)
                s2 = world.step(s2)
                candidates.append((s2, plan + [a], world.objective(s2)))
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]
        if beam and beam[0][2] > best_obj:
            best_obj = beam[0][2]
            best_plan = beam[0][1]
    return best_plan


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

    # Choose which RULES to hide entirely (not individual fields).
    rule_indices = list(range(n_rules))
    rng.shuffle(rule_indices)
    hidden_rule_idx_set: Set[int] = set(rule_indices[:n_hidden])
    visible_rule_indices = [i for i in range(n_rules) if i not in hidden_rule_idx_set]
    hidden_rule_indices = [i for i in range(n_rules) if i in hidden_rule_idx_set]

    # Build hidden_records in the old format for scoring compat.
    # Each hidden rule is treated as having all 4 components hidden.
    hidden_records: List[Dict[str, Any]] = []
    for idx in hidden_rule_indices:
        hidden_records.append({
            "rule_name": rules[idx].name,
            "hidden": ["trigger_kind", "trigger_k", "effect_kind", "effect_delta"],
        })

    world = World(initial=initial, rules=tuple(rules), horizon=7)

    # Ground truth for metacog calibration.
    # Visible rules: all 4 components are known (true_known=True).
    # Hidden rules: all 4 components are unknown (true_known=False).
    metacog_gt: List[Dict[str, Any]] = []
    for idx, rule in enumerate(rules):
        is_hidden = idx in hidden_rule_idx_set
        for component in ["trigger_kind", "trigger_k", "effect_kind", "effect_delta"]:
            metacog_gt.append({
                "rule_name": rule.name,
                "component": component,
                "true_known": not is_hidden,
            })

    # Impact ranking: ablate each hidden rule and measure objective delta
    # under the ORACLE PLAN on the full world (not a zero-action trajectory).
    # This ensures the ranking reflects what actually matters when a good agent
    # executes, not what matters when nothing happens.
    oracle_plan = _oracle_plan(world)
    baseline_obj = world.objective(world.execute(oracle_plan))

    impact = []
    for idx in hidden_rule_indices:
        rule_name = rules[idx].name
        # Build world without this hidden rule
        reduced_rules = tuple(r for i, r in enumerate(rules) if i != idx)
        reduced_world = World(initial=initial, rules=reduced_rules, horizon=world.horizon)
        # Use the same oracle plan; its performance on the reduced world measures
        # how much the hidden rule matters for that plan.
        reduced_obj = reduced_world.objective(reduced_world.execute(oracle_plan))
        impact.append((rule_name, abs(baseline_obj - reduced_obj)))
    impact.sort(key=lambda x: -x[1])

    # true_unknown_ranking: list of rule names, most impactful first.
    true_ranking = [name for name, _ in impact]

    return Instance(
        id=f"IF-{difficulty}-{seed:08x}",
        seed=seed,
        difficulty=difficulty,
        world=world,
        visible_rule_indices=visible_rule_indices,
        hidden_rule_indices=hidden_rule_indices,
        hidden_fields=hidden_records,
        metacog_ground_truth=metacog_gt,
        true_unknown_ranking=true_ranking,
        oracle_objective=None,
    )
