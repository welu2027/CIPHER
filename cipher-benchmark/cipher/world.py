"""Core world model: state, rules, actions, and the step engine.

A world has E entities, each holding two properties in Z_M: `phase` and `flux`.
Rules are (trigger, effect) pairs. On each step, rules fire in order; a rule
whose trigger matches the current state applies its effect.

Actions are issued by the agent (or the benchmark harness). After each action
the rule engine advances one step.

Everything here is intentionally abstract — no real-world nouns — so nothing
about these worlds can have appeared in pretraining data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, List, Literal, Tuple

MODULUS = 7  # property space is Z_7
MAX_ENTITIES = 5

# ---------- State ----------

@dataclass(frozen=True)
class EntityState:
    phase: int
    flux: int

    def with_phase(self, p: int) -> "EntityState":
        return EntityState(phase=p % MODULUS, flux=self.flux)

    def with_flux(self, f: int) -> "EntityState":
        return EntityState(phase=self.phase, flux=f % MODULUS)


@dataclass(frozen=True)
class State:
    entities: Tuple[EntityState, ...]

    def replace_entity(self, i: int, e: EntityState) -> "State":
        lst = list(self.entities)
        lst[i] = e
        return State(tuple(lst))

    def as_tuple(self):
        return tuple((e.phase, e.flux) for e in self.entities)

    @property
    def n(self) -> int:
        return len(self.entities)


# ---------- Triggers ----------

TriggerKind = Literal["phase_eq", "flux_eq", "phase_gt", "flux_gt", "parity_odd", "parity_even", "phase_eq_phase"]

@dataclass(frozen=True)
class Trigger:
    kind: TriggerKind
    i: int                # primary entity index
    j: int = -1           # secondary entity for relational triggers
    k: int = 0            # threshold / value
    hidden_kind: bool = False
    hidden_i: bool = False
    hidden_k: bool = False

    def evaluate(self, state: State) -> bool:
        e = state.entities[self.i]
        if self.kind == "phase_eq":
            return e.phase == self.k
        if self.kind == "flux_eq":
            return e.flux == self.k
        if self.kind == "phase_gt":
            return e.phase > self.k
        if self.kind == "flux_gt":
            return e.flux > self.k
        if self.kind == "parity_odd":
            return (e.phase % 2) == 1
        if self.kind == "parity_even":
            return (e.phase % 2) == 0
        if self.kind == "phase_eq_phase":
            return e.phase == state.entities[self.j].phase
        raise ValueError(self.kind)

    def describe(self, reveal: bool = True) -> str:
        kind = self.kind if (reveal or not self.hidden_kind) else "?"
        i = str(self.i) if (reveal or not self.hidden_i) else "?"
        k = str(self.k) if (reveal or not self.hidden_k) else "?"
        if kind == "phase_eq": return f"phase(E{i}) == {k}"
        if kind == "flux_eq": return f"flux(E{i}) == {k}"
        if kind == "phase_gt": return f"phase(E{i}) > {k}"
        if kind == "flux_gt": return f"flux(E{i}) > {k}"
        if kind == "parity_odd": return f"phase(E{i}) is odd"
        if kind == "parity_even": return f"phase(E{i}) is even"
        if kind == "phase_eq_phase":
            j = str(self.j) if (reveal or not self.hidden_i) else "?"
            return f"phase(E{i}) == phase(E{j})"
        return f"<{kind} E{i} k={k}>"


# ---------- Effects ----------

EffectKind = Literal["flux_add", "phase_add", "align_phase", "swap_pf", "zero_flux"]

@dataclass(frozen=True)
class Effect:
    kind: EffectKind
    target: int
    delta: int = 0
    source: int = -1  # for align_phase: copy phase from source to target
    hidden_kind: bool = False
    hidden_target: bool = False
    hidden_delta: bool = False

    def apply(self, state: State) -> State:
        t = state.entities[self.target]
        if self.kind == "flux_add":
            return state.replace_entity(self.target, t.with_flux(t.flux + self.delta))
        if self.kind == "phase_add":
            return state.replace_entity(self.target, t.with_phase(t.phase + self.delta))
        if self.kind == "align_phase":
            src = state.entities[self.source]
            return state.replace_entity(self.target, t.with_phase(src.phase))
        if self.kind == "swap_pf":
            return state.replace_entity(self.target, EntityState(phase=t.flux, flux=t.phase))
        if self.kind == "zero_flux":
            return state.replace_entity(self.target, t.with_flux(0))
        raise ValueError(self.kind)

    def describe(self, reveal: bool = True) -> str:
        kind = self.kind if (reveal or not self.hidden_kind) else "?"
        target = str(self.target) if (reveal or not self.hidden_target) else "?"
        delta = str(self.delta) if (reveal or not self.hidden_delta) else "?"
        if kind == "flux_add": return f"flux(E{target}) += {delta}"
        if kind == "phase_add": return f"phase(E{target}) += {delta}"
        if kind == "align_phase": return f"phase(E{target}) <- phase(E{self.source})"
        if kind == "swap_pf": return f"swap phase/flux of E{target}"
        if kind == "zero_flux": return f"flux(E{target}) <- 0"
        return f"<{kind} E{target}>"


# ---------- Rules ----------

@dataclass(frozen=True)
class Rule:
    name: str
    trigger: Trigger
    effect: Effect

    def fire_if_applicable(self, state: State) -> State:
        if self.trigger.evaluate(state):
            return self.effect.apply(state)
        return state

    def describe(self, reveal_trigger: bool = True, reveal_effect: bool = True) -> str:
        return (f"{self.name}: IF {self.trigger.describe(reveal_trigger)} "
                f"THEN {self.effect.describe(reveal_effect)}")


# ---------- Actions ----------

ActionKind = Literal["pulse", "damp", "shift", "unshift", "align", "observe", "wait"]

@dataclass(frozen=True)
class Action:
    kind: ActionKind
    i: int = 0
    j: int = -1

    def apply(self, state: State) -> State:
        if self.kind == "observe" or self.kind == "wait":
            return state
        e = state.entities[self.i]
        if self.kind == "pulse":
            return state.replace_entity(self.i, e.with_flux(e.flux + 1))
        if self.kind == "damp":
            return state.replace_entity(self.i, e.with_flux(e.flux - 1))
        if self.kind == "shift":
            return state.replace_entity(self.i, e.with_phase(e.phase + 1))
        if self.kind == "unshift":
            return state.replace_entity(self.i, e.with_phase(e.phase - 1))
        if self.kind == "align":
            src = state.entities[self.j]
            return state.replace_entity(self.i, e.with_phase(src.phase))
        raise ValueError(self.kind)


def all_actions(n_entities: int) -> List[Action]:
    acts: List[Action] = [Action("wait")]
    for i in range(n_entities):
        acts += [Action("pulse", i), Action("damp", i),
                 Action("shift", i), Action("unshift", i),
                 Action("observe", i)]
        for j in range(n_entities):
            if i != j:
                acts.append(Action("align", i, j))
    return acts


# ---------- World ----------

@dataclass
class World:
    initial: State
    rules: Tuple[Rule, ...]
    horizon: int = 7  # max actions the agent may take

    def step(self, state: State) -> State:
        for r in self.rules:
            state = r.fire_if_applicable(state)
        return state

    def execute(self, plan: List[Action], start: State | None = None) -> State:
        s = start or self.initial
        for a in plan:
            s = a.apply(s)
            s = self.step(s)
        return s

    def objective(self, state: State) -> int:
        """Higher is better. Sum over entities of (phase * flux) mod M, minus
        flux penalty for flux values above 4 (unstable region)."""
        total = 0
        for e in state.entities:
            total += (e.phase * e.flux) % MODULUS
            if e.flux >= 5:
                total -= 3
        return total
