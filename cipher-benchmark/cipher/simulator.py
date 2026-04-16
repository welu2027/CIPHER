"""Run a model's plan against the full (non-masked) world and report outcomes.

The simulator is invoked at *scoring time only* — it has access to the real
rules. The prompt shown to the model hides the redacted components, but the
World stored in the Instance always carries the full semantics so the simulator
can execute faithfully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .world import World, State, Action
from .generator import Instance


@dataclass
class PlanResult:
    final_state: State
    objective: int
    trace: List[State]  # states after each action (post rule-step)


def run_actions(world: World, actions: List[Action], start: State | None = None) -> PlanResult:
    s = start if start is not None else world.initial
    trace: List[State] = []
    for a in actions:
        s = a.apply(s)
        s = world.step(s)
        trace.append(s)
    if not trace:
        trace = [s]
    return PlanResult(final_state=s, objective=world.objective(s), trace=trace)


def run_plan(inst: Instance, exploratory: List[Action], final_plan: List[Action]) -> PlanResult:
    """Execute exploratory probes then the committed final plan.

    Total actions are capped at world.horizon; excess actions are truncated.
    Exploratory probes DO cost horizon budget (so wasteful probing is penalized
    naturally by having fewer actions for the final plan).
    """
    budget = inst.world.horizon
    combined = (exploratory[: max(0, budget)]
                + final_plan[: max(0, budget - len(exploratory[: budget]))])
    combined = combined[:budget]
    return run_actions(inst.world, combined)
