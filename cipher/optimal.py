"""Oracle: beam search over the action space to estimate the best achievable
objective under the full rule set. Used to normalize the agent's score.

For horizon=7 and ~5 entities the full action branching factor is ~25, so
exhaustive search is 25^7 ~ 6e9 — too big. We use beam search with a modest
beam width (default 64) which empirically gets within a couple of points of
optimal on worlds of this size and is vastly cheaper.
"""

from __future__ import annotations

from typing import List, Tuple

from .world import World, State, Action, all_actions
from .simulator import run_actions


def oracle_score(world: World, beam_width: int = 64) -> Tuple[int, List[Action]]:
    actions = all_actions(world.initial.n)
    # beam of (state, plan-so-far, objective)
    beam: List[Tuple[State, List[Action], int]] = [(world.initial, [], world.objective(world.initial))]
    best_obj = world.objective(world.initial)
    best_plan: List[Action] = []

    for _ in range(world.horizon):
        candidates: List[Tuple[State, List[Action], int]] = []
        for state, plan, _obj in beam:
            for a in actions:
                s2 = a.apply(state)
                s2 = world.step(s2)
                candidates.append((s2, plan + [a], world.objective(s2)))
        # keep top beam_width by objective
        candidates.sort(key=lambda x: -x[2])
        beam = candidates[:beam_width]
        if beam and beam[0][2] > best_obj:
            best_obj = beam[0][2]
            best_plan = beam[0][1]

    return best_obj, best_plan
