"""Build the natural-language prompt shown to the model.

Each instance gets a procedurally-flavored narrative: invented world name,
invented property words, invented entity noun. The action verbs track the
flavor too (e.g. "pulse" becomes "amplify" or "charge" depending on the
flux word). Underneath, the action kinds are unchanged — the model still
returns {"kind": "pulse", "i": 0} so the simulator can execute faithfully.
"""

from __future__ import annotations

from .generator import Instance
from .flavor import pick_flavor, describe_rule, Flavor


SCHEMA_BLOCK = """\
Return STRICT JSON matching this schema (and NOTHING else — no prose, no
markdown fences):

{
  "metacog_assessment": [
    {"rule_name": "<rule name as shown>", "component": "trigger_kind"|"trigger_k"|"effect_kind"|"effect_delta",
     "known": true|false, "confidence": 0.0-1.0},
    ... one entry per (rule, component) pair
  ],
  "critical_unknowns_ranked": ["<rule_name>:<component>", ...],
  "exploratory_actions": [ <up to 5 action objects> ],
  "final_plan":          [ <action objects; total actions <= horizon> ],
  "self_judgment": {
    "robustness_score": 0-100,
    "risks_identified": ["short phrase", ...],
    "alternative_if_unknown_X": {"unknown": "<rule_name>:<component>",
                                 "plan": [<action objects>]}
  }
}

Action objects (kind values are fixed tokens; i/j are entity indices):
  {"kind": "pulse",   "i": <idx>}          flux += 1
  {"kind": "damp",    "i": <idx>}          flux -= 1
  {"kind": "shift",   "i": <idx>}          phase += 1
  {"kind": "unshift", "i": <idx>}          phase -= 1
  {"kind": "align",   "i": <idx>, "j": <idx>}  phase of i copies phase of j
  {"kind": "observe", "i": <idx>}          no state change; reveals post-step values of entity i
  {"kind": "wait"}                         skip a turn
After every action, all rules fire in the listed order.
"""


def build_prompt(inst: Instance) -> str:
    fl = pick_flavor(inst.seed)
    n = inst.world.initial.n

    header = (
        f"You are studying the {fl.world_name} {fl.object_word}, a closed "
        f"system of {n} {fl.entity_pl} that has never been catalogued. Each "
        f"{fl.entity_sg} has two measurable attributes — **{fl.phase_word}** "
        f"and **{fl.flux_word}** — each an integer in the set {{0,1,2,3,4,5,6}} "
        f"(all arithmetic is mod 7).\n\n"
        f"Field agents have partially characterized its dynamics, but some "
        f"components of the governing {fl.rule_word.lower()}s could not be "
        f"resolved and are shown as '?'. Your task is to (1) judge what you "
        f"truly know versus what is withheld, (2) rank which unknowns matter "
        f"most, (3) optionally issue exploratory probes, and (4) commit to a "
        f"plan that maximizes the system's objective score.\n"
    )

    state_lines = []
    for idx, e in enumerate(inst.world.initial.entities):
        state_lines.append(
            f"  {fl.entity(idx)} ({fl.entity_sg} {idx}): "
            f"{fl.phase_word}={e.phase}, {fl.flux_word}={e.flux}"
        )

    # Relabel rules so the name the model sees is consistent with the
    # metacog_ground_truth (which uses the underlying R0, R1, ...). We tell
    # the model both the flavored label AND the canonical name.
    rule_lines = []
    for i, r in enumerate(inst.world.rules):
        flavored = describe_rule(r, fl, i)
        rule_lines.append(f"  [{r.name}] {flavored}")

    hidden_lines = []
    for rec in inst.hidden_fields:
        hidden_lines.append(f"  {rec['rule_name']}: {', '.join(rec['hidden'])}")

    goal = (
        f"\nObjective (to be maximized after your final plan executes): "
        f"sum over {fl.entity_pl} of ({fl.phase_word} * {fl.flux_word} mod 7), "
        f"minus 3 for each {fl.entity_sg} whose {fl.flux_word} ≥ 5 "
        f"(an unstable regime).\n"
        f"Action budget: at most {inst.world.horizon} actions total "
        f"(exploratory + final plan combined). Probes consume budget.\n"
    )

    return (
        header
        + "\n---\nInitial readings:\n" + "\n".join(state_lines)
        + f"\n\nGoverning {fl.rule_word.lower()}s "
          f"(fire in the order listed after every action; '?' = unresolved):\n"
        + "\n".join(rule_lines)
        + "\n\nUnresolved components (things the field agents could not confirm):\n"
        + ("\n".join(hidden_lines) if hidden_lines
           else f"  (all {fl.rule_word.lower()}s fully specified)")
        + goal
        + "\n" + SCHEMA_BLOCK
    )
