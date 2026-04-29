"""Build the natural-language prompt shown to the model.

Each instance gets a procedurally-flavored narrative: invented world name,
invented property words, invented entity noun. The action verbs track the
flavor too (e.g. "pulse" becomes "amplify" or "charge" depending on the
flux word). Underneath, the action kinds are unchanged - the model still
returns {"kind": "pulse", "i": 0} so the simulator can execute faithfully.
"""

from __future__ import annotations

from .generator import Instance
from .flavor import pick_flavor, describe_rule, Flavor


SCHEMA_BLOCK = """\
Return STRICT JSON matching this schema (and NOTHING else - no prose, no
markdown fences):

{
  "metacog_assessment": [
    {"rule_name": "<rule name as shown>", "component": "trigger_kind"|"trigger_k"|"effect_kind"|"effect_delta",
     "known": true|false, "confidence": 0.0-1.0},
    ... one entry per (rule, component) pair - include ALL visible rules AND
    your best-effort assessment of each hidden law (use the hidden law labels
    H0, H1, ... in the order they are declared above)
  ],
  "critical_unknowns_ranked": ["<hidden_law_label>", ...],
  "exploratory_actions": [ <up to 5 action objects> ],
  "final_plan":          [ <action objects; total actions <= horizon> ],
  "self_judgment": {
    "robustness_score": 0-100,
    "risks_identified": ["short phrase", ...],
    "alternative_if_unknown_X": {"unknown": "<hidden_law_label>",
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
After every action, all rules fire in the listed order (visible first, then
hidden laws in the order declared - you do not know when hidden laws fire or
what they do).
"""


def build_prompt(inst: Instance) -> str:
    fl = pick_flavor(inst.seed)
    n = inst.world.initial.n
    n_hidden = len(inst.hidden_rule_indices)
    n_visible = len(inst.visible_rule_indices)

    header = (
        f"You are studying the {fl.world_name} {fl.object_word}, a closed "
        f"system of {n} {fl.entity_pl} that has never been catalogued. Each "
        f"{fl.entity_sg} has two measurable attributes - **{fl.phase_word}** "
        f"and **{fl.flux_word}** - each an integer in the set {{0,1,2,3,4,5,6}} "
        f"(all arithmetic is mod 7).\n\n"
        f"Field agents have characterized {n_visible} of the governing "
        f"{fl.rule_word.lower()}s, but {n_hidden} additional "
        f"{'law' if n_hidden == 1 else 'laws'} could not be recovered in full - "
        f"their triggers, effects, and even which entities they involve are "
        f"unknown. Your task is to (1) judge what you truly know versus what is "
        f"withheld, (2) rank which hidden {fl.rule_word.lower()}s matter most, "
        f"(3) optionally issue exploratory probes, and (4) commit to a plan "
        f"that maximizes the system's objective score.\n"
    )

    state_lines = []
    for idx, e in enumerate(inst.world.initial.entities):
        state_lines.append(
            f"  {fl.entity(idx)} ({fl.entity_sg} {idx}): "
            f"{fl.phase_word}={e.phase}, {fl.flux_word}={e.flux}"
        )

    # Show only the visible rules; hidden rules are completely omitted.
    rule_lines = []
    for pos, rule_idx in enumerate(inst.visible_rule_indices):
        r = inst.world.rules[rule_idx]
        flavored = describe_rule(r, fl, pos)
        rule_lines.append(f"  [{r.name}] {flavored}")

    # Declare the hidden laws by placeholder label only.
    hidden_decl_lines = []
    for h_pos in range(n_hidden):
        label = f"H{h_pos}"
        hidden_decl_lines.append(
            f"  [{label}] (complete form not recovered - trigger, effect, "
            f"and affected {fl.entity_pl} are all unknown)"
        )

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
        + f"\n\nCharacterized {fl.rule_word.lower()}s "
          f"(fire in the listed order after every action):\n"
        + "\n".join(rule_lines)
        + f"\n\nUnrecovered {fl.rule_word.lower()}s "
          f"(existence confirmed; full form unknown):\n"
        + ("\n".join(hidden_decl_lines) if hidden_decl_lines
           else f"  (none - all {fl.rule_word.lower()}s fully characterized)")
        + goal
        + "\n" + SCHEMA_BLOCK
    )
