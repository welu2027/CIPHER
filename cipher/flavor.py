"""Procedural flavor layer.

Each instance gets a deterministic, seeded rebranding of the underlying
abstract world into a micro-narrative with invented vocabulary. The math is
unchanged — the flavor is purely cosmetic — but:

  - judges (and the model) see a "scientific discovery in an unknown world"
    framing instead of a Z_7 algebra drill,
  - every instance uses different invented tokens, so no model can have
    memorized "Zorath lattice" or "resonance flux" as a coherent concept,
  - the prompt can no longer be trivially copy-pasted into a Python REPL
    (the model has to first parse natural language, which is where
    metacognitive failures surface).

Vocabulary is drawn from invented phonotactic stems — nothing with a real-
world referent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


# ---- Invented vocabulary pools (all nonsense, deliberately) ----

WORLD_NAMES = [
    "Zorath", "Quelian", "Vythra", "Orlen", "Thresh", "Miraxi", "Kelvros",
    "Andor", "Pelagon", "Xenith", "Orrek", "Tysil", "Nubari", "Corvath",
    "Draxen", "Hyloth", "Ulmira", "Brenn", "Gyrath", "Loskar",
]

ENTITY_NOUNS = [
    ("glyph", "glyphs"), ("rune", "runes"), ("node", "nodes"),
    ("orb", "orbs"), ("sigil", "sigils"), ("shard", "shards"),
    ("beacon", "beacons"), ("lattice point", "lattice points"),
    ("filament", "filaments"), ("cell", "cells"),
]

PHASE_WORDS = [
    "resonance", "helix", "cadence", "tilt", "spin", "crest",
    "alignment", "polarity", "aspect", "phase",
]

FLUX_WORDS = [
    "torsion", "ember", "surge", "drift", "pulse", "gradient",
    "charge", "flux", "strain", "bloom",
]

RULE_WORDS = ["Law", "Axiom", "Principle", "Edict", "Tenet", "Canon"]

OBJECT_WORDS = [
    "mesh", "lattice", "array", "constellation", "choir", "stack",
    "weave", "spire", "garden", "cluster",
]


@dataclass(frozen=True)
class Flavor:
    world_name: str           # "Zorath"
    object_word: str          # "lattice"
    entity_sg: str            # "glyph"
    entity_pl: str            # "glyphs"
    entity_prefix: str        # "G"  (so entity 0 is "G0")
    phase_word: str           # "resonance"
    flux_word: str            # "torsion"
    rule_word: str            # "Law"

    def entity(self, i: int) -> str:
        return f"{self.entity_prefix}{i}"

    def rule_label(self, idx: int) -> str:
        greek = "αβγδεζηθικλμν"
        return f"{self.rule_word} {greek[idx % len(greek)]}"


def pick_flavor(seed: int) -> Flavor:
    rng = random.Random(seed ^ 0xF1AF0A)
    ent_sg, ent_pl = rng.choice(ENTITY_NOUNS)
    return Flavor(
        world_name=rng.choice(WORLD_NAMES),
        object_word=rng.choice(OBJECT_WORDS),
        entity_sg=ent_sg,
        entity_pl=ent_pl,
        entity_prefix=ent_sg[0].upper(),
        phase_word=rng.choice(PHASE_WORDS),
        flux_word=rng.choice(FLUX_WORDS),
        rule_word=rng.choice(RULE_WORDS),
    )


# ---- Description rendering in flavor terms ----

def describe_trigger(trigger, fl: Flavor, reveal: bool = True) -> str:
    kind = trigger.kind if (reveal or not trigger.hidden_kind) else "?"
    i = fl.entity(trigger.i) if (reveal or not trigger.hidden_i) else "?"
    k = str(trigger.k) if (reveal or not trigger.hidden_k) else "?"
    P, F = fl.phase_word, fl.flux_word
    if kind == "phase_eq": return f"{P} of {i} equals {k}"
    if kind == "flux_eq": return f"{F} of {i} equals {k}"
    if kind == "phase_gt": return f"{P} of {i} exceeds {k}"
    if kind == "flux_gt": return f"{F} of {i} exceeds {k}"
    if kind == "parity_odd": return f"{P} of {i} is odd"
    if kind == "parity_even": return f"{P} of {i} is even"
    if kind == "phase_eq_phase":
        j = fl.entity(trigger.j) if (reveal or not trigger.hidden_i) else "?"
        return f"{P} of {i} matches {P} of {j}"
    if kind == "?":
        return f"(a withheld condition on {i})"
    return f"{kind}({i},{k})"


def describe_effect(effect, fl: Flavor, reveal: bool = True) -> str:
    kind = effect.kind if (reveal or not effect.hidden_kind) else "?"
    tgt = fl.entity(effect.target) if (reveal or not effect.hidden_target) else "?"
    delta = str(effect.delta) if (reveal or not effect.hidden_delta) else "?"
    P, F = fl.phase_word, fl.flux_word
    if kind == "flux_add": return f"{F} of {tgt} shifts by {delta} (mod 7)"
    if kind == "phase_add": return f"{P} of {tgt} shifts by {delta} (mod 7)"
    if kind == "align_phase":
        src = fl.entity(effect.source)
        return f"{P} of {tgt} is drawn to match {P} of {src}"
    if kind == "swap_pf": return f"{P} and {F} of {tgt} exchange places"
    if kind == "zero_flux": return f"{F} of {tgt} collapses to 0"
    if kind == "?": return f"(an unspecified change to {tgt})"
    return f"{kind}({tgt})"


def describe_rule(rule, fl: Flavor, rule_idx: int) -> str:
    label = fl.rule_label(rule_idx)
    return (f"{label}: whenever {describe_trigger(rule.trigger, fl, reveal=False)}, "
            f"{describe_effect(rule.effect, fl, reveal=False)}.")
