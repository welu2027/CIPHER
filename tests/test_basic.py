"""Smoke tests: run the full pipeline end-to-end on a few instances."""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cipher import generate_instance, build_prompt, validate_response
from cipher.simulator import run_plan
from cipher.scorer import score_response
from cipher.optimal import oracle_score
from cipher.world import Action


def test_generation_is_deterministic():
    a = generate_instance(seed=123, difficulty="medium")
    b = generate_instance(seed=123, difficulty="medium")
    assert a.world.initial.as_tuple() == b.world.initial.as_tuple()
    assert len(a.world.rules) == len(b.world.rules)


def test_prompt_contains_entities_and_rules():
    inst = generate_instance(seed=7, difficulty="easy")
    p = build_prompt(inst)
    # flavored prompt always mentions entity 0 via its flavor prefix letter
    assert "0:" in p or "0 (" in p
    assert "[R0]" in p  # canonical rule name still exposed for scoring
    # at least one redaction marker appears for any instance (easy n_hidden=1)
    assert "?" in p


def test_end_to_end_scoring():
    inst = generate_instance(seed=99, difficulty="easy")
    best, best_plan = oracle_score(inst.world)
    # feed the oracle plan back as the model response and expect near-max
    # objective sub-score.
    raw = {
        "metacog_assessment": [
            {"rule_name": gt["rule_name"], "component": gt["component"],
             "known": gt["true_known"],
             "confidence": 0.95 if gt["true_known"] else 0.1}
            for gt in inst.metacog_ground_truth
        ],
        "critical_unknowns_ranked": list(inst.true_unknown_ranking),
        "exploratory_actions": [],
        "final_plan": [{"kind": a.kind, "i": a.i, "j": a.j} for a in best_plan],
        "self_judgment": {
            "robustness_score": 80,
            "risks_identified": [inst.hidden_fields[0]["rule_name"]] if inst.hidden_fields else ["none"],
            "alternative_if_unknown_X": {
                "unknown": inst.true_unknown_ranking[0] if inst.true_unknown_ranking else "",
                "plan": [{"kind": "wait"}],
            },
        },
    }
    resp = validate_response(raw)
    assert not resp.errors, resp.errors
    sb = score_response(resp, inst, best_obj=best)
    assert sb.objective >= 0.9
    assert sb.calibration >= 0.9


def test_horizon_truncation():
    inst = generate_instance(seed=5, difficulty="easy")
    probes = [Action("observe", 0)] * 10
    plan = [Action("pulse", 0)] * 10
    result = run_plan(inst, probes, plan)
    # trace length equals horizon regardless of input length
    assert len(result.trace) == inst.world.horizon


if __name__ == "__main__":
    test_generation_is_deterministic()
    test_prompt_contains_entities_and_rules()
    test_end_to_end_scoring()
    test_horizon_truncation()
    print("all tests passed")
