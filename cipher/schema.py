"""Validate and normalize a model's JSON response into typed Python objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

from .world import Action


@dataclass
class MetacogClaim:
    rule_name: str
    component: str
    known: bool
    confidence: float


@dataclass
class SelfJudgment:
    robustness_score: float
    risks_identified: List[str]
    alternative_unknown: str
    alternative_plan: List[Action]


@dataclass
class ParsedResponse:
    metacog_assessment: List[MetacogClaim]
    critical_unknowns_ranked: List[str]
    exploratory_actions: List[Action]
    final_plan: List[Action]
    self_judgment: SelfJudgment
    errors: List[str] = field(default_factory=list)


_ALLOWED_ACTION_KINDS = {"pulse", "damp", "shift", "unshift", "align", "observe", "wait"}


def _parse_action(obj: Any, errors: List[str]) -> Action | None:
    if not isinstance(obj, dict):
        errors.append(f"action not an object: {obj!r}")
        return None
    kind = obj.get("kind")
    if kind not in _ALLOWED_ACTION_KINDS:
        errors.append(f"unknown action kind: {kind!r}")
        return None
    i = int(obj.get("i", 0))
    j = int(obj.get("j", -1))
    return Action(kind=kind, i=i, j=j)


def _parse_actions(arr: Any, errors: List[str]) -> List[Action]:
    if not isinstance(arr, list):
        errors.append("actions field not a list")
        return []
    out: List[Action] = []
    for item in arr:
        a = _parse_action(item, errors)
        if a is not None:
            out.append(a)
    return out


def validate_response(raw: Dict[str, Any]) -> ParsedResponse:
    errors: List[str] = []

    metacog: List[MetacogClaim] = []
    for entry in raw.get("metacog_assessment", []) or []:
        if not isinstance(entry, dict):
            errors.append(f"metacog entry not an object: {entry!r}")
            continue
        try:
            metacog.append(MetacogClaim(
                rule_name=str(entry["rule_name"]),
                component=str(entry["component"]),
                known=bool(entry["known"]),
                confidence=max(0.0, min(1.0, float(entry.get("confidence", 0.5)))),
            ))
        except (KeyError, ValueError, TypeError) as ex:
            errors.append(f"bad metacog entry {entry!r}: {ex}")

    critical = raw.get("critical_unknowns_ranked", []) or []
    if not isinstance(critical, list):
        errors.append("critical_unknowns_ranked not a list")
        critical = []
    critical = [str(x) for x in critical]

    exploratory = _parse_actions(raw.get("exploratory_actions", []), errors)
    final_plan = _parse_actions(raw.get("final_plan", []), errors)

    sj_raw = raw.get("self_judgment", {}) or {}
    if not isinstance(sj_raw, dict):
        errors.append("self_judgment not an object")
        sj_raw = {}
    try:
        robustness = float(sj_raw.get("robustness_score", 50))
    except (ValueError, TypeError):
        robustness = 50.0
        errors.append("robustness_score not a number")
    risks = sj_raw.get("risks_identified", []) or []
    if not isinstance(risks, list):
        errors.append("risks_identified not a list")
        risks = []
    risks = [str(r) for r in risks]

    alt = sj_raw.get("alternative_if_unknown_X", {}) or {}
    if not isinstance(alt, dict):
        alt = {}
    alt_unknown = str(alt.get("unknown", "")) if alt else ""
    alt_plan = _parse_actions(alt.get("plan", []), errors) if alt else []

    return ParsedResponse(
        metacog_assessment=metacog,
        critical_unknowns_ranked=critical,
        exploratory_actions=exploratory,
        final_plan=final_plan,
        self_judgment=SelfJudgment(
            robustness_score=robustness,
            risks_identified=risks,
            alternative_unknown=alt_unknown,
            alternative_plan=alt_plan,
        ),
        errors=errors,
    )
