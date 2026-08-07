"""Evaluation harness for the analyst copilot.

Shipping an LLM feature without an eval harness is the GenAI equivalent of
shipping a model without a test set, and it is the single thing that
separates a working RAG system from a demo. This module answers four
questions, each with a different method, because no single metric covers
them:

1. **Groundedness** — does every claim trace to the provided evidence?
   Checked deterministically where possible (cited features must appear in
   the actual attributions) and by judge where not.
2. **Retrieval quality** — did we fetch the right typology document?
   Checked against ground truth, since the generator labels typology.
3. **Typology accuracy** — did the copilot name the right attack?
   Exact match against ground truth. No judge needed.
4. **Narrative quality** — is the prose actually useful to an analyst?
   LLM-as-judge, and *only* here, because it is the one dimension with no
   programmatic ground truth.

The judge is calibrated before it is trusted: `calibrate_judge` scores a set
of human-labelled examples and reports agreement. A judge that agrees with
human labels 60% of the time is not a measurement instrument, and reporting
its scores as if it were would be worse than reporting nothing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """One golden-set case: inputs plus known-correct answers."""

    case_id: str
    risk_score: float
    amount: float
    decision: str
    attributions: list[dict[str, Any]]
    # Ground truth from the generator — available because we made the data.
    true_typology: str
    # Optional human-written reference narrative for judge calibration.
    reference_summary: str | None = None
    human_quality_label: int | None = None  # 1-5, for calibration only


@dataclass
class CaseResult:
    case_id: str
    groundedness: float
    retrieval_hit: bool
    typology_correct: bool
    quality_score: float | None
    violations: list[str] = field(default_factory=list)


@dataclass
class CopilotEvalReport:
    n_cases: int
    groundedness_mean: float
    retrieval_precision: float
    typology_accuracy: float
    quality_mean: float | None
    refusal_rate: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    results: list[CaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "── Copilot evaluation " + "─" * 32,
            f"  cases                 {self.n_cases}",
            f"  groundedness          {self.groundedness_mean:.3f}   <- gate on this",
            f"  retrieval precision   {self.retrieval_precision:.3f}",
            f"  typology accuracy     {self.typology_accuracy:.3f}",
        ]
        if self.quality_mean is not None:
            lines.append(f"  narrative quality     {self.quality_mean:.2f} / 5")
        lines += [
            f"  refusal rate          {self.refusal_rate:.3f}",
            f"  tokens                {self.total_input_tokens:,} in / "
            f"{self.total_output_tokens:,} out",
            f"  est. cost             ${self.estimated_cost_usd:.4f}",
        ]
        ungrounded = [r for r in self.results if r.violations]
        if ungrounded:
            lines.append("")
            lines.append(f"  {len(ungrounded)} case(s) with groundedness violations:")
            for r in ungrounded[:5]:
                lines.append(f"    {r.case_id}: {'; '.join(r.violations[:2])}")
        return "\n".join(lines)


def check_groundedness(
    draft: Any,
    attributions: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    """Deterministic groundedness: are cited features real?

    This is the half of groundedness that needs no judge at all. The copilot
    lists the features it relied on; every one of them must appear in the
    attributions it was given. A cited feature that was never in the input is
    a fabrication, full stop — and catching it costs nothing.
    """
    available = {a["feature"] for a in attributions}
    violations: list[str] = []

    cited = getattr(draft, "evidence_cited", []) or []
    if not cited:
        return 0.0, ["no evidence cited"]

    grounded = 0
    for feature in cited:
        if feature in available:
            grounded += 1
        else:
            violations.append(f"cited '{feature}' which is not in the attributions")

    # A claimed typology must correspond to a document that was retrieved.
    retrieved = set(getattr(draft, "retrieved_docs", []) or [])
    typ = getattr(draft, "likely_typology", None)
    if typ and retrieved and typ not in retrieved:
        violations.append(
            f"claimed typology '{typ}' without the corresponding reference doc"
        )

    return grounded / len(cited), violations


_JUDGE_SYSTEM = """\
You are evaluating a fraud case narrative written for a human analyst working \
a review queue under time pressure.

Score the narrative 1-5 on usefulness:

5 - An analyst could act on this immediately. States what happened, cites the \
    specific evidence, and names a concrete next check.
4 - Useful and accurate, but either buries the key point or omits an obvious \
    next step.
3 - Accurate but generic. Restates the score without adding interpretation.
2 - Vague, or hedges so heavily it gives the analyst nothing to act on.
1 - Misleading, internally contradictory, or asserts things the evidence does \
    not support.

Judge usefulness to the analyst, not prose style. A blunt, well-evidenced \
two-sentence narrative outranks a polished but vague paragraph.

Confident language on weak evidence should score low. Explicitly flagging \
ambiguity when the evidence is genuinely ambiguous is correct behaviour and \
should NOT be penalised.\
"""

_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


async def judge_quality(
    client: Any,
    model: str,
    summary: str,
    attributions: list[dict[str, Any]],
    recommended_action: str,
) -> tuple[int, str]:
    """LLM-as-judge score for narrative usefulness."""
    attr_text = "\n".join(
        f"- {a['feature']} = {a['value']} (SHAP {a['contribution']:+.4f})"
        for a in attributions
    )
    response = await client.messages.create(
        model=model,
        max_tokens=1000,
        system=[{"type": "text", "text": _JUDGE_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "adaptive"},
        output_config={"effort": "low", "format": {"type": "json_schema",
                                                   "schema": _JUDGE_SCHEMA}},
        messages=[{
            "role": "user",
            "content": (
                f"# Evidence the writer was given\n\n{attr_text}\n\n"
                f"# Narrative\n\n{summary}\n\n"
                f"# Recommended action\n\n{recommended_action}\n\n"
                "Score it."
            ),
        }],
    )
    if response.stop_reason == "refusal":
        return 3, "judge refused; neutral score assigned"
    text = next((b.text for b in response.content if b.type == "text"), "{}")
    payload = json.loads(text)
    return payload["score"], payload["reasoning"]


async def calibrate_judge(
    client: Any,
    model: str,
    labelled_cases: list[EvalCase],
    drafts: list[Any],
) -> dict[str, float]:
    """Measure judge-human agreement before trusting the judge.

    Reports exact agreement, within-one agreement, and mean signed bias. A
    judge that is consistently one point generous is still usable once you
    know the offset; a judge with low within-one agreement is not usable at
    all, and its scores should not be reported as a quality metric.
    """
    pairs: list[tuple[int, int]] = []
    for case, draft in zip(labelled_cases, drafts, strict=False):
        if case.human_quality_label is None or draft is None:
            continue
        score, _ = await judge_quality(
            client, model, draft.summary, case.attributions, draft.recommended_action
        )
        pairs.append((case.human_quality_label, score))

    if not pairs:
        return {"n": 0, "exact_agreement": 0.0, "within_one": 0.0, "bias": 0.0}

    exact = sum(1 for h, j in pairs if h == j) / len(pairs)
    within_one = sum(1 for h, j in pairs if abs(h - j) <= 1) / len(pairs)
    bias = sum(j - h for h, j in pairs) / len(pairs)

    return {
        "n": float(len(pairs)),
        "exact_agreement": round(exact, 3),
        "within_one": round(within_one, 3),
        "bias": round(bias, 3),
        # Below this, judge scores are noise and must not gate CI.
        "usable": float(within_one >= 0.80),
    }


# Claude Opus 5 pricing, USD per million tokens.
_PRICE_IN = 5.00
_PRICE_OUT = 25.00


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1e6) * _PRICE_IN + (output_tokens / 1e6) * _PRICE_OUT


def build_report(
    results: list[CaseResult],
    input_tokens: int,
    output_tokens: int,
    n_refused: int,
) -> CopilotEvalReport:
    n = len(results)
    quality = [r.quality_score for r in results if r.quality_score is not None]
    return CopilotEvalReport(
        n_cases=n,
        groundedness_mean=sum(r.groundedness for r in results) / n if n else 0.0,
        retrieval_precision=sum(r.retrieval_hit for r in results) / n if n else 0.0,
        typology_accuracy=sum(r.typology_correct for r in results) / n if n else 0.0,
        quality_mean=sum(quality) / len(quality) if quality else None,
        refusal_rate=n_refused / n if n else 0.0,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        estimated_cost_usd=estimate_cost(input_tokens, output_tokens),
        results=results,
    )


# CI gate. Groundedness is the hard floor: a copilot that fabricates evidence
# in a system that blocks people's cards is not shippable at any quality score.
GATES: dict[str, float] = {
    "groundedness_mean": 0.95,
    "typology_accuracy": 0.60,
    "refusal_rate_max": 0.05,
}


def check_gates(report: CopilotEvalReport) -> tuple[bool, list[str]]:
    """Return (passed, failures). Called by CI to block a regression."""
    failures = []
    if report.groundedness_mean < GATES["groundedness_mean"]:
        failures.append(
            f"groundedness {report.groundedness_mean:.3f} < "
            f"{GATES['groundedness_mean']} — copilot is citing evidence it was not given"
        )
    if report.typology_accuracy < GATES["typology_accuracy"]:
        failures.append(
            f"typology accuracy {report.typology_accuracy:.3f} < "
            f"{GATES['typology_accuracy']}"
        )
    if report.refusal_rate > GATES["refusal_rate_max"]:
        failures.append(
            f"refusal rate {report.refusal_rate:.3f} > {GATES['refusal_rate_max']}"
        )
    return len(failures) == 0, failures


def load_golden_set(path: str | Path) -> list[EvalCase]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**row) for row in data]


__all__ = [
    "EvalCase",
    "CaseResult",
    "CopilotEvalReport",
    "check_groundedness",
    "judge_quality",
    "calibrate_judge",
    "build_report",
    "check_gates",
    "load_golden_set",
    "GATES",
]
