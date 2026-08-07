"""Build a golden set and run the copilot evaluation end to end.

    python -m argus_ml.evaluation.run_copilot_eval --n 40

The golden set is built from the *test window* — transactions no model was
trained on — and takes its typology labels from the generator, so retrieval
and typology accuracy have real ground truth rather than an LLM's opinion of
what the right answer was.

Exit code is non-zero when a CI gate fails, so this drops straight into a
workflow step.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from argus_ml.evaluation.copilot_eval import (
    CaseResult,
    EvalCase,
    build_report,
    check_gates,
    check_groundedness,
    judge_quality,
)

ARTIFACTS = Path(__file__).resolve().parent.parent.parent / "artifacts"


def build_golden_set(artifacts: Path, n: int, seed: int = 42) -> list[EvalCase]:
    """Sample fraudulent test-window transactions and attach model evidence."""
    model = joblib.load(artifacts / "xgboost_risk.joblib")
    feature_names = json.loads((artifacts / "feature_names.json").read_text())
    reference = pd.read_parquet(artifacts / "reference_sample.parquet")

    frauds = reference[reference["is_fraud"]]
    if frauds.empty:
        raise RuntimeError("reference sample contains no fraud — cannot build a golden set")

    sample = frauds.sample(min(n, len(frauds)), random_state=seed)
    X = sample[feature_names].to_numpy(dtype=np.float32)

    scores = model.predict_proba(X)
    explanations = model.explain(X, top_k=6)

    cases: list[EvalCase] = []
    for i, (_, row) in enumerate(sample.iterrows()):
        attributions = [
            {
                "feature": name,
                "value": round(float(value), 4),
                "contribution": round(float(contrib), 5),
                "direction": "increases_risk" if contrib > 0 else "decreases_risk",
            }
            for name, value, contrib in explanations[i]
        ]
        cases.append(
            EvalCase(
                case_id=str(row["transaction_id"]),
                risk_score=float(scores[i]),
                amount=float(row["amount"]),
                decision="review",
                attributions=attributions,
                true_typology=str(row["typology"]),
            )
        )
    return cases


async def run(n: int, artifacts: Path, use_judge: bool) -> int:
    from anthropic import AsyncAnthropic

    cases = build_golden_set(artifacts, n)
    print(f"golden set: {len(cases)} cases")

    results: list[CaseResult] = []
    in_tokens = out_tokens = 0
    n_refused = 0

    client = AsyncAnthropic()

    for case in cases:
        draft = await _draft_one(case)
        if draft is None or getattr(draft, "refused", False):
            n_refused += 1
            results.append(
                CaseResult(
                    case_id=case.case_id, groundedness=0.0, retrieval_hit=False,
                    typology_correct=False, quality_score=None,
                    violations=["no draft produced"],
                )
            )
            continue

        in_tokens += draft.input_tokens
        out_tokens += draft.output_tokens

        groundedness, violations = check_groundedness(draft, case.attributions)
        retrieval_hit = case.true_typology in (draft.retrieved_docs or [])
        typology_correct = draft.likely_typology == case.true_typology

        quality = None
        if use_judge:
            score, _ = await judge_quality(
                client, "claude-opus-5", draft.summary,
                case.attributions, draft.recommended_action,
            )
            quality = float(score)

        results.append(
            CaseResult(
                case_id=case.case_id,
                groundedness=groundedness,
                retrieval_hit=retrieval_hit,
                typology_correct=typology_correct,
                quality_score=quality,
                violations=violations,
            )
        )

    report = build_report(results, in_tokens, out_tokens, n_refused)
    print()
    print(report.summary())

    (artifacts / "copilot_eval.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    passed, failures = check_gates(report)
    if not passed:
        print("\nGATE FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nall gates passed")
    return 0


async def _draft_one(case: EvalCase):
    """Call the copilot service directly, bypassing HTTP."""
    import sys as _sys

    api_dir = Path(__file__).resolve().parents[3] / "api"
    if str(api_dir) not in _sys.path:
        _sys.path.insert(0, str(api_dir))

    from app.core.config import Settings
    from app.services.copilot import AnalystCopilot

    copilot = AnalystCopilot(Settings())
    return await copilot.draft(
        risk_score=case.risk_score,
        amount=case.amount,
        decision=case.decision,
        attributions=case.attributions,
        similar_cases=[],
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the analyst copilot")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    p.add_argument(
        "--no-judge", action="store_true",
        help="skip LLM-judge quality scoring (deterministic metrics only)",
    )
    args = p.parse_args()
    sys.exit(asyncio.run(run(args.n, args.artifacts, not args.no_judge)))


if __name__ == "__main__":
    main()
