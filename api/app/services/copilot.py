"""Analyst copilot — LLM-drafted case narratives grounded in model evidence.

The thing that makes this useful rather than decorative is what it is *not*
allowed to do. The copilot never sees the raw transaction and never decides
anything. It receives:

  1. the SHAP attributions the model actually produced,
  2. the retrieved typology reference for the pattern those attributions match,
  3. summaries of similar historical cases and how analysts resolved them,

and writes the narrative an analyst would otherwise write by hand. Every
factual claim in its output has to trace back to one of those three inputs,
which is what makes the groundedness check in `evaluation.py` meaningful.

This is deliberately a *drafting* tool. The analyst accepts, edits or rejects
it, and that accept-rate is the copilot's real quality metric — not a
benchmark score. `Case.copilot_accepted` is where that signal lands.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Retrieval corpus: fraud typology reference.
#
# Small and hand-written on purpose. A vector store over a scraped corpus
# would be more impressive-sounding and much worse: these six documents are
# the actual decision-relevant knowledge, and keeping them curated means the
# groundedness check has a fixed, auditable ground truth to score against.
# --------------------------------------------------------------------------

TYPOLOGY_REFERENCE: dict[str, dict[str, str]] = {
    "card_testing": {
        "name": "Card testing",
        "signature": (
            "A burst of very low-value authorisations (typically under $5) across "
            "several merchants within minutes, usually card-not-present, from a "
            "device not previously seen on the account."
        ),
        "intent": (
            "The attacker holds a batch of stolen card numbers and is checking "
            "which are still live before selling or using them. The small "
            "amounts are chosen to stay below review thresholds."
        ),
        "analyst_guidance": (
            "Confirm by the ratio of transaction count to distinct merchants "
            "within one hour. Genuine customers rarely transact more than twice "
            "an hour across different merchants at trivial amounts. If "
            "confirmed, the card is compromised regardless of whether any "
            "individual charge succeeded — block and reissue."
        ),
    },
    "account_takeover": {
        "name": "Account takeover",
        "signature": (
            "A new device and new country appear together, followed by "
            "escalating transaction values against high-liquidity merchant "
            "categories (electronics, crypto, money transfer, travel)."
        ),
        "intent": (
            "The attacker has obtained account credentials and is converting "
            "the balance or credit line into resellable goods before the "
            "cardholder notices."
        ),
        "analyst_guidance": (
            "The tell is the *combination* — new device alone is weak evidence "
            "(customers replace phones), and foreign country alone is weak "
            "evidence (customers travel). Together with escalating amounts, it "
            "is strong. Always attempt out-of-band contact with the cardholder "
            "before releasing."
        ),
    },
    "cnp_burst": {
        "name": "Card-not-present burst",
        "signature": (
            "Three to eleven mid-to-large e-commerce authorisations inside an "
            "hour, typically electronics, travel or general retail."
        ),
        "intent": (
            "A spree against a card known to be live, racing the fraud system's "
            "detection window."
        ),
        "analyst_guidance": (
            "Distinguish from legitimate shopping sessions by amount relative "
            "to the customer's own baseline and by whether the merchants are "
            "ones the customer has used before. A genuine session usually "
            "concentrates on one merchant."
        ),
    },
    "geo_velocity": {
        "name": "Impossible travel",
        "signature": (
            "Card-present authorisations at a physical distance the cardholder "
            "could not have covered in the elapsed time. Magstripe entry mode "
            "is a strong corroborating signal."
        ),
        "intent": (
            "The card has been physically cloned. The magstripe fallback occurs "
            "because a cloned card carries no working EMV chip."
        ),
        "analyst_guidance": (
            "Implied ground speed above roughly 900 km/h cannot be explained by "
            "commercial travel. Note that this is one of the few typologies "
            "where the evidence is close to conclusive rather than probabilistic."
        ),
    },
    "bust_out": {
        "name": "Bust-out",
        "signature": (
            "A gradual escalation over days toward the credit limit, with no "
            "single transaction that looks unusual in isolation."
        ),
        "intent": (
            "The account holder — or someone who has held the account long "
            "enough to build trust — intends to extract the full credit line "
            "and default."
        ),
        "analyst_guidance": (
            "This is the hardest typology to confirm from a single "
            "transaction, because each one is individually plausible. Review "
            "the 7-day trajectory of amount-to-limit ratio rather than the "
            "current authorisation. Rising utilisation combined with a change "
            "in merchant category mix is the signal."
        ),
    },
    "structuring": {
        "name": "Structuring",
        "signature": (
            "Repeated transfers pinned just below a reporting threshold "
            "(commonly $10,000), through money-transfer merchants."
        ),
        "intent": (
            "Deliberate evasion of a regulatory reporting requirement — often "
            "money laundering rather than card fraud proper."
        ),
        "analyst_guidance": (
            "Amounts clustering in the $9,000-$9,950 band across multiple "
            "transfers is the signature. This typically requires escalation to "
            "the financial-crime team rather than ordinary fraud disposition."
        ),
    },
}


# Maps the feature names the model produces onto the typology they most
# strongly indicate. This is the "retrieval" step — deterministic and
# inspectable rather than an embedding lookup, because the mapping is known.
_FEATURE_TO_TYPOLOGY: dict[str, str] = {
    "implied_velocity_kmh": "geo_velocity",
    "entry_magstripe": "geo_velocity",
    "threshold_proximity": "structuring",
    "cat_money_transfer": "structuring",
    "merchant_concentration_24h": "card_testing",
    "txn_count_1h": "card_testing",
    "is_new_device": "account_takeover",
    "is_new_country": "account_takeover",
    "is_foreign_country": "account_takeover",
    "amount_to_limit_ratio": "bust_out",
    "amount_sum_7d": "bust_out",
    "amount_zscore": "cnp_burst",
    "channel_ecommerce": "cnp_burst",
}


@dataclass
class CopilotDraft:
    """What the copilot produced, plus everything needed to audit it."""

    summary: str
    likely_typology: str | None
    confidence: str            # "low" | "medium" | "high"
    recommended_action: str
    evidence_cited: list[str]
    retrieved_docs: list[str]
    model: str
    input_tokens: int
    output_tokens: int
    refused: bool = False


def retrieve_context(
    attributions: list[dict[str, Any]],
    top_k: int = 2,
) -> tuple[list[str], list[str]]:
    """Select typology reference docs relevant to this case's attributions.

    Returns (doc_keys, rendered_docs). Scoring is by summed absolute SHAP
    contribution of the features that map to each typology, so the retrieved
    context is a function of what the model actually keyed on rather than of
    keyword overlap with free text.
    """
    scores: dict[str, float] = {}
    for attr in attributions:
        typ = _FEATURE_TO_TYPOLOGY.get(attr["feature"])
        if typ is None:
            continue
        # Only risk-increasing contributions indicate a typology.
        if attr.get("contribution", 0) <= 0:
            continue
        scores[typ] = scores.get(typ, 0.0) + abs(float(attr["contribution"]))

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    keys = [k for k, _ in ranked]

    docs = []
    for key in keys:
        ref = TYPOLOGY_REFERENCE[key]
        docs.append(
            f"## {ref['name']}\n"
            f"**Signature:** {ref['signature']}\n"
            f"**Attacker intent:** {ref['intent']}\n"
            f"**Analyst guidance:** {ref['analyst_guidance']}"
        )
    return keys, docs


_SYSTEM_PROMPT = """\
You are a fraud analyst assistant. You draft the case narrative that a human \
analyst reviews before making a decision.

You are given three things: the risk model's SHAP feature attributions for one \
transaction, reference documentation on fraud typologies, and summaries of \
similar historical cases.

Rules:
- Every factual claim you make must be traceable to the attributions, the \
reference docs, or the historical cases you were given. Do not introduce \
details that are not in your inputs.
- You are drafting, not deciding. The analyst decides.
- If the evidence is weak or contradictory, say so and set confidence to "low". \
An honest "the signals here are ambiguous" is more useful than a confident \
narrative built on thin evidence.
- Write for an analyst who is working a queue under time pressure. Lead with \
what happened, then the evidence, then what you would check next.
- Do not speculate about the cardholder's character or circumstances.\
"""

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": (
                "Two to four sentences. What the model flagged and why, in plain "
                "language an analyst can paste into a case note."
            ),
        },
        "likely_typology": {
            "type": "string",
            "enum": [
                "card_testing", "account_takeover", "cnp_burst",
                "geo_velocity", "bust_out", "structuring", "unclear",
            ],
        },
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_action": {
            "type": "string",
            "description": "One sentence: what the analyst should check or do next.",
        },
        "evidence_cited": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Feature names from the attributions that support the summary.",
        },
    },
    "required": [
        "summary", "likely_typology", "confidence",
        "recommended_action", "evidence_cited",
    ],
    "additionalProperties": False,
}


def _render_attributions(attributions: list[dict[str, Any]]) -> str:
    lines = []
    for a in attributions:
        direction = "increases" if a["contribution"] > 0 else "decreases"
        lines.append(
            f"- {a['feature']} = {a['value']} "
            f"({direction} risk, SHAP {a['contribution']:+.4f})"
        )
    return "\n".join(lines) or "(no attributions available)"


def _render_similar_cases(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return "(no similar resolved cases on file)"
    lines = []
    for c in cases:
        verdict = (
            "confirmed fraud" if c.get("analyst_verdict")
            else "cleared as legitimate" if c.get("analyst_verdict") is False
            else "unresolved"
        )
        lines.append(
            f"- ${c.get('amount', 0):,.2f}, model score "
            f"{c.get('risk_score', 0):.2f} -> {verdict}"
            + (f'. Analyst noted: "{c["resolution_note"]}"' if c.get("resolution_note") else "")
        )
    return "\n".join(lines)


class AnalystCopilot:
    """Drafts case narratives with Claude, grounded in retrieved evidence."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key or None)
        return self._client

    async def draft(
        self,
        risk_score: float,
        amount: float,
        decision: str,
        attributions: list[dict[str, Any]],
        similar_cases: list[dict[str, Any]] | None = None,
        anomaly_score: float | None = None,
    ) -> CopilotDraft | None:
        """Produce a case narrative, or None if the copilot is unavailable."""
        if not self.settings.copilot_enabled:
            return None

        doc_keys, docs = retrieve_context(attributions)
        similar_cases = similar_cases or []

        user_content = f"""\
# Transaction under review

Model risk score: {risk_score:.4f}
Engine decision: {decision}
Amount: ${amount:,.2f}
{f"Unsupervised anomaly score: {anomaly_score:.4f}" if anomaly_score is not None else ""}

# Model attributions (SHAP)

{_render_attributions(attributions)}

# Relevant typology reference

{chr(10).join(docs) if docs else "(no typology matched the attributions)"}

# Similar resolved cases

{_render_similar_cases(similar_cases)}

Draft the case narrative."""

        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self.settings.llm_model,
                max_tokens=2000,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        # The system prompt and typology reference are stable
                        # across every case, so caching them turns the
                        # per-case cost into just the transaction payload.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.settings.llm_effort,
                    "format": {"type": "json_schema", "schema": _OUTPUT_SCHEMA},
                },
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception:
            # The copilot is an assistive layer. If it fails, the analyst still
            # has the score, the decision and the SHAP attributions — which is
            # everything they need to work the case.
            logger.exception("copilot draft failed; serving case without narrative")
            return None

        if response.stop_reason == "refusal":
            logger.warning(
                "copilot refused: %s",
                getattr(response.stop_details, "category", "unknown"),
            )
            return CopilotDraft(
                summary="", likely_typology=None, confidence="low",
                recommended_action="", evidence_cited=[],
                retrieved_docs=doc_keys, model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                refused=True,
            )

        text = next((b.text for b in response.content if b.type == "text"), "{}")
        payload = json.loads(text)

        return CopilotDraft(
            summary=payload["summary"],
            likely_typology=(
                None if payload["likely_typology"] == "unclear"
                else payload["likely_typology"]
            ),
            confidence=payload["confidence"],
            recommended_action=payload["recommended_action"],
            evidence_cited=payload["evidence_cited"],
            retrieved_docs=doc_keys,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


__all__ = ["AnalystCopilot", "CopilotDraft", "retrieve_context", "TYPOLOGY_REFERENCE"]
