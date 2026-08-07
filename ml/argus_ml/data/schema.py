"""Canonical data contract for Argus.

This module is the single source of truth for the shape of a transaction.
Both the offline training pipeline and the online scoring API import from
here, which is what keeps train/serve parity honest: if the contract drifts,
both sides break at the same time instead of silently disagreeing.

Nothing in `TransactionLabel` may ever be used as a model feature.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Channel(StrEnum):
    """How the transaction reached the network."""

    CARD_PRESENT = "card_present"
    ECOMMERCE = "ecommerce"
    ATM = "atm"
    TRANSFER = "transfer"


class EntryMode(StrEnum):
    """Physical entry mode. `manual` and `magstripe` carry elevated risk."""

    CHIP = "chip"
    CONTACTLESS = "contactless"
    MAGSTRIPE = "magstripe"
    MANUAL = "manual"
    ONLINE = "online"


class MerchantCategory(StrEnum):
    """Trimmed MCC-style taxonomy. Kept small so one-hot stays tractable."""

    GROCERY = "grocery"
    FUEL = "fuel"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    ELECTRONICS = "electronics"
    TRAVEL = "travel"
    ENTERTAINMENT = "entertainment"
    GAMBLING = "gambling"
    CRYPTO = "crypto"
    MONEY_TRANSFER = "money_transfer"
    UTILITIES = "utilities"
    HEALTHCARE = "healthcare"


class FraudTypology(StrEnum):
    """Attack pattern that generated a fraudulent transaction.

    Analysis and error-slicing only. Never a feature, never an input to the
    API. It exists so we can report recall *per attack type* rather than
    hiding a blind spot behind a single aggregate number.
    """

    NONE = "none"
    CARD_TESTING = "card_testing"
    ACCOUNT_TAKEOVER = "account_takeover"
    CNP_BURST = "cnp_burst"
    GEO_VELOCITY = "geo_velocity"
    BUST_OUT = "bust_out"
    STRUCTURING = "structuring"


class Customer(BaseModel):
    """A cardholder and their learned behavioural baseline."""

    customer_id: str
    home_lat: float
    home_lon: float
    home_country: str
    # Lognormal spend profile — captures "this customer's normal ticket size".
    spend_mu: float
    spend_sigma: float
    # Poisson rate, transactions per day.
    txn_rate: float
    preferred_categories: list[MerchantCategory]
    account_age_days: int
    credit_limit: float


class Merchant(BaseModel):
    """A merchant endpoint. `risk_index` drives the legitimate-traffic mix."""

    merchant_id: str
    category: MerchantCategory
    lat: float
    lon: float
    country: str
    risk_index: float = Field(ge=0.0, le=1.0)


class Transaction(BaseModel):
    """A single authorisation request — the atom of the whole system.

    Field order mirrors the wire format the API accepts, so a payload can be
    validated against this model directly with no translation layer.
    """

    transaction_id: str
    timestamp: datetime
    customer_id: str
    card_id: str
    merchant_id: str
    merchant_category: MerchantCategory
    amount: float = Field(gt=0)
    currency: str = "USD"
    channel: Channel
    entry_mode: EntryMode
    device_id: str
    ip_country: str
    lat: float
    lon: float
    is_recurring: bool = False

    @field_validator("amount")
    @classmethod
    def _round_amount(cls, v: float) -> float:
        return round(v, 2)


class TransactionLabel(BaseModel):
    """Ground truth, available only in the offline world.

    `label_timestamp` models the real-world chargeback delay: a fraud label
    does not arrive at authorisation time, it arrives weeks later. Training
    that ignores this leaks the future into the past.
    """

    transaction_id: str
    is_fraud: bool
    typology: FraudTypology = FraudTypology.NONE
    label_timestamp: datetime | None = None


class ScoreRequest(BaseModel):
    """Inbound scoring payload — a bare transaction, nothing more."""

    transaction: Transaction


class Decision(StrEnum):
    """Terminal action for a scored transaction."""

    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class FeatureAttribution(BaseModel):
    """One SHAP contribution, surfaced to the analyst UI."""

    feature: str
    value: float
    contribution: float
    direction: str  # "increases_risk" | "decreases_risk"


class ScoreResponse(BaseModel):
    """What the API returns. Deliberately verbose — an analyst has to be able
    to justify this decision to a customer, and a regulator may ask later."""

    transaction_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    decision: Decision
    model_version: str
    threshold_version: str
    top_attributions: list[FeatureAttribution]
    anomaly_score: float | None = None
    latency_ms: float
    scored_at: datetime
