<div align="center">

# 🛡️ Argus

**Real-time transaction fraud detection with explainable decisions, shadow-mode model evaluation, and a full audit trail.**

[![CI](https://github.com/ayushkumarsaini/Argus/actions/workflows/ci.yml/badge.svg)](https://github.com/ayushkumarsaini/Argus/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-15-000000?logo=nextdotjs&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-EB0F00)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## What this is

Argus scores card transactions for fraud risk in real time, decides whether to
allow / review / block them by **expected cost**, explains every decision with
SHAP attributions, and routes the borderline ones to a human analyst queue.

It is built as a complete platform rather than a notebook: authentication with
role-based access control, a feature store with genuine train/serve parity,
shadow-mode challenger evaluation, drift monitoring, an append-only audit log,
and an analyst console.

## The console

<div align="center">

### Overview — live decisioning
![Overview dashboard](docs/screenshots/01-overview.png)

</div>

Volume, block and review rates, p95/p99 scoring latency, and **realised
precision** — the share of analyst-worked alerts that turned out to be genuine
fraud. Offline PR-AUC says what the model *should* do; realised precision says
what it *is* doing, and the gap between them is the number worth watching.

<div align="center">

### Case queue — ranked by expected loss
![Case queue](docs/screenshots/02-queue.png)

</div>

The ordering is the point. The queue sorts by **risk × amount**, not by risk
score: an analyst hour spent on a 0.92-risk $14 charge is an hour not spent on a
0.61-risk $8,400 one. Sorting a fraud queue by score alone is one of the most
common and most expensive mistakes in the domain.

<div align="center">

### Case detail — why this transaction was flagged
![Case detail with SHAP attributions](docs/screenshots/03-case-detail.png)

</div>

Exact SHAP attributions split into risk drivers and mitigating factors, with
feature names translated into language an analyst can paste into a case note.
Every decision is answerable, which is what makes the platform deployable at all.

<div align="center">

### Model health — the baseline sits above the champion
![Model health](docs/screenshots/04-model-health.png)

</div>

PR-AUC as the headline, calibration reliability, and **recall broken out per
attack type** — because an aggregate of 0.70 can hide a typology the model
catches none of.

<details>
<summary><b>Policy and audit log</b></summary>

<div align="center">

![Decision policy](docs/screenshots/05-policy.png)
![Audit log](docs/screenshots/06-audit.png)

</div>

The policy page exposes the break-even curve so a risk manager can read the
decision boundary rather than trust a black box. The audit log is append-only —
the repository exposes no update or delete path — and records every privileged
action with the actor's verified subject claim.

</details>

> Screenshots are captured after `docker compose up` plus a replay run; see
> [`docs/screenshots/`](docs/screenshots/) for the capture checklist.

## Why it is built the way it is

Most portfolio fraud projects follow the same path — load a Kaggle CSV, train
XGBoost, report 99.8% accuracy, stop. Every one of those steps is a mistake
that a fraud interview will find in ten minutes. Argus takes the opposite
position on each:

| Common approach | What Argus does | Why it matters |
|---|---|---|
| Static PCA-anonymised CSV | Generated transaction **stream** with 6 real fraud typologies | `V1..V28` features cannot be engineered or explained; a stream lets velocity features, drift and walk-forward validation actually exist |
| Random train/test split | **Temporal split** with a 7-day embargo and label-delay masking | Fraud is bursty; a random split scatters one compromised card across train and test, inflating scores by 20+ PR-AUC points |
| Train on all labels | Labels **masked until the chargeback would have arrived** | Chargebacks land ~3 weeks late. Training on labels that did not exist yet is the subtlest leak in fraud |
| Report accuracy / ROC-AUC | **PR-AUC**, precision at analyst capacity, recall **per typology** | At this run's 0.2% positive rate, "never fraud" scores 99.8% accuracy and ROC-AUC flatters a useless model |
| Single global threshold | **Expected-cost decision policy** | 5% risk on a $12 coffee is allowed; the same 5% on a $4,000 laptop is reviewed. One threshold cannot express that |
| Offline pandas features, separate serving code | **One `compute_features()`** called by trainer and API alike | Train/serve skew is the most common production ML bug; here it is a startup crash, not a slow bleed |
| Raw model output as probability | **Calibrated** with method chosen by positive count | The policy computes expected cost from the score — an uncalibrated 0.9 that means 0.4 corrupts every decision downstream |

## Architecture

```
                    ┌──────────────────────────────────────┐
   transaction ───► │  FastAPI scoring service             │
                    │                                      │
                    │  1. load rolling state ──► Redis     │
                    │  2. compute_features()  ◄── shared   │
                    │  3. assert feature contract          │
                    │  4. champion score  + challenger     │
                    │     (shadow, never acted on)         │
                    │  5. Gaussian anomaly score           │
                    │  6. expected-cost policy             │
                    │  7. SHAP attribution                 │
                    │  8. persist state + audit record     │
                    └───────────┬──────────────────────────┘
                                │
             allow  ◄───────────┼───────────►  block
                                │
                             review
                                ▼
                    ┌──────────────────────────────────────┐
                    │  Next.js analyst console             │
                    │  Stack Auth · RBAC · case queue      │
                    │  ranked by EXPECTED LOSS             │
                    └──────────────────────────────────────┘
                                │
                    analyst verdict ──► fast label ──► retraining loop
```

### The models, and why each exists

| Model | Role | Course concept |
|---|---|---|
| **Logistic regression** | Interpretable baseline every other model must beat; also the real production challenger in regulated settings | C1 — regularisation, feature scaling, class weighting |
| **XGBoost** (calibrated) | Production champion. Still state of the art on tabular fraud | C2 — decision trees, information gain, tree ensembles |
| **Gaussian anomaly detector** | Unsupervised layer. Catches novel attacks that have **no labels yet**, where the supervised model is structurally blind | C3 — density estimation, ε selection by F1 |
| **Ensemble** | 0.85 × supervised + 0.15 × anomaly — measured, and **not promoted**; see Results | Uncorrelated failure modes |

The anomaly detector is implemented from the mathematics rather than pulled
from a library, because the multivariate Gaussian fit and ε-selection are the
substance of the technique — and the per-feature deviation breakdown used in
the UI falls out of it for free.

### Fraud typologies simulated

Each has a distinct signature, so different model families catch different
attacks — which is the entire argument for running more than one.

| Typology | Signature | Hard to catch because |
|---|---|---|
| **Card testing** | 6–22 micro-charges, minutes apart, new device | — (easy; velocity features nail it) |
| **CNP burst** | 3–11 mid-size e-commerce charges inside an hour | — |
| **Account takeover** | New device + new country, escalating amounts | Looks like travel |
| **Geo-velocity** | Physically impossible travel, magstripe fallback | Needs `implied_velocity_kmh` |
| **Bust-out** | Slow build over days, then drain toward credit limit | **Every single transaction looks normal**; only the trajectory betrays it |
| **Structuring** | Repeated transfers pinned just under $10k | Needs `threshold_proximity` |

## Results

> Numbers below are produced by `python -m argus_ml.train` and written to
> `ml/artifacts/evaluation.json`. They are regenerated on every run — nothing
> here is hand-copied.

**Run:** 5,000 customers × 120 days → 806,945 transactions, 1,393 fraud
(0.173%). Temporal split with a 7-day embargo: train 484,477 / val 120,910 /
test 107,685. 242 training frauds masked by label delay. Test window carries
216 fraud (0.201%).

| Model | PR-AUC | ROC-AUC | Brier | ECE | P@capacity | R@capacity | Savings/1k |
|---|---|---|---|---|---|---|---|
| Logistic baseline | 0.8932 | 0.9931 | 0.00400 | 0.0076 | 0.3829 | 0.9537 | $689.26 |
| **XGBoost (calibrated)** | **0.9720** | 0.9961 | 0.00013 | 0.0001 | 0.3922 | 0.9769 | **$708.01** |
| XGBoost (uncalibrated) | 0.9720 | 0.9961 | 0.00013 | 0.0002 | 0.3922 | 0.9769 | $708.01 |
| Gaussian anomaly | 0.0486 | 0.8112 | 0.60261 | 0.7277 | 0.0645 | 0.1620 | $553.95 |
| Ensemble (0.85/0.15) | 0.9588 | 0.9759 | 0.01368 | 0.1092 | 0.3866 | 0.9630 | $705.25 |

XGBoost/baseline PR-AUC lift: **1.09×**. Champion training takes 186 boosting
rounds before early stopping on `aucpr`.

### Recall by typology, at analyst capacity

| Typology | Logistic | XGBoost |
|---|---|---|
| Account takeover | 1.000 | 1.000 |
| CNP burst | 1.000 | 1.000 |
| Geo-velocity | 1.000 | 1.000 |
| Structuring | 1.000 | 1.000 |
| Card testing | 0.992 | 1.000 |
| **Bust-out** | **0.690** | **0.828** |

Bust-out is the one typology neither model closes, and that is the expected
result rather than a defect: every individual bust-out transaction is
unremarkable, and only the multi-day trajectory betrays it. It is the strongest
argument in this repo for sequence features over per-transaction features.

### Three findings this run produced that are worth stating plainly

**1. The ensemble is worse than the champion, so it is not the champion.**
Blending 0.85 × XGBoost + 0.15 × anomaly *lowers* PR-AUC (0.9588 vs 0.9720) and
degrades calibration by three orders of magnitude (ECE 0.1092 vs 0.0001).
Averaging a 0.049-PR-AUC signal into a 0.972 one drags it down — the failure
modes are uncorrelated, but that only helps when both components are
individually useful. The anomaly detector therefore stays in the system as a
**parallel signal that routes to review**, and does not enter the champion
score. Shipping the ensemble because "ensembles are better" would have cost
real precision.

**2. Aggregate calibration metrics hide the effect they are supposed to show.**
Calibrated and uncalibrated report near-identical Brier and ECE, which looks
like calibration did nothing. It is an artefact of count-weighting: 99.79% of
test rows fall in the lowest score bin, where both models are already exact, so
that bin dominates the average. In the bins where review decisions actually get
made the gap is real — at [0.93, 1.00) the calibration error is 0.0028 vs
0.0076, and at [0.80, 0.87) it is 0.066 vs 0.158. The lesson is about the
metric, not the model: report reliability per bin, not one number.

**3. PR-AUC of 0.972 is a property of the simulator, not evidence of a great model.**
Real card portfolios sit far lower. The generator's fraud is separable because
it was written to be, and no claim about real-world performance follows from
this number. The relative comparisons — XGBoost over logistic, per-typology
gaps, the ensemble result above — are what transfer; the absolute figure does
not.

### Top features by gain

| Feature | Gain share |
|---|---|
| `distance_from_home_km` | 31.9% |
| `known_device_count` | 15.4% |
| `txn_count_1h` | 5.6% |
| `log_amount` | 5.6% |
| `is_new_device` | 4.8% |
| `amount` | 4.4% |

## Feature engineering

58 features across six families, all computed from state that strictly
precedes the transaction:

- **Intrinsic** — amount, log-amount, hour, weekend/night, merchant risk index
- **Customer-relative** — z-score vs the customer's own spend distribution, ratio to credit limit, `threshold_proximity`
- **Velocity** — counts and sums over 1h / 24h / 7d, merchant concentration
- **Novelty** — new device / country / category, known-device count
- **Geo** — distance from home, distance from previous, **implied ground speed**
- **Categorical** — fixed-vocabulary one-hots (stable names across retrains, so SHAP output never shifts meaning between model versions)

## Quickstart

```bash
git clone https://github.com/<you>/Argus.git && cd Argus

# 1. Train (generates data, trains all models, writes artefacts)
cd ml && uv venv && uv pip install -e ".[dev]"
python -m argus_ml.train --customers 8000 --days 180

# 2. Bring up the stack
cd .. && cp .env.example .env      # fill in Stack Auth keys
docker compose up --build

# 3. Feed it live traffic (otherwise the console is an empty shell)
cd ml && python -m argus_ml.replay --rate 20 --limit 5000

# API      http://localhost:8000/docs
# Console  http://localhost:3000
```

The replay streams **test-window** transactions — the ones no model was
trained on — through the real HTTP path, with exponentially distributed
inter-arrival times. A constant rate would make the latency percentiles on the
dashboard look better than they deserve to.

### Auth and the build

`AUTH_ENABLED=false` lets the **API** run without a Stack Auth project, and the
API force-rejects that flag when `ENVIRONMENT=production`, so it cannot become
the reason production is unprotected.

It does **not** cover the console. `web/stack.ts` constructs `StackServerApp` at
module scope, so every route that imports it — including the root layout, and
therefore `/_not-found` — throws without credentials, and `next build` fails at
page-data collection with *"you haven't provided a project ID"*. The three
`NEXT_PUBLIC_STACK_*` / `STACK_SECRET_SERVER_KEY` values must be present in
`web/.env.local` (or the deployment's env) **for the build itself**, not only at
runtime. Set them before `npm run build` and before deploying to Vercel.

## Roles

| Role | Can |
|---|---|
| `VIEWER` | Read dashboards and aggregate metrics |
| `ANALYST` | Work the queue, claim cases, add notes, set dispositions |
| `REVIEWER` | Everything above, plus overturn blocks and release funds |
| `ADMIN` | Everything above, plus change policy thresholds and promote models |

Threshold changes and model promotion are ADMIN-only because they move money.
Every such action writes an audit row with the actor's verified subject claim.

## Project layout

```
argus/
├── ml/                        offline: generate, featurise, train, evaluate
│   └── argus_ml/
│       ├── data/              schema (shared contract) + stream generator
│       ├── features/          compute_features() — the parity guarantee
│       ├── models/            baseline · gbdt · anomaly
│       ├── evaluation/        PR-AUC, calibration, cost curves, per-typology
│       └── monitoring/        drift detection
├── api/                       online: FastAPI scoring + case management
│   └── app/
│       ├── core/              config · auth (JWKS) · db · redis state store
│       ├── services/          scoring · policy · copilot
│       └── routers/           score · cases · metrics · admin · health
├── web/                       Next.js analyst console
└── infra/                     compose · CI · migrations
```

## Known limitations

Stated rather than hidden — these are the questions an interviewer will ask,
and having the answer ready is worth more than pretending they do not exist:

- **Policy config is in-process.** A multi-replica deployment needs it in Redis with pub/sub invalidation. Single-node today.
- **Challenger comparison is biased.** The case queue only contains what the champion flagged, so analyst-resolved labels are not a random sample. Reported as directional, not as a promotion criterion.
- **Synthetic data.** Realistic in structure, but the generator's fraud is *my* model of fraud. Real portfolios contain patterns nobody has thought to simulate — which is precisely why the unsupervised layer exists.
- **Batch scoring is serialised per customer** to keep state consistent, capping batch throughput.

## Licence

MIT
