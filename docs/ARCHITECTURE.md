# Architecture

This document explains *why* Argus is shaped the way it is. The README covers
what it does; this covers the decisions, the trade-offs taken, and the things
that are deliberately not solved yet.

---

## The central problem

A fraud engine has to satisfy four constraints simultaneously, and most of the
interesting design pressure comes from the fact that they pull against each
other:

| Constraint | Consequence |
|---|---|
| **Decide in milliseconds** | No expensive lookups on the request path; state must be pre-aggregated |
| **Decide correctly** | Needs behavioural history, which is expensive to compute |
| **Explain every decision** | Attribution must run inline, not as an offline batch job |
| **Never drift from training** | Online and offline feature computation must be provably identical |

Constraints 1 and 2 conflict directly. The resolution is the bounded rolling
state described below.

---

## Feature store: the parity guarantee

**The single most common production ML bug is training/serving skew.** The
offline pipeline computes a feature one way — a pandas `groupby` over the whole
table — and the online service computes it another way, per request, from a
cache. The two quietly disagree, model quality degrades, and nobody can explain
why because both code paths look correct in isolation.

Argus makes this structurally impossible rather than a matter of discipline.
There is exactly one function:

```python
compute_features(txn: dict, state: CustomerState, ctx: EntityContext) -> dict[str, float]
```

Both worlds call it:

```
OFFLINE                                    ONLINE
build_offline_features()                   ScoringService.score()
  replays the sorted stream                  loads CustomerState from Redis
  updating CustomerState per row             calls compute_features()
  calling compute_features()                 scores, writes state back
```

Because `CustomerState` is the same class in both paths, a feature can only be
computed from information that genuinely preceded the transaction. Leakage
becomes hard to write rather than something you have to remember not to do.

Three enforcement points back this up:

1. `feature_names()` derives the canonical ordering from a probe transaction.
   `train.py` asserts the training matrix matches it.
2. The trained model persists its `feature_names`. `ModelBundle.load()` asserts
   the artefact and the runtime contract agree, and **refuses to start** if not.
3. `test_offline_online_parity` replays a generated stream one transaction at a
   time — exactly as the API would — and asserts equality to 1e-9 against the
   offline matrix.

The offline pass is a single-threaded Python loop rather than a vectorised
groupby. That is slower, and it is an accepted trade: guaranteed parity is
worth far more than the wall-clock saving.

### Why the state is bounded

`CustomerState` holds only events inside the widest feature window (7 days),
plus Welford accumulators for lifetime mean/variance and a set of seen
devices/countries/categories. Memory is flat regardless of how long someone has
been a customer, which is what makes a Redis round trip viable at low latency.

Welford rather than storing all amounts: numerically stable, O(1) memory, and
lets us keep a lifetime baseline without keeping lifetime history.

---

## Why three models

| Model | Catches | Blind to |
|---|---|---|
| Logistic regression | Linear signal; fully interpretable | Interactions |
| XGBoost | Everything in the labelled distribution | Anything never labelled |
| Gaussian anomaly detector | Whatever is far from normal | Fraud that looks normal |

The critical asymmetry is the third row. **Chargebacks arrive ~3 weeks late.**
A novel attack launched today is, by construction, absent from the supervised
training set — and XGBoost will score it confidently benign. Density estimation
has no such blind spot: it learns what normal looks like from the overwhelming
majority class and flags low-probability regions regardless of labels.

The two fail in uncorrelated ways, which is the argument for running both.

**It is not, on this data, an argument for blending them.** The 0.85/0.15
ensemble was built and measured, and it lost: PR-AUC 0.9588 against the
champion's 0.9720, with calibration three orders of magnitude worse
(ECE 0.1092 vs 0.0001). Uncorrelated failure modes only help when both
components are individually useful, and the anomaly detector scores 0.0486
PR-AUC — averaging it in drags the champion down rather than adding a floor.

So the ensemble is evaluated on every run and **deliberately not promoted**.
The anomaly detector stays in the system as a *parallel* signal that routes to
review on unlabelled novelty, which is the role its 3-week label-delay argument
actually justifies. Keeping the losing blend in the report is intentional: the
negative result is the interesting part, and removing it would hide the
reasoning.

---

## Decision policy: why not a threshold

A calibrated probability is not a decision. Consider two transactions both
scored at **0.05**, under the default cost model (chargeback fee $25, review
cost $3.50, review leakage 12%):

| | $12 coffee | $4,000 laptop |
|---|---|---|
| E[cost \| allow] | 0.05 × $37 = **$1.85** | 0.05 × $4,025 = $201.25 |
| E[cost \| review] | $3.50 + $0.22 = $3.72 | $3.50 + $24.15 = **$27.65** |
| Decision | **allow** | **review** |

Same score, opposite action. A single global threshold cannot express that.

Note the score in this example is 5%, not the 40% that reads more
intuitively as "risky". That distinction is load-bearing and was found by
the policy tests rather than by inspection: at a $25 chargeback fee,
`E[cost | allow] ≥ 0.4 × $25 = $10` before the transaction value is counted
at all, which already exceeds the $3.50 review cost. Solving
`E[allow] < E[review]` for the $12 ticket gives **p < 0.1075** — so at 40%
risk there is *no* amount, however small, that resolves to allow. An earlier
draft of this document asserted otherwise.

Argus computes expected cost per action and takes the minimum, subject to two
guard rails:

- **`min_block_score`** — never auto-block on a weak score even when the amount
  makes the arithmetic favour it. Blocking a genuine customer carries
  reputational cost the linear model understates.
- **`max_review_rate`** — the review queue cannot exceed what the analyst team
  can actually work. Above that rate, only reviews whose expected saving
  justifies an analyst slot survive.

This is also the seam where the reinforcement-learning policy lands in a later
phase. `decide()` already has the signature a learned policy would take, so
swapping the analytic rule for a trained agent touches nothing else.

### Why calibration is load-bearing here

Because the policy computes expected cost *from the score*, an uncalibrated
0.9 that really means 0.4 corrupts every decision downstream. This is why
`XGBoostRiskModel` fits a calibrator on a disjoint temporal slice, and why the
calibration method is chosen by positive count:

- **≥500 positives** → isotonic (non-parametric, tracks any shape)
- **<500 positives** → Platt scaling (two parameters, cannot collapse)

That threshold exists because of a bug found while building this. With too few
positives, isotonic degenerates into a near-constant step function: thousands
of distinct scores collapse onto a handful of values, ranking resolution is
destroyed, and PR-AUC craters — even though the transform is technically
monotone and "should" preserve ranking. Platt scaling cannot tie scores, so it
is the safe choice on thin data.

---

## Evaluation: the metrics that were deliberately not used

| Not used | Why |
|---|---|
| Accuracy | "Never fraud" scores 99.8% |
| ROC-AUC (as headline) | At 0.2% positives it is dominated by the negative class; reported only for comparability |
| F1 at 0.5 | Nobody operates at 0.5 on a 0.2% base rate |

What is used instead:

- **PR-AUC** — the honest summary at this class balance
- **Precision and recall at analyst capacity** — the operating point a real team lives at
- **Recall per typology** — an aggregate of 0.70 can hide an attack type caught 0% of the time
- **Calibration (Brier, ECE)** — because the policy consumes probabilities
- **Net savings** — money prevented, less friction cost, less analyst time

One caveat on the calibration metrics, learned from the shipped run: Brier and
ECE are count-weighted, and at a 0.2% base rate ~99.8% of rows sit in the
lowest score bin where every model is already exact. That bin swamps the
average, so aggregate ECE reports almost no difference between a calibrated and
an uncalibrated model even when the high-score bins differ by 2×. Per-bin
reliability is the instrument that actually shows it.

### Temporal splitting and label delay

Fraud is bursty: one compromised card produces a cluster of correlated
transactions within minutes. A random split scatters that cluster across train
and test, the model memorises the episode, and the test score inflates — often
by 20+ PR-AUC points. Argus splits strictly by time with a 7-day embargo
between splits so 7-day velocity windows cannot straddle a boundary.

**Label delay** is enforced explicitly: at the training cut-off, fraud whose
chargeback would not yet have arrived is masked to negative, exactly as
production would have seen it. Training on labels from the future is the
subtlest form of leakage in fraud, and it produces offline numbers that
production never reproduces.

---

## GenAI layer: what the copilot is not allowed to do

The copilot never sees the raw transaction and never decides anything. It
receives the SHAP attributions the model produced, the retrieved typology
reference matching those attributions, and summaries of similar resolved
cases — then writes the narrative an analyst would write by hand.

Retrieval is deterministic (a feature→typology mapping scored by SHAP
contribution) rather than an embedding lookup. For a six-document corpus where
the mapping is known, a vector store would be more impressive-sounding and
strictly worse: the deterministic version is auditable, and the groundedness
check has a fixed ground truth to score against.

**The eval harness is the point.** Four questions, four methods:

| Question | Method |
|---|---|
| Are cited features real? | Deterministic set membership — no judge needed |
| Was the right doc retrieved? | Ground truth from the generator |
| Was the typology right? | Exact match |
| Is the prose useful? | LLM-as-judge — the only dimension without programmatic truth |

The judge is calibrated against human labels before it is trusted, and
`calibrate_judge` reports a `usable` flag that goes false below 80% within-one
agreement. A judge that agrees with humans 60% of the time is not a measurement
instrument, and reporting its scores as if it were is worse than reporting
nothing.

CI gates on **groundedness ≥ 0.95**. A copilot that fabricates evidence in a
system that blocks people's cards is not shippable at any quality score.

---

## Security model

| Concern | Approach |
|---|---|
| Token theft via XSS | Access token stays in an httpOnly cookie; client components call Next.js route handlers that attach it server-side |
| Forged identity | API verifies JWTs against Stack Auth's JWKS independently — never trusts a client header |
| Auth-provider outage | Signature checked locally against a cached public key; no call to Stack Auth on the request path |
| Privilege escalation | Four-tier RBAC; block release is REVIEWER+, policy and model changes are ADMIN |
| Open proxy | Route handler enforces an action allow-list |
| Unprotected production | `auth_enabled=False` raises at startup when `ENVIRONMENT=production` |
| Repudiation | Append-only audit log; the repository exposes only an insert |

---

## Known limitations

Stated rather than hidden — these are the questions an interviewer will ask.

**Policy config is in-process.** `PolicyConfig` lives on `app.state`. A
multi-replica deployment needs it in Redis with pub/sub invalidation; today a
threshold change applies only to the replica that served the request. Single
node is correct; horizontal scale is not.

**Challenger comparison is biased.** The case queue only contains what the
champion flagged, so analyst-resolved labels are not a random sample of
traffic. The endpoint reports this in its own response payload and treats the
comparison as directional, never as a promotion criterion on its own. Doing
this properly needs a small randomised holdout that bypasses the champion.

**Batch scoring is serialised per customer** to keep rolling state consistent,
which caps batch throughput. Correct fix: partition by customer and parallelise
across partitions.

**Redis state has no cross-region story.** A customer whose transactions land
in two regions would have two divergent velocity windows.

**The synthetic data is my model of fraud.** It is realistic in structure —
diurnal rhythms, per-customer spend profiles, six typologies with genuinely
distinct signatures, chargeback delay — but real portfolios contain patterns
nobody has thought to simulate. That is precisely why the unsupervised layer
exists, and it is also why the generator is a starting point rather than a
claim about real-world performance.

**Anomaly detector fit is O(n·d²).** The full-covariance Mahalanobis distance
over 1.2M rows × 58 features is the slowest step in the pipeline by a wide
margin. The diagonal fallback is far cheaper and loses little; the full
covariance is kept because modelling feature correlation is the point of using
a multivariate Gaussian at all.

---

## What comes next

| Phase | Work |
|---|---|
| v0.7 | Randomised holdout for unbiased challenger evaluation |
| v0.8 | Policy config to Redis + pub/sub for multi-replica |
| v0.9 | Automated retraining triggered by drift, with promotion gated on the offline suite |
| v1.0 | RL decision policy under analyst-capacity budget |
