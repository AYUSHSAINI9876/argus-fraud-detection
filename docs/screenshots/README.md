# Screenshot capture checklist

The README references six images from this directory. They are **not committed
as placeholders** — an empty or mocked screenshot is worse than none, because a
reviewer who clicks through and finds a stub trusts the rest of the repo less.

Capture them once the stack is running with real traffic, then commit.

## Prerequisites

```bash
# 1. artefacts must exist (they are committed, so this is usually a no-op)
ls ml/artifacts/xgboost_risk.joblib

# 2. bring up the stack
docker compose up --build

# 3. feed it real traffic — without this every page is empty
cd ml && python -m argus_ml.replay --rate 40 --limit 8000
```

Let the replay finish. The dashboard aggregates over a time window, so
screenshots taken mid-replay will show a half-filled chart.

## Before capturing

- **Use dark mode.** The console is designed dark-first; the light variant
  exists for accessibility, not for marketing.
- **Window at 1440×900 or wider.** Narrower and the case-detail side rail wraps
  under the main column.
- **Zoom at 100%.** Browser zoom produces soft text that looks like a
  compression artefact in a README.
- **Work two or three cases first** (claim → note → dispose) so the audit log
  and realised-precision tile have content. An audit log screenshot with one
  row does not demonstrate anything.

## The six shots

| File | Page | Frame it so this is visible |
|---|---|---|
| `01-overview.png` | `/` | All four stat tiles **and** the decision-volume chart. The p99 latency figure is the detail people look for. |
| `02-queue.png` | `/queue` | The sort control plus at least 8 rows, so the expected-loss ordering is legible — ideally with a lower-risk/higher-amount row ranked above a higher-risk/small one. That row *is* the argument. |
| `03-case-detail.png` | `/cases/[id]` | The risk badge, the rationale block, and the full SHAP attribution list including at least one mitigating factor. Pick a case with a `geo_velocity` or `account_takeover` signature — the drivers read clearly. |
| `04-model-health.png` | `/models` | The comparison table with the baseline row **above** the champion, plus the per-typology recall panel showing bust-out lowest. |
| `05-policy.png` | `/policy` | Current parameters and the break-even curve side by side. |
| `06-audit.png` | `/audit` | Several rows with different action types — ideally including a `case.release_block`, which is the one that moves money. |

## Format

- **PNG**, not JPEG — screenshots of text compress badly as JPEG.
- Keep each **under ~400 KB**. If a shot is larger, the window is too big;
  resize rather than compressing to mush.
- Do not annotate with arrows or callouts. The README prose already says what
  to look at, and annotations date badly.

## A note on the data

Every figure in these screenshots comes from the synthetic generator. That is
worth saying out loud in any writeup that uses them — the numbers are real
outputs of a real pipeline, but the underlying fraud is simulated.
