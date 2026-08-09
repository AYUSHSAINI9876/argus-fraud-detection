# Screenshots

These are referenced by the root `README.md`. Capture them **after** bringing
the stack up and replaying traffic, otherwise every panel renders its empty
state and the screenshots undersell the project.

```bash
docker compose up --build
cd ml && python -m argus_ml.replay --rate 20 --limit 5000
```

Then capture at **1600×1000**, dark theme, browser chrome cropped out.

| File | Route | Frame it on |
|---|---|---|
| `01-overview.png` | `/` | The four stat tiles plus the decision-volume chart with real traffic in it |
| `02-queue.png` | `/queue` | The expected-loss column — that ordering is the point of the page |
| `03-case-detail.png` | `/cases/<id>` | A case whose SHAP bars show both risk drivers and mitigating factors |
| `04-model-health.png` | `/models` | The comparison table with the baseline row visible above the champion |
| `05-policy.png` | `/policy` | The break-even curve |
| `06-audit.png` | `/audit` | A few rows with before → after diffs, ideally including a block release |

Pick a case for `03` that has a genuinely interesting story — an account-takeover
pattern with `is_new_device` and `is_new_country` both firing reads far better
than a borderline one.

Keep each file under ~400 KB (PNG, or JPEG at quality 85) so the repo stays
light.
