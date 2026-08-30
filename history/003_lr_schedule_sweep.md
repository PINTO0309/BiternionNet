# 003: Epoch budget and lr-schedule sweep (constant 1000 -> cosine 100)

Created 2026-08-30 (user requests, in order: add a WSD schedule; add "constant until the loss plateaus,
then cosine" with manual timing; "isn't 200 epochs too few?"; "fix 1000 constant epochs, then cosine").
All runs: `towncentre-biternion-vonmises` (fixed as in 001), seed 0, `data/towncentre/manifest.jsonl`,
single model, centre crop at test time. Metric = test MAAD (deg) from `history.jsonl` / `last.pt`.

## 0. How to read

- The paper trains 50 epochs at a constant AdaDelta step = 4 000 steps = **100 epochs of this port** (001 §3).
- Epoch-to-epoch test MAAD fluctuates by +-2-3 deg at a constant lr (443 test heads), so windows of 50 epochs
  (2 000 steps) are reported as mean +- std, plus the window minimum and the final epoch.
- The paper's Table 3 value for Biternion + von Mises is 20.8 +- 24.7 deg (average of 5 networks, with TTA and a
  post-training BN statistics pass; different random person split). Comparisons are indicative, not exact.

## 1. Schedules implemented (`biternionnet/schedules.py`)

| `lr_schedule` | Behaviour |
|---|---|
| `constant` | paper setting |
| `wsd` | linear warmup (`warmup_fraction` 0.05) -> constant -> linear decay over the last `decay_fraction` (0.3) to `lr * final_lr_ratio` (0.1), per step |
| `plateau_cosine` | constant until the train-loss decrease over `plateau_window` epochs falls below `plateau_threshold` (or `decay_start_epoch`, or the epoch budget), then cosine over `cosine_epochs` to `lr * final_lr_ratio`, then stop |

`--resume-from` restores model / optimizer / history / schedule state, so a constant run can be continued into a
decay phase after inspecting its logs (`scripts/schedule_report.py`). Note that AdaDelta's accumulators give an
implicit warmup, so the warmup phase of WSD is essentially redundant here; the decay phase is what matters.

## 2. Constant lr, 200 epochs (`runs/tc-btvm-manual`)

| epochs | train loss (10-ep mean) | test MAAD 10-ep mean / min |
|---|---|---|
| 50 | 0.104 | 33.6 / 24.6 |
| 100 | 0.073 | 26.7 / 24.4 |
| 150 | 0.057 | 28.6 / 21.4 |
| 200 | 0.042 | 23.9 / 19.9 |

Train loss keeps falling; the train-loss plateau detector (window 10, 2 %) would fire at epoch 31 (noise), a
20-epoch window at 5 % at epoch 59-61. The test metric, not the train loss, is the plateau signal on this data.

## 3. Constant lr, 1000 epochs (`runs/tc-btvm-c1000`)

| epoch | train loss (50-ep mean, ~= angle with dropout on) | test MAAD 50-ep mean +- std | 50-ep min |
|---|---|---|---|
| 200 | 0.050 (18.4 deg) | 24.11 +- 3.41 | 19.37 |
| 400 | 0.025 (12.8 deg) | 22.13 +- 2.54 | 18.65 |
| 600 | 0.016 (10.2 deg) | 21.79 +- 3.03 | 18.71 |
| 800 | 0.012 (8.9 deg) | 21.35 +- 2.75 | 18.27 |
| 1000 | 0.010 (8.2 deg) | **20.94 +- 1.93** | 18.61 |

Slope of the test MAAD: -1.9 deg/100 ep (100-200), -1.6 (200-400), -0.6 (400-600), -0.45 (600-800), -0.3
(800-1000). No overfitting reversal up to 1000 epochs; diminishing returns from ~500. Best 50-epoch window:
20.68 deg at epoch ~926, i.e. the paper's 20.8 deg with a single model.

## 4. Constant 1000 -> cosine 100 (`runs/tc-btvm-long`, resumed from `tc-btvm-c1000`)

Presets added: `towncentre-biternion-vonmises-long`, `towncentre-biternion-long` (epochs 1100,
`plateau_cosine`, `decay_start_epoch` 1001, `cosine_epochs` 100, no auto trigger).

| window | test MAAD mean +- std | min |
|---|---|---|
| constant, epochs 951-1000 | 20.94 +- 1.93 | 18.61 |
| **cosine, epochs 1051-1100** | **19.13 +- 0.42** | 18.28 |
| cosine, epochs 1091-1100 | 18.78 +- 0.26 | - |
| final epoch 1100 | **18.50** | - |

During the decay the metric is flat while lr > 0.5 (20.9 -> 19.8) and drops once lr < 0.4 (19.5 -> 18.8).
The main effect of the decay is removing the constant-lr noise (std 1.9 -> 0.4); the mean improves by 1.8 deg.

## 5. Decisions

- 200 epochs is ~3 deg short of the asymptote; the long presets use 1000 constant + 100 cosine epochs (44 000 steps).
- Baseline for later ablations: **19.13 +- 0.42 deg (last 50 epochs), final 18.50 deg** (`runs/tc-btvm-long`).
  Note: this run predates fix 001-G (Lanczos clipping); 004 re-runs the baseline.
- Train loss ~0.007 (~7 deg with dropout) vs test 18.5 deg -> the remaining gap is a data / regularisation
  problem, addressed in 004.
