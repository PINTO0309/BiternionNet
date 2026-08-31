# 004: Augmentation ablation (neighbour frames / photometric / scale jitter)

Created 2026-08-30 (user request: implement augmentations 1-3 from the proposal, then run the ablation manually).
Implementation is done; the runs are executed by hand and the results section is filled in afterwards.

## 0. Motivation

003 §5: after 1000 + 100 epochs the training loss corresponds to ~7 deg while the test MAAD is 18.5 deg, with only
random crop + flip as augmentation. TownCentre (002) has 480 k frames of which 1 % are labelled, small blurry
crops, and label noise of the order of 10 deg. The candidates were screened against the augmentation code of
`High-Angle_Robust_Fast_FaceAlignment` (`src/hrffa/data/dataset.py`, `dataset/augment/geometric.py`): the
label-invariant photometric operations transfer, the 3D geometric ones (camera homography, depth reprojection,
generated images) do not apply to pan-only 28 px crops.

## 1. What was implemented

| # | Augmentation | Where | Switch |
|---|---|---|---|
| A | **Neighbouring frames**: the +-k unlabelled frames around every labelled *training* frame get the same angle (median head turn 14 deg per 100 frames -> < 1 deg over +-3 frames). Records carry `source: neighbor`, `anchor_frame`, `frame_offset`. Test/val and the person split are identical to the k=0 manifest | `converters.convert_towncentre_raw` | `biternion-convert --neighbor-frames 3` |
| B | **Photometric** preset `cctv`: brightness/contrast/gamma (p 0.8; x0.6-1.4, +-25/255, gamma 0.7-1.4), motion blur (p 0.3, 2-3 px), gaussian noise (p 0.3, sigma 3-12/255), random erasing (p 0.3, 10-20 % of the side, 1 box), gaussian blur k3 (p 0.15), JPEG q50-90 (p 0.15), no grayscale. Applied on the 50x50 resized image before the crop. `cctv-light` halves the probabilities | `biternionnet/augment.py`, `data.prepare_image` | `--photometric cctv` |
| C | **Scale jitter**: pre-crop resize multiplied by U(0.9, 1.1) (clamped so the image never gets smaller than the crop: 46-55 px) | `data.resize_for_crop` | `--scale-jitter 0.9 1.1` |

`data/towncentre/manifest_nb3.jsonl`: train 26 897 heads (3 904 anchors + 22 993 neighbours, x6.89), test 443
identical to `manifest.jsonl`; 269 steps per epoch at batch 100.

![augmented samples: left = unaugmented centre crop, right = 7 random draws of B + C](assets/004/augmentation_samples.png)

## 2. Runs (equal optimizer budget, 44 000 steps; seed 0; `--num-workers 4`)

Neighbour-frame runs use 164 epochs (= 44 116 steps), cosine over the last 15 (= 4 035 steps), matching the long
presets' 1000 + 100 x 40 steps.

| Run | Manifest | Augmentation | Command |
|---|---|---|---|
| R0 | manifest.jsonl | none (baseline, re-run after 001-G) | `biternion-train --experiment towncentre-biternion-vonmises-long --manifest data/towncentre/manifest.jsonl --seed 0 --num-workers 4 --output runs/aug-r0-baseline` |
| R1 | manifest_nb3.jsonl | A | `biternion-train --experiment towncentre-biternion-vonmises --manifest data/towncentre/manifest_nb3.jsonl --seed 0 --num-workers 4 --epochs 164 --lr-schedule plateau_cosine --disable-plateau-trigger --decay-start-epoch 150 --cosine-epochs 15 --output runs/aug-r1-nb3` |
| R2 | manifest.jsonl | B | `... --experiment towncentre-biternion-vonmises-long --photometric cctv --output runs/aug-r2-photo` |
| R3 | manifest.jsonl | C | `... --experiment towncentre-biternion-vonmises-long --scale-jitter 0.9 1.1 --output runs/aug-r3-scale` |
| R4 | manifest_nb3.jsonl | A + B + C | R1 command + `--photometric cctv --scale-jitter 0.9 1.1 --output runs/aug-r4-all` |

Comparison: `uv run --locked python scripts/compare_runs.py runs/aug-r0-baseline runs/aug-r1-nb3 runs/aug-r2-photo runs/aug-r3-scale runs/aug-r4-all --steps 2000 --markdown`
(mean +- std over the last 2 000 steps = 50 epochs for R0/R2/R3, 7 epochs for R1/R4; min; final epoch).

**Decision rule**: adopt an augmentation if the mean over the last 2 000 steps improves by more than 2x the
baseline std (~0.8 deg); confirm the winner with seeds 1 and 2. Reference from 003: 19.13 +- 0.42, final 18.50.

## 3. Results (2026-08-30, seed 0)

`scripts/compare_runs.py ... --steps 2000 --markdown` (window = last 2 000 steps: 50 epochs for R0/R2/R3, 7 for R1/R4):

| run | epochs | steps | win_ep | maad_deg mean | std | min | final |
|---|---|---|---|---|---|---|---|
| aug-r0-baseline | 1100 | 44000 | 50 | 18.93 | 0.39 | 18.46 | 18.60 |
| aug-r1-nb3 | 164 | 44116 | 7 | 18.59 | 0.18 | 18.16 | 18.66 |
| aug-r2-photo | 1100 | 44000 | 50 | 18.51 | 0.58 | 17.73 | 18.40 |
| aug-r3-scale | 1100 | 44000 | 50 | 18.24 | 0.44 | 17.52 | 17.99 |
| aug-r4-all | 164 | 44116 | 7 | 19.51 | 0.28 | 19.11 | 19.38 |

Diagnostics from `history.jsonl` (window mean of test MAAD at 25 / 50 / 75 % of the step budget, just before
the cosine phase, and the gain during the cosine phase; train loss converted to the equivalent angle with dropout on):

| run | train loss at end (~angle) | MAAD @25 % | @50 % | @75 % | before decay | final | gain in decay |
|---|---|---|---|---|---|---|---|
| R0 baseline | 0.0080 (7.3 deg) | 24.47 | 21.90 | 20.52 | 20.95 | 18.60 | +2.02 |
| R1 nb3 | 0.0106 (8.4 deg) | 20.04 | 18.89 | 18.51 | 18.71 | 18.66 | +0.12 |
| R2 photo | 0.0146 (9.8 deg) | 23.45 | 21.75 | 20.30 | 20.57 | 18.40 | +2.06 |
| R3 scale | 0.0117 (8.8 deg) | 21.99 | 20.53 | 20.03 | 19.60 | 17.99 | +1.37 |
| R4 all | 0.0254 (13.0 deg) | 19.99 | 21.02 | 19.61 | 19.73 | 19.38 | +0.22 |

Reading:

- **R0** re-run after 001-G: 18.93 +- 0.39 (003 had 19.13 +- 0.42 before the clip fix) - a 0.2 deg shift of the
  same configuration, which is the size of the nuisance variance to keep in mind below.
- **R3 scale jitter** is the best single change: -0.69 deg on the window mean, -0.61 on the final epoch, lower
  minimum (17.52). Just below the 0.8 deg decision threshold; needs seeds 1 / 2.
- **R2 photometric**: -0.42 deg, train loss 1.8x the baseline (it regularises as intended), larger epoch noise.
- **R1 neighbour frames**: same end point as the baseline (-0.34) but reaches it far earlier - 20.0 deg at
  25 % of the budget vs 24.5 - and the constant phase plateaus at ~18.5-18.7 from 75 % on, so the cosine phase
  adds only 0.12. At equal steps the 7x data buys convergence speed, not accuracy; the epoch noise is halved
  (std 0.18).
- **R4 all** is *worse* than the baseline (+0.58): train loss 0.025 (3x the baseline), the metric is flat at
  ~19.6-21 from 25 % of the budget and the decay adds 0.22. With 7x correlated data plus photometric plus scale
  jitter the 0.63 M-parameter net is under-fitted at 44 k steps; the combination needs either more steps or a
  lighter photometric preset.

Decision: no augmentation passes the > 0.8 deg rule on its own. Scale jitter (R3) is the candidate to
confirm; the combination must be re-run with a larger budget before it is judged.

## 3.1 Follow-up runs (planned)

| Run | Purpose | Command |
|---|---|---|
| R0-s1, R0-s2 | baseline seed spread | R0 command with `--seed 1` / `--seed 2`, outputs `runs/aug-r0-baseline-s1`, `-s2` |
| R3-s1, R3-s2 | confirm scale jitter | R3 command with `--seed 1` / `--seed 2`, outputs `runs/aug-r3-scale-s1`, `-s2` |
| R5 photo+scale | combination without neighbour frames | `biternion-train --experiment towncentre-biternion-vonmises-long --manifest data/towncentre/manifest.jsonl --seed 0 --num-workers 4 --photometric cctv --scale-jitter 0.9 1.1 --output runs/aug-r5-photo-scale` |
| R6 nb3+scale | neighbour frames with the lightest extra augmentation | R1 command + `--scale-jitter 0.9 1.1 --output runs/aug-r6-nb3-scale` |
| R4-long | all three at ~2.1x budget (94 k steps: 300 constant + 50 cosine epochs x 269) | R4 command with `--epochs 350 --decay-start-epoch 301 --cosine-epochs 50 --num-workers 16 --output runs/aug-r4-all-long` (started 2026-08-30) |
| R4-light | all three, `cctv-light` | R4 command with `--photometric cctv-light --output runs/aug-r4-all-light` |

## 3.1.1 Additional scenario (2026-08-31): direct 46x46 resize, no crop

`--resize-size 46 46` overrides the preset's 50x50 pre-crop resize so every image is resized straight to
the network input and the random 46-from-50 crop becomes the identity (no translation augmentation; the
full head box stays in frame instead of losing a 4 px border). `--scale-jitter` on top of it re-introduces
crops from 46..51 px versions, so the pure no-crop arm drops it. Commands (balanced manifest, 350/301/50):

```text
D1 pure resize:      ... --resize-size 46 46                       --output runs/abl-resize46
D2 resize + jitter:  ... --resize-size 46 46 --scale-jitter 0.9 1.1 --output runs/abl-resize46-jitter
D3 no-crop 64x64:    ... --resize-size 64 64 --input-size 64 64    --output runs/abl-resize64
(reference = the same command with the preset 50x50 + crop, i.e. synth-biternion-vonmises-relu/swish)
```

**D1 result (2026-08-31, `runs/abl-resize46-swish`, balanced manifest, swish, 350/301/50 = 165k steps):**

| | test last7 | test macro | test_nb MAAD | test_nb macro | test_nb 45 | test_nb 90 |
|---|---|---|---|---|---|---|
| reference: balanced swish (50->46 crop + jitter) | 18.32 +- 0.12 | 18.80 | 20.74 | 22.48 | 18.6 | 24.1 |
| D1 pure 46x46 resize (no crop, no jitter) | 20.26 +- 0.05 | 22.46 | 21.89 | 24.72 | **27.8** | 27.6 |

Removing the crop/scale augmentation costs ~2 deg overall and ~2.2 deg macro; the 45-deg sector collapses
(+9 deg). The 46-from-50 random crop (+ scale jitter) is load-bearing, not a legacy detail - **D1 rejected**.
D2 (46x46 resize + scale jitter, which re-introduces crops from 46-51 px) would separate the translation
component from the scale component if the question becomes relevant again.

D3 (`--input-size`, added the same day) resizes straight to a 64x64 network input: the backbone then
yields a 64@9x9 map (`Linear(5184, 512)`, ~2.9 M params vs 1.6 M at 46). Note the TownCentre crops have a
median height of 29 px, so 64 px is >2x upsampling - the scenario probes whether the extra resolution of
the resize (not of the data) helps. ONNX export follows the checkpoint automatically
(`<prefix>_1x3x64x64.onnx`), and the deployment contract becomes "resize directly to 64x64".

## 3.2 R4-long result (2026-08-30)

| run | epochs | steps | win_ep | maad_deg mean | std | min | final |
|---|---|---|---|---|---|---|---|
| aug-r4-all-long | 350 | 94150 | 7 | 18.80 | 0.33 | 18.20 | 18.92 |

Trajectory (7-epoch window): 19.80 (ep 50) / 19.68 (100) / 19.99 (150) / 20.20 (200) / 19.35 (250) / 19.38 (300)
-> cosine 301-350 -> 18.80; train loss 0.061 -> 0.022 (ep 300) -> 0.017 (end).

- The constant phase is **flat at ~19.4-20.2 deg from epoch 50 to 300** while the train loss keeps falling: the
  extra 40 k steps did not move the test metric. The "under-fitted at 44 k steps" reading of §3 was only half
  right - the combination is limited by the augmentation strength, not by the budget.
- The cosine phase gains 0.58 deg (R4: 0.22, R1: 0.12, R0/R2/R3: 1.4-2.1).
- End point 18.80 +- 0.33 at 2.1x the budget: equal to the baseline (18.93) and clearly behind R3 (18.24)
  and R1 (18.59) at 1x. **A + B + C together is not adopted.**

Working hypothesis: with 7x correlated neighbour frames the photometric preset `cctv` is too strong (train loss
stays 2-3x the baseline's).

**Decision (user, 2026-08-30): A + B + C in the R4-long configuration becomes the default anyway** - the dataset
is too small and too noisy for the single-augmentation deltas (0.3-0.7 deg, comparable to the seed spread) to
be a sound basis for dropping augmentations, and the next step is to reinforce the dataset itself (006) before
re-tuning augmentation strength. Preset: `towncentre-biternion-vonmises-aug` (epochs 350, decay 301, cosine 50,
`photometric=cctv`, `scale_jitter=(0.9, 1.1)`), to be used with `manifest_nb3.jsonl`. The §3.1 follow-ups are
kept as optional checks.

## 4. Next candidates (not implemented)

- In-plane rotation +-10-15 deg with the label shifted by the same angle (pan is an image-plane angle; sign to be
  verified on a 0 deg sample). The only remaining label-changing geometric augmentation besides the flip.
- Angle interpolation between labelled frames (only where the change is small), or self-training on the 480 k
  unlabelled frames with the best model as teacher.
- Per-bin (8 x 45 deg) MAAD in the evaluation to see whether the rare profiles (002 §3) improve.

## 3.1.2 Neighbour-label jitter (2026-08-31)

Neighbour records copy their anchor's integer angle, so tens of samples share one exact degree - visible
as 1-deg spikes in the training-label donuts even for the flat manifests. `--neighbor-label-jitter J`
adds uniform +-J deg online (fresh draw each access, anchors and synthetic untouched, stored in the
checkpoint config). Suggested J = 1.5 (the measured head-turn rate is ~0.14 deg/frame, so +-10 frames
drift ~1.4 deg). Off by default. First ablation (capped600 swish pair, 017 §12): jitter 1.5 improves test 18.63 -> 18.32
(+-0.06 epoch noise) and test_neighbor macro 22.77 -> 22.57, overall test_neighbor -0.07 - small but
consistently non-negative at zero cost; recommended for neighbour-heavy manifests, single seed so far.
