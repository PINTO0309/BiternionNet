# 017: Composite manifests - current / balanced training with a shared enlarged test side

Created 2026-08-31 (user requests, in order: flatten the angle distribution by adjusting the neighbour
amount per bin while keeping all anchors and all synthetic records; make current/balanced selectable; also
allocate some data to the test side; implement). Builder: `biternionnet/balance.py`,
CLI `scripts/build_composite_manifests.py`; tests in `tests/test_balance.py`.

## 1. What is generated

`build_composite_manifests --combined data/towncentre/manifest_nb3_synthetic_all_elevations.jsonl` writes
two manifests that differ **only in train**; selection at training time is just `--manifest`:

| | train | steps/epoch | 94k-step budget |
|---|---|---|---|
| `manifest_current.jsonl` | anchors 3,904 + k=3 neighbours 22,993 + synthetic 7,525 = 34,422 | 345 | epochs 273 (const 234 + cosine 39) |
| `manifest_balanced.jsonl` | anchors 3,904 + quota neighbours 35,839 + synthetic 7,525 = 47,268 | 473 | epochs 199 (const 171 + cosine 28) |

Balanced: neighbours are re-selected per flip-effective 10-deg mirror-pair bin up to the auto target
**T = 1,313** (nearest |frame_offset| first, cap +-10, deterministic seed) - achieved effective min = max =
1,313.0 (max/min 1.00; hash-bound holdout state, see §7). Raw per-bin counts are *not* flat (the two sides of a mirror pair fill where frames
exist); what training sees under p=0.5 flips is the flip-effective count, which is flat.
`plot_angle_distribution.py --flip-effective` renders that view.

Shared test side (byte-identical in both files):

- `test` 443 - the original anchors, unchanged (comparability with all earlier runs; per-epoch eval and
  checkpoint selection keep using only this split);
- `test_neighbor` 8,418 - +-10 frames of the test anchors, same angle (correlated samples: report
  anchor-clustered errors when quoting per-bin numbers);
- `test_synthetic` 850 - a ~10 % hash-bound holdout (statistically stratified; see §7) of the synthetic set, removed from train
  in BOTH manifests (so the current/balanced pair differs only in neighbours). Domain-shifted diagnostic
  split; 87 % of synthetic labels are intent-based (see 007 review) - read `label_source` before trusting it.

## 2. Evaluation

`biternion-eval --split test_neighbor` etc.; per-45-deg-bin MAAD (`bin_XXX_maad_deg`, `bin_XXX_count`,
`bin_macro_maad_deg`) is included in every angle evaluation by `biternionnet/evaluation.py` (008), and
`--predictions-output` writes per-item rows for paired bootstraps.

## 3. Caveat carried over from the 007 review

On the original 443-head test the 90/270-deg bins have 33/42 heads (bootstrap SE ~5 deg), so per-bin
conclusions should be drawn on `test_neighbor` (n 683/961 in those sectors, correlated) or on pooled
sectors, with paired comparisons.

## 4. Runs (planned)

Same 94k-step budget, seed 0..2:

```text
A: --manifest data/towncentre/manifest_current.jsonl  --epochs 273 --decay-start-epoch 235 --cosine-epochs 39
B: --manifest data/towncentre/manifest_balanced.jsonl --epochs 199 --decay-start-epoch 172 --cosine-epochs 28
(both: towncentre-biternion-vonmises + plateau_cosine --disable-plateau-trigger --photometric cctv --scale-jitter 0.9 1.1)
```

Judge on: overall `maad_deg` on `test` (guard), `bin_macro_maad_deg` and the profile bins on `test` and
`test_neighbor`. Expectation: balanced trades a little overall MAAD (the test set stays bimodal) for better
profile bins.

## 5. Figures

![balanced manifest, flip-effective view](assets/017/angle_distribution_balanced.jpg)
![current manifest](assets/017/angle_distribution_current.jpg)
![planning figure](assets/017/angle_distribution_composite_plan.jpg)

## 6. Update 2026-08-31 (second synthetic batch)

`production-mask20-v002-edit01-crop` added 1,675 synthetic heads (front / profile bands 0/45/90/270/315,
label_source sixdrepnet360 358 / intent 1,317; masked-face variants). Regenerated with the same command:
synthetic 8,375 -> holdout 855, synthetic_train 7,520; current train 34,417 (345 steps/epoch, budget
273/234/39); balanced train unchanged at 47,196 - the extra synthetic displaces neighbour quota
(35,772, was 37,278) and the auto target stays **T = 1,311** (still limited by the 110-deg pair's
neighbour availability). Test side: `test` and `test_neighbor` unchanged; `test_synthetic` grew 686 -> 855,
so results on it are not comparable with evaluations made before this update.

## 7. Hash-bound holdout (2026-08-31)

The `test_synthetic` holdout selection changed from per-bin random sampling to a **crc32(custom_id) hash
gate** (HRFFA-style, salted by `--seed`): a synthetic record's train/holdout membership is now stable when
later batches are added (the previous scheme reshuffled it - the uniform200 holdout changed 686 -> 679 when
the mask20 batch arrived, which would leak train images of older models into a re-generated test_synthetic).
Stratification per bin is now statistical rather than exact. This regeneration reshuffles membership one last
time; evaluations on test_synthetic made before it are void.
