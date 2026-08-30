# 006: Synthetic generation plan for the sparse pan bands

Created 2026-08-30 (user request: list, per angle band, how many images have to be generated to reinforce
the yaw/pan coverage of TownCentre; a synthetic generation run will be attempted). Counts come from
`data/towncentre/manifest.jsonl` (train 3 904 / test 443); the generation-yield assumption comes from 005.

## 0. How the numbers are defined

- Bins are 10 deg wide and centred on multiples of 10 deg (pan 0 = facing the camera, 90 = profile facing image-right,
  180 = back of head).
- Training uses a horizontal flip with p = 0.5, which maps pan theta -> 360 - theta. What the network effectively
  sees per bin is therefore the **flip-effective count** `(raw(theta) + raw(360 - theta)) / 2`; the table lists
  mirror pairs once (0..180) and the deficit applies to both bins of a pair.
- Adding `G` usable images at pan theta raises the effective count of *both* theta and 360 - theta by `G / 2`.
  Hence **usable needed = 2 x deficit** for a mirror pair (generate on one side only; the flip does the other),
  and `= deficit` for the self-mirrored bins 0 and 180.
- **Planned = usable / 0.5.** The yield of 0.5 is the fraction of `yaw_extreme` images in 005 that survived the
  pose filter (|est - intent| <= 25 deg, |est pitch| <= 35 deg): 268 / 500. A CCTV-specific generation config may
  do better or worse; re-measure on the pilot batch and rescale the plan.
- Two targets: **floor 120** (only fill bins below 120, cheap) and **uniform 200** (every bin at least 200
  effective, i.e. flat coverage at ~80 % of the current peak of 254 at 180 deg).
- These are counts of *anchor-quality* heads. When training on the neighbour-frame manifest (x6.9) the synthetic
  records must be weighted accordingly (sampling weight, or replicate ~7x), otherwise they are diluted.

## 1. Per-band table

| pan band (deg) | mirror band | train raw | mirror raw | flip-effective | test (both) | deficit to 120 | usable needed | planned (÷0.5) | deficit to 200 | usable needed | planned (÷0.5) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (355..5) | — | 137 | 137 | 137 | 22 | 0 | 0 | 0 | 63 | 63 | 126 |
| 10 (5..15) | 345..355 | 80 | 168 | 124 | 20 | 0 | 0 | 0 | 76 | 152 | 304 |
| 20 (15..25) | 335..345 | 38 | 142 | 90 | 24 | 30 | 60 | 120 | 110 | 220 | 440 |
| 30 (25..35) | 325..335 | 39 | 156 | 98 | 25 | 22 | 45 | 90 | 102 | 205 | 410 |
| 40 (35..45) | 315..325 | 48 | 238 | 143 | 20 | 0 | 0 | 0 | 57 | 114 | 228 |
| 50 (45..55) | 305..315 | 19 | 208 | 114 | 22 | 6 | 13 | 26 | 86 | 173 | 346 |
| 60 (55..65) | 295..305 | 38 | 173 | 106 | 20 | 14 | 29 | 58 | 94 | 189 | 378 |
| 70 (65..75) | 285..295 | 28 | 121 | 74 | 16 | 46 | 91 | 182 | 126 | 251 | 502 |
| 80 (75..85) | 275..285 | 38 | 86 | 62 | 16 | 58 | 116 | 232 | 138 | 276 | 552 |
| 90 (85..95) | 265..275 | 71 | 92 | 82 | 22 | 38 | 77 | 154 | 118 | 237 | 474 |
| 100 (95..105) | 255..265 | 78 | 41 | 60 | 16 | 60 | 121 | 242 | 140 | 281 | 562 |
| 110 (105..115) | 245..255 | 83 | 22 | 52 | 11 | 68 | 135 | 270 | 148 | 295 | 590 |
| 120 (115..125) | 235..245 | 105 | 18 | 62 | 13 | 58 | 117 | 234 | 138 | 277 | 554 |
| 130 (125..135) | 225..235 | 173 | 30 | 102 | 29 | 18 | 37 | 74 | 98 | 197 | 394 |
| 140 (135..145) | 215..225 | 235 | 27 | 131 | 21 | 0 | 0 | 0 | 69 | 138 | 276 |
| 150 (145..155) | 205..215 | 251 | 37 | 144 | 38 | 0 | 0 | 0 | 56 | 112 | 224 |
| 160 (155..165) | 195..205 | 233 | 42 | 138 | 37 | 0 | 0 | 0 | 62 | 125 | 250 |
| 170 (165..175) | 185..195 | 287 | 68 | 178 | 37 | 0 | 0 | 0 | 22 | 45 | 90 |
| 180 (175..185) | — | 254 | 254 | 254 | 34 | 0 | 0 | 0 | 0 | 0 | 0 |
| **total** | | 3904 | | | 443 | | **841** | **1682** | | **3350** | **6700** |

Aggregated per 45 deg band (flip-effective, both sides): 0 deg 565, 45 deg 460, **90 deg 330**, 135 deg 438,
180 deg 884. The profiles (70-120 deg) are the real hole: effective 52-82 per 10 deg bin, i.e. 1/3-1/5 of the
back-of-head peak.

## 2. Generation request per band (target uniform 200, one side)

| requested pan band (one side; flip covers the mirror) | planned images (target 200) |
|---|---|
| 15-45 deg | 1078 |
| 45-75 deg | 1226 |
| 75-105 deg | 1588 |
| 105-135 deg | 1538 |
| 135-165 deg | 750 |
| 5-15 deg | 304 |
| 0 deg (355-5) | 126 |
| 165-180 deg | 90 |

Total planned ~ 6700 images for uniform 200 (~ 1682 for floor 120). Request only one side (e.g. the
subject's left / image-right) or use the pipeline's `signed_abs` mode with half the count per sign; both give the
same effective coverage after the flip, the two-sided variant adds appearance diversity.

## 3. Generation spec (for a new HRFFA `gpt_head_gen` config)

| parameter | value | reason |
|---|---|---|
| camera elevation (`cam`) | +30..+60 deg | TownCentre is an oblique top-down CCTV view (crown and shoulders visible, 002 §2) |
| pitch | -10..+10 deg | pedestrians look level; 005 showed that pitch composites are useless here |
| yaw / pan | bands of §2; **over-request the outer bands** (ask 75-105 to obtain ~60-90) | the generator under-rotates: intent >= 60 came out with median 56 (005 §2) |
| back views | **allowed** (`reject_back: false`) | 100-180 deg is needed; the current sets have none |
| roll | 0 | |
| framing | head + shoulders, head height 30-45 % of the image; scene: street / plaza, outdoor daylight, other pedestrians allowed | TownCentre content; the crop is downscaled to ~28 px anyway |
| appearance diversity | age / gender / skin tone / hair / hats / bags as in the existing configs | 32 % of TownCentre persons have a single labelled head, so identity diversity is the main value of synthesis |

## 4. Labelling and QA of the generated images

1. Head box: DEIMv2-Wholebody49 class 7 (as in `auto_qa.jsonl`), plus the 8 direction classes (ids 8-15) as a
   coarse pan check (Front / Right_Front / Right_Side / Right_Back / Back / Left_Back / Left_Side / Left_Front).
2. Pan label: SixDRepNet360 on the *full-resolution* crop (005 §1), sign-corrected (`pan = -yaw`), for the
   front half. For back views (|pan| > 90) SixDRepNet is unverified; use the DEIMv2 direction class as the
   acceptance test and the intent as the label unless a pilot shows SixDRepNet to be reliable there.
3. Accept if the measured pan is within 25 deg of the requested band and |pitch| <= 35 deg; bin by the
   *measured* pan, not by the request.
4. Downscale the head crop (15 % margin) to the TownCentre size distribution (median 29 px, 17-59) before it
   enters the pipeline, so the resize-to-50 step sees the same blur; `source: "synthetic"` in the manifest.

## 5. Experiment

Baseline: the default preset `towncentre-biternion-vonmises-aug` on `manifest_nb3.jsonl` (004 §3.2: 18.80 +- 0.33,
final 18.92). Add the synthetic records with a sampling weight that makes them ~10-20 % of each batch, same step
budget. Judge on per-bin MAAD (8 x 45 deg; to be added to `evaluate_model`) - the 90 / 270 deg bins are the
target - and on the overall MAAD only as a guard against regression.
