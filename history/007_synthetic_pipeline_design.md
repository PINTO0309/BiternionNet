# 007: TownCentre synthetic-data pipeline design

Created 2026-08-31; revised 2026-08-31 after the Fable 5 design review and the decision to retain
eye-level/low-angle outputs (user request: inspect and adapt the synthetic head generation pipeline from
`High-Angle_Robust_Fast_FaceAlignment`, then plan a BiternionNet-specific reinforcement dataset under
`data/synthetic`; planning only in this entry, with no API submission or image generation).

## 0. Decision

- Port the proven Batch API state machine and QA workflow into this repository, but do **not** copy the HRFFA
  pipeline unchanged. Its Validation/Pilot/Production sizes are hard-coded to 10/500/5,000 and its prompt and QA
  explicitly reject back-of-head images, while this dataset needs configurable per-pan quotas and 100-180 deg
  back views.
- Make the high-angle-qualified count in each measured 10 deg pan band the stopping condition. The 1,682 / 6,700
  request counts in 006 are initial estimates based on a global 0.5 yield, not fixed production sizes.
- Do not discard an otherwise valid image merely because its apparent camera elevation is eye-level or lower
  than requested. Preserve and materialize it with an elevation class. Only images classified as matching the
  requested high angle count toward the floor-120/uniform-200 high-angle deficits; retained eye-level images are
  reported as a separate usable pool.
- Generate a 19-image validation set and a 380-image all-band pilot first. Use the pilot to measure the
  request-to-measured-pan confusion matrix, per-band yield, and the reliability of SixDRepNet360 on generated
  back views before any larger submission.
- Before the paid pilot, manually label a new 200-300-image `test_profiles` split from previously unlabelled
  TownCentre tracks. Its people must be disjoint from both the existing train and test people. Freeze the
  selection, annotation, and paired-bootstrap protocol before training any synthetic-data arm.
- First fill only to the effective floor of 120 (841 high-angle-qualified heads, initially about 1,682 requests).
  Train and evaluate that dataset before authorizing the uniform-200 expansion (3,350 high-angle-qualified heads,
  about 6,700 total requests). The Pilot's high-angle-qualified images count toward these totals; its valid
  eye-level images accumulate separately.
- Keep the TownCentre test split unchanged. Synthetic images are training-only and are selected by a fixed
  10 % batch sampling quota with the original 26,897-sample epoch length. A 20 % arm is deferred until the
  accepted unique-image count can satisfy a four-draw-per-image-per-epoch cap.

No `data/synthetic` directory is created in this planning step. It will be created by the implementation's
`plan` command, before the first API request is submitted.

## 1. Evidence and constraints

### 1.1 Source pipeline inspected

The source was inspected at HRFFA commit `1155c7f7b3f07c649c64f45516750f86ca0e7015` (2026-08-30):

| source | useful part | required change |
|---|---|---|
| `src/hrffa/dataset/augment/gpt_head_gen.py` | deterministic request plans, JSONL validation, Batch submit/status/collect, custom-ID reconciliation, retries, checksums, output pruning/archiving | remove 10/500/5,000 and 10 %-subset assumptions; make model/stages/quotas configurable; add measured-band top-ups |
| `src/hrffa/dataset/qa/gpt_head_review.py` | image integrity, duplicate, DEIM head/body/direction checks, contact sheets, hash-bound human approval | allow back views; replace `roll_no_back`; add pose-source and pan-band gates |
| `src/hrffa/dataset/pseudolabel/deimv2.py` | DEIMv2-Wholebody49 preprocessing and classes 0, 7, 8-20 | port as a small optional QA backend |
| `src/hrffa/dataset/qa/sixdrepnet.py` | full-resolution head-crop preprocessing and Y/P/R inference | add explicit `pan = -yaw mod 360`, circular-error utilities, caching, and back-view reliability report |
| `configs/head_image_generation.yaml` | diversity schedules, high-camera prompt, storage/API profile | replace pitch-oriented bins with 19 absolute-pan bins and TownCentre scenes |
| `tests/test_gpt_head_pipeline.py` | mocked API/state/retry/QA coverage | port the relevant tests and add quota, back-view, crop, manifest, and sampling tests |

The HRFFA repository is MIT licensed. Any substantially copied implementation must keep its copyright and MIT
notice in adapted file headers, including the source URL and commit above. The DEIMv2 model is
Apache-2.0; the SixDRepNet360 model must retain its distributor's license separately.

### 1.2 API assumptions rechecked

As of 2026-08-31, official OpenAI documentation confirms that:

- GPT-Image-2 supports `POST /v1/images/generations`, `quality=low`, opaque output, PNG/JPEG/WebP, and the three
  existing standard sizes. JPEG compression is supported. See the
  [image endpoint](https://developers.openai.com/api/reference/resources/images/methods/generate) and
  [GPT-Image-2 model page](https://developers.openai.com/api/docs/models/gpt-image-2).
- Batch creation accepts `/v1/images/generations`; a JSONL input may contain at most 50,000 requests and be at
  most 200 MB. See [Create batch](https://developers.openai.com/api/reference/resources/batches/methods/create).

The existing request format is therefore still usable. The documentation lists
`gpt-image-2-2026-04-21`, but documentation is not proof that this account can submit that snapshot or that the
observed bill will equal a table projection. Validation must first perform an access preflight and record the
requested model, returned model identifier when exposed (or its absence), usage fields, and actual account-side
charge when available. Only then may a snapshot and request profile be frozen for Pilot and Production.

### 1.3 BiternionNet gaps that must be fixed before training

- `ManifestDataset` tolerates extra metadata and already resolves image paths relative to the manifest, so a
  synthetic record can use the existing `angle_deg` task directly.
- The train loader currently uses only `shuffle=True`; it ignores `source` and sample weights. Appending the
  floor-120 set naturally contributes only about 3 % of the 26,897 neighbour-frame training records.
- `evaluate_model` reports only aggregate MAAD. It does not provide the 8 x 45 deg per-pan metrics required by
  006.
- The current test split has only 33 records in the 90 deg bin and 42 in the 270 deg bin (17 in 225 deg).
  Fable's current-best-model bootstrap found standard errors of 4.8 / 5.1 deg for 90 / 270, 3.6 deg when the
  two bins are pooled, and about 2.1 deg after averaging three seeds. A fixed 0.5 deg promotion target is
  therefore below the resolution of this evaluation.
- The 005 `camera_high` records directly invalidate an eye-level pitch gate: for 750 records with camera
  elevation +30..+70 deg, SixD pitch has median -38.9 deg (q10 -53.3, q90 -26.4). For the near-level-head
  hypothesis `expected pitch = -camera elevation`, `sixd_pitch + camera_elevation` has median +9.8 deg and
  median absolute residual 11.7 deg. Elevation classification must use a camera-relative residual, not
  `abs(pitch) <= 35 deg`; an eye-level classification is retained rather than treated as an image rejection.
- The source QA model files are not present here. The inspected copies are 197 MB (DEIMv2, SHA-256
  `c8e9ef7e79214c6eda7b7efa4d78e3dd3924d78d7f3ee069bfae483a25bc8fe4`) and 90 MB (SixDRepNet360, SHA-256
  `422b69d17bf02e164bd1062f5e970f67f1688cadf7d700800fad5fa12a16fc82`). They belong under ignored
  `data/models/`, with a checked download/copy helper and license metadata; they must not be committed as blobs.

## 2. Proposed repository design

### 2.1 Code and configuration

```text
configs/synthetic_towncentre.yaml
src/biternionnet/synthetic/
  __init__.py
  generate.py       # adapted Batch plan/submit/status/collect/resume state machine
  qa.py             # image, DEIM, pose and human-review gates
  detector.py       # adapted DEIMv2 wrapper
  pose.py           # adapted SixDRepNet360 wrapper
  quotas.py         # target counts, measured bins, yield and top-up planning
  materialize.py    # crop/downscale/deduplicate/manifest/report
  sampling.py       # source quota plus per-image epoch repetition cap
tests/test_synthetic_generate.py
tests/test_synthetic_qa.py
tests/test_synthetic_materialize.py
tests/test_synthetic_sampling.py
```

Expose one CLI, for example `biternion-synthetic`, with commands that never blur read-only planning and paid
submission:

```text
plan -> submit -> status/watch -> collect -> qa -> review/approve -> materialize -> report -> top-up-plan
```

`plan`, `qa`, `materialize`, `report`, and `top-up-plan` are local. Only `submit` creates paid remote work, and it
must print the request count, stage, model snapshot, and estimated upper-bound cost before submission.

### 2.2 Generated-data layout

```text
data/synthetic/
  README.md
  batches/<stage>-<run-id>/
    batch_state.json
    generation_plan.jsonl
    batch_input_*.jsonl
    images/*.jpg             # full-resolution JPEG q92 source/QA images
    auto_qa.jsonl
    pose_qa.jsonl
    human_review.csv
    approval.json
  crops/*.jpg                # all pan/quality-approved crops, including retained eye-level views
  annotations.jsonl          # consolidated provenance and QA records
  manifest.jsonl             # all pan/quality-approved synthetic train records
  manifest_high_angle.jsonl  # derived primary-experiment view; high-angle matches only
  report.json
  angle_distribution.jpg
```

Create `data/towncentre/manifest_nb3_synthetic.jsonl` from the high-angle view for the primary experiment, and
optionally `manifest_nb3_synthetic_all_elevations.jsonl` for a separately named ablation that also samples the
retained eye-level pool. Synthetic paths should be `../synthetic/crops/...`; the 443 test rows in either combined
manifest must be byte-for-byte equivalent in labels and image paths to `manifest_nb3.jsonl`. Raw Batch outputs
may be pruned only after the JPEG, plan, QA, and hash manifests pass the ported archive audit.

Store the new manual profile evaluation records separately as
`data/towncentre/test_profiles.jsonl`, with `split="test_profiles"`, annotator/provenance fields, and a frozen
SHA-256. Never append them to a training manifest.

## 3. Generation specification

### 3.1 Pose bins and target counts

Use 19 absolute-pan bins centred at 0, 10, ..., 180 deg. For 10-170 deg use `signed_abs`, balanced between signs;
0 and 180 deg are self-mirrored. The table's accepted quota is the number that passes both pan/quality review and
the high-angle classification; it is the total across both signs. Eye-level/low-angle images remain in the corpus
but do not reduce these deficits. The initial request estimate is high-angle-qualified quota / 0.5 and will be
replaced by pilot-measured per-bin yield.

The two signs are generated for appearance and scene diversity, not because mirror coverage requires both:
the existing horizontal flip already supplies the opposite-pan training view. Report accepted counts by sign so
one side cannot silently dominate.

| absolute pan | floor-120 accepted | initial requests | uniform-200 accepted | initial requests |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 63 | 126 |
| 10 | 0 | 0 | 152 | 304 |
| 20 | 60 | 120 | 220 | 440 |
| 30 | 45 | 90 | 205 | 410 |
| 40 | 0 | 0 | 114 | 228 |
| 50 | 13 | 26 | 173 | 346 |
| 60 | 29 | 58 | 189 | 378 |
| 70 | 91 | 182 | 251 | 502 |
| 80 | 116 | 232 | 276 | 552 |
| 90 | 77 | 154 | 237 | 474 |
| 100 | 121 | 242 | 281 | 562 |
| 110 | 135 | 270 | 295 | 590 |
| 120 | 117 | 234 | 277 | 554 |
| 130 | 37 | 74 | 197 | 394 |
| 140 | 0 | 0 | 138 | 276 |
| 150 | 0 | 0 | 112 | 224 |
| 160 | 0 | 0 | 125 | 250 |
| 170 | 0 | 0 | 45 | 90 |
| 180 | 0 | 0 | 0 | 0 |
| **total** | **841** | **1,682** | **3,350** | **6,700** |

Do not submit all estimated requests at once. Submit at most 250-500 per shard, recompute accepted deficits after
each collected/QA'd shard, and target only the remaining measured bands. If the generator under-rotates a band,
derive the next requested range from the pilot confusion matrix rather than globally adding an arbitrary offset.

### 3.2 Image prompt/profile

- Camera elevation: +30..+60 deg; head pitch: -10..+10 deg; roll: 0 deg.
- Scene: outdoor street, plaza, station forecourt, crossing, or shopping street in daylight or plausible CCTV
  lighting. Exactly one primary pedestrian, complete head, neck, and shoulders; other people should be absent in
  Validation/Pilot because `require_single_head` must remain unambiguous.
- Framing at generation resolution: head height 30-45 % with enough crop margin. Pure back views are explicitly
  allowed for requested 135-180 deg bands; prompts and retry corrections must not contain HRFFA's “never show a
  back-of-head” instruction.
- Appearance schedule: balanced gender presentation, age bands, Fitzpatrick skin tones, hair styles including
  bald/short/long/tied hair, and limited hats/hoods/glasses. Keep masks low because face landmarks are used for
  front-half QA; hats remain useful because crown appearance matters at high elevation.
- Candidate API profile: GPT-Image-2 snapshot, `quality=low`, `background=opaque`, `n=1`, and JPEG q92 returned
  by the API. Validation must prove account access and record the returned model before the profile is called
  pinned. Retain the tested 50/30/20 portrait/square/landscape schedule unless the validation set shows framing
  drift.

## 4. Labelling and QA

### 4.1 Machine gates common to every band

1. Validate image bytes, expected dimensions, JPEG format, SHA-256, and absence of duplicates.
2. Run DEIMv2-Wholebody49 on the full-resolution image. Require exactly one accepted head, a body/shoulder
   context, head height and margin limits, and a direction class consistent with the requested broad sector.
3. Run pose and direction inference before cropping. Materialize only with the crop margin selected by the
   TownCentre contact-sheet calibration in section 5; preserve the full-resolution source for QA.
4. Record every estimate and decision; never overwrite intent with an estimate without recording
   `label_source`, `intent_pan_deg`, `estimated_pan_deg`, `direction`, model hashes, and rejection reasons.

### 4.2 Pan label decision

- Front and profile (`abs pan <= 90`): infer SixDRepNet360 at full resolution and use
  `angle_deg = (-sixd_yaw) mod 360`. Accept only if circular error from intent is <=25 deg and the DEIM
  direction is not contradictory. Bin by this measured pan.
- Do **not** gate on `abs(sixd_pitch) <= 35 deg`. SixD pitch is camera-relative. For a near-level head define
  the initial physical expectation `p0 = -camera_elevation` and residual
  `r_pitch = wrap180(sixd_pitch - p0) = wrap180(sixd_pitch + camera_elevation)`. On the front/profile Validation
  records, calculate the robust bias `b = median(r_pitch)` and the distribution of `abs(r_pitch - b)`; freeze a
  candidate `T_data = q90(abs(r_pitch - b)) + 5 deg`. Use
  `T_pitch = max(10 deg, T_data)` only if `T_data <= 25 deg`; otherwise the requested high-angle generation
  profile fails Validation. Classify an image as `camera_elevation_class="high_angle_match"` when it satisfies
  both `abs(r_pitch - b) <= T_pitch` and `sixd_pitch <= -0.5 * camera_elevation`. A pan/quality-valid image that
  fails only these high-angle conditions is **not rejected**: retain and materialize it as
  `camera_elevation_class="eye_level_or_low_angle"` with `counts_toward_high_angle_quota=false`. Use
  `unresolved` when the estimator itself is unusable. The 005 distribution (`b` about +9.8 deg, median absolute
  uncentred residual 11.7 deg) is a sanity check, not a threshold copied into the new run.
- Use the same pitch estimates as a batch-level high-angle check: Validation must span camera elevations near
  30, 45, and 60 deg; high-angle-qualified pitch must be negative with magnitude consistent with the frozen
  residual band, and higher requested elevation must shift the group estimate in the negative direction. The
  report stores `camera_elevation`, `sixd_pitch`, `p0`, `r_pitch`, `b`, threshold, elevation class, and quota flag
  for every image. Thresholds are not retuned on Pilot yield. Use rear-half pitch as a diagnostic until Pilot
  proves sector-wise reliability; rear elevation classification additionally requires human visual review
  rather than trusting an unvalidated pose component.
- Back half (`abs pan > 90`): treat the 380-image pilot as a validation experiment. SixD may be used only if,
  within each broad rear sector, at least 70 % of reviewed images are within 25 deg of intent and DEIM conflicts
  are <=10 %. Otherwise follow 006: use intended pan as `angle_deg`, require the expected DEIM rear direction,
  require a human `intent_match` decision, and mark `label_source="intent_rear"`. Do not silently coerce a DEIM
  45 deg sector into a precise 10 deg measurement.
- Always retain `label_confidence` so a later loss/sampling-weight experiment can down-weight rear intent labels.
  The first floor-120 experiment should either use confidence 1 for approved samples or exclude ambiguous rear
  records; it must not mix unreviewed labels.

### 4.3 Sign and direction convention

The following table is the provisional convention to be checked image-by-image in Validation. `pan` is the
TownCentre label, positive clockwise as seen in the 006 polar convention; the SixD column follows
`pan = -sixd_yaw mod 360`. In particular, DEIM `left_side` means the subject's left side is visible and the
subject faces image-right; it maps to pan 90 deg.

| pan sector centre | DEIM direction | visual meaning | expected SixD yaw |
|---:|---|---|---:|
| 0 | `front` | faces camera | 0 |
| 45 | `left_front` | front-left, faces image-right | -45 |
| 90 | `left_side` | subject left, image-right | -90 |
| 135 | `left_back` | back-left, toward image-right | -135 |
| 180 | `back` | back of head | +/-180 |
| 225 | `right_back` | back-right, toward image-left | +135 |
| 270 | `right_side` | subject right, image-left | +90 |
| 315 | `right_front` | front-right, faces image-left | +45 |

The 19-image Validation plan must deliberately cover all eight DEIM sectors, both signs around the profile and
rear boundaries, and at least one unambiguous back-of-head image. This matters because SixDRepNet360 returned no
`abs(yaw) > 90 deg` case in the 005 study. The signed mapping, examples, and any corrected table are stored in a
hash-bound `sign_calibration.json`; Pilot is blocked until the human reviewer approves it.

### 4.4 Human gates

- Validation: review all 19 images for camera elevation, requested pan, framing, roll, photorealism, body
  integrity, scene match, and an explicit back-of-head example. Compare crop candidates against a stratified
  TownCentre contact sheet and require comparable head truncation and surrounding margin. Approval requires a
  resolved sign table, no systematic left/right error, an approved pitch calibration, and at least 15/19 intent
  matches; otherwise revise prompts and create a new immutable Validation run. Images that pass pan/quality but
  not the high-angle classification remain in the eye-level pool even when the high-angle profile is rejected.
- Pilot: 20 requests per absolute-pan band (380 total), balanced by sign. Review every accepted rear-half image
  and at least five accepted images per front-half band. Publish per-band request count, machine pass count,
  reviewed intent match, measured-bin histogram, pitch-residual distribution, counts by elevation class, and
  rejection reasons.
- Stop and revise the **high-angle request profile** rather than scale it when any target band's high-angle-
  qualified yield is below 0.30, front/profile sign agreement is below 0.90, or rear label source cannot pass the
  rule above. This stop does not delete valid eye-level outputs; it prevents them from masquerading as fulfilled
  high-angle quotas. A larger request count is not a substitute for a usable label.

## 5. TownCentre materialization

The actual TownCentre training crops have the following train distribution (measured from all 3,904 anchors):

- height: min 18, q05 20, median 29, q95 49, max 60 px;
- width: min 17, q05 19, median 27, q95 46, max 58 px;
- width/height: median 0.933 (q05 0.895, q95 1.030).

DEIM's head box is not guaranteed to match the Benfold & Reid tracker box used to make the TownCentre crops, so
15 % is only a candidate margin. During Validation, render the same detections at 0, 5, 10, 15, and 20 % margin
beside a pan- and size-stratified contact sheet drawn from TownCentre **training** anchors. A reviewer selects the
setting with comparable crown/chin truncation and surrounding context, records the decision in the signed
approval, and freezes it before Pilot. If no global margin matches, define and validate a deterministic rule by
head size or aspect ratio; do not tune it against test performance.

For each pan/quality-approved synthetic head, including the retained eye-level class, deterministically sample
an `(H, W)` pair from the empirical anchor
distribution (stratified by pan if there are enough real records), crop with the frozen rule, and downscale with
`cv2.INTER_AREA`. Store that tiny crop as JPEG with fixed encoding settings. Do not save every sample as a square
28 px image: matching the real joint shape distribution avoids introducing an artificial aspect-ratio cue.

The materializer must produce a contact sheet at native tiny resolution and after the normal resize-to-50 path,
plus real-versus-synthetic margin/truncation, blur, brightness, and contrast comparisons. The existing `cctv`
training augmentation still runs after resize; it does not replace this native-resolution domain matching.

A synthetic manifest row should contain at least:

```json
{"split":"train","task":"angle_deg","angle_deg":90.0,"image":"crops/example.jpg","source":"synthetic","custom_id":"...","abs_pan_bin":90,"label_source":"sixdrepnet360","label_confidence":1.0,"camera_elevation_class":"eye_level_or_low_angle","counts_toward_high_angle_quota":false,"generation_run":"pilot-v001"}
```

## 6. Training and evaluation integration

### 6.1 Profile evaluation must exist before Pilot

The current 443-image test set is underpowered for this decision: its circular 45 deg bins contain only 33
images at 90 deg and 42 at 270 deg. Before paid Pilot generation, make a manually labelled `test_profiles` split
of 200-300 images (target 250), covering pan 60..120 and 240..300 with approximately equal counts by side and
angle subrange.

This is feasible without identity leakage: the local corpus has 501 person directories (35,594 frames) absent
from both the 1,573 train and 185 test people. Select only from those directories, target at least 150 people,
and cap each person at two retained frames. Freeze a hash-seeded person/frame candidate order before examining
model predictions; manual pose screening may fill the predeclared angle quotas, but current-model error must not
be a selection criterion. Two annotators independently label pan, and disagreements over 15 deg are adjudicated.
Store person ID, frame ID, both labels, final label, annotator IDs, and selection provenance.

Add per-bin MAAD to evaluation. Report the eight circular 45 deg bins, record counts, macro average, the two
target sides, and overall MAAD. Synthetic Validation/Pilot images are never evaluation data, and neither
`test_profiles` nor the existing test split is used for checkpoint selection. Because `test_profiles` is used
once for the predeclared production-promotion decision, later measurements on it are not an independent final
confirmation; a claim requiring confirmatory evidence needs another held-out person set.

### 6.2 Source quota and repetition cap

Implement a source-aware quota batch sampler, not an unconstrained weighted sampler. Keep 26,897 draws per epoch
and batch size 100; allocate 2,690 synthetic draws for the 10 % arm, distribute them as evenly as possible over
accepted images, rotate the extra draws by epoch, and enforce at most four draws of any synthetic image per
epoch. Log both requested and realized source fraction plus the maximum and quantiles of per-image repetitions.

| accepted set | natural share if appended | 10 % draws/image | 20 % draws/image | 20 % status |
|---|---:|---:|---:|---|
| floor 120: 841 | 3.0 % | 3.20 | 6.40 | blocked by cap |
| uniform 200: 3,350 | 11.1 % | 0.80 | 1.61 | eligible |

A 20 % arm needs 5,379 synthetic draws, hence at least `ceil(5379 / 4) = 1,345` accepted unique images. Do not
run it with the 841-image floor set and do not relax the cap to force the arm. Start at 10 %; test 20 % only
after a promoted top-up has supplied at least 1,345 diverse accepted images (the uniform-200 set satisfies this).

The table uses the primary high-angle view and is therefore a worst case for repetitions; retained eye-level
records add unique images when an all-elevations ablation is explicitly selected. Do not silently blend them
into R1: the primary `manifest_nb3_synthetic.jsonl` is high-angle-only, while the separately named
`manifest_nb3_synthetic_all_elevations.jsonl` makes their training effect measurable. Both use the same 10 %
total synthetic quota and four-draw cap.

### 6.3 Predeclared comparison and promotion rule

Run matched seeds with the same initialization seeds, checkpoint-selection rule, and step budget:

1. R0: current `towncentre-biternion-vonmises-aug`, `manifest_nb3.jsonl`.
2. R1: floor-120 high-angle view at 10 % of epoch draws.
3. R1E (separate ablation, not a replacement for R1): all retained elevation classes at the same 10 % quota.
4. R2 (deferred): 20 % only after the uniqueness condition in section 6.2 is satisfied.

R1 is the predeclared promotion comparison. R1E measures whether the retained eye-level pool helps, but it
cannot replace a failed R1 promotion result unless a new decision rule and multiplicity handling are frozen
before its predictions are examined.

For every evaluation item and matched seed compute circular absolute errors and
`d = error(R0) - error(R1)`, so positive values favour synthesis. Average matched-seed differences per item,
then perform a paired, person-cluster bootstrap (10,000 resamples of people, retaining all selected frames from
each sampled person). Report the mean difference and percentile 95 % CI for `test_profiles`, each target side,
the existing 90/270 bins, and overall test MAAD. This preserves the strong model-to-model correlation that an
unpaired MAAD bootstrap discards.

Authorize the uniform-200 top-up only when the predeclared primary `test_profiles` CI is wholly above zero and
the paired overall-test CI does not demonstrate degradation (is not wholly below zero). Remove the undetectable
fixed 0.5 deg improvement and 0.3 deg regression thresholds. The exact split hash, metric code hash, seed list,
bootstrap seed, resampling unit, number of resamples, and decision rule must be frozen before Pilot; do not
choose a bin aggregation or arm after seeing results.

## 7. Cost, storage, and operational limits

Official documentation on 2026-08-31 lists GPT-Image-2 low-quality output at about $0.005-$0.006 per
standard-size image under standard processing and lists Batch image-token rates at half the standard rates. See the
[image-generation cost table](https://developers.openai.com/api/docs/guides/image-generation#cost-and-latency)
and [pricing](https://developers.openai.com/api/docs/pricing). These are reference values, not a verified quote
for this account, request body, snapshot, or Batch result. Do not use the previous output-only dollar table or a
fixed 25 % retry reserve as a spending authorization; both compound unverified assumptions and omit input tokens,
failed requests, QA rejection, and top-ups.

Operationally, set `account_verified_snapshot = null`, `observed_cost_per_request = null`, and
`observed_cost_per_accepted = null` until Validation fills them. The documented snapshot/rates remain separate
`reference_*` fields so a reference value cannot accidentally satisfy a submission gate.

The 19-request Validation run is the pricing and usage calibration run. Its collector must aggregate, by image
size, every returned usage field (input text/image tokens, output tokens, completed/failed requests), the actual
model identifier, response count, bytes, and account-side charge when the organization exposes it. Report:

- actual or best-observable total cost and cost per completed response;
- pan/quality pass yield, high-angle-qualified yield, retained eye-level yield, and rejection reasons;
- cost per accepted image and observed JPEG storage per response;
- projected requests and cost for Pilot, remaining floor-120 deficits, and uniform-200, with the formula and
  uncertainty/scenario assumptions shown rather than hidden in a retry multiplier.

The source HRFFA JPEG q92 mean of about 210 KB/image is likewise only an initial storage reference. Replace it
with Validation bytes/image before Pilot, reserve at least twice the projected raw/archive footprint during
collection, and prune only through the audited command.

Set both a per-run request cap and a spend cap. A top-up planner that would exceed either cap must stop and print
the remaining accepted deficits; it must never submit automatically. The operator must approve the post-
Validation model profile, yield/cost report, and revised caps before the 380-request Pilot can be submitted.

## 8. Implementation and execution sequence

1. Port code, notices, YAML, and mocked tests; generalize stage sizes and model/API profile.
2. Add model bootstrap/checksum verification, the capped source-quota sampler, per-bin evaluation, paired
   person-cluster bootstrap, and their tests.
3. Select and manually label 200-300 `test_profiles` records from train/test-disjoint people. Freeze its hash,
   annotation audit, metric code, seeds, and section 6 decision protocol before any paid generation.
4. Dry-run `plan` and verify exact quotas, sign/sector coverage, request schemas, immutability, relative paths,
   account model-access preflight, and provisional request/spend caps.
5. Create Validation (19), necessarily including all direction sectors, both profile signs, camera elevations
   near 30/45/60 deg, and a back-of-head image. Approve the sign table, camera-relative elevation classifier,
   TownCentre crop margin, visual quality, and actual returned model. Preserve valid eye-level outputs.
6. Between Validation and Pilot, aggregate actual usage, charge if observable, response storage, QA yield, and
   retry causes. Re-estimate Pilot/floor/uniform request and spend scenarios, freeze the model profile and gates,
   and obtain operator approval for revised caps.
7. Create Pilot (380), run full-resolution DEIM/SixD QA, human rear review, and publish the yield/confusion report.
8. Recalculate and submit floor-120 top-ups in 250-500 request shards until the 841 high-angle-qualified quotas
   are filled or a gate/cap stops the run. Retain valid eye-level outputs from every shard.
9. Materialize TownCentre-sized crops and both elevation-filtered manifests; verify distributions,
   repetition-cap feasibility, and no train/test/`test_profiles` identity leakage.
10. Run R0/R1 for three matched seeds and apply the paired-CI decision rule in section 6. Run R1E only as the
    separately reported all-elevations ablation.
11. Only on success, top up toward the 3,350 high-angle-qualified uniform-200 quotas. Rerun 10 %, and add the
    20 % arm only after the 1,345-unique-image condition is met.

Implementation is complete only when a no-network fixture run exercises plan -> collect -> QA -> materialize ->
manifest -> dataset load, and all existing tests still pass. Data generation is complete only when the report's
high-angle-qualified per-band counts meet the chosen target and all retained eye-level records are accounted for;
a completed Batch by itself is not completion.
