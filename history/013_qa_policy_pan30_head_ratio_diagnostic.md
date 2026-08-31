# 013: QA policy: pan 30 deg and diagnostic-only head-height ratio

Created 2026-08-31 (user request: change requested-pan tolerance to 30 degrees and remove head-height ratio from
QA acceptance)

## 0. Decision

Generation configurations already bound into Batch state remain byte-for-byte unchanged. The current QA policy
is instead stored in `configs/synthetic_qa_policy_v2.yaml` and applied by default:

- circular error from requested pan must be at most 30 degrees;
- DEIMv2 head-height ratio continues to be measured and reported, but has no pass/fail effect;
- head count, margins, upper-body context, broad direction, pose, duplicate, dimensions, and image validity
  checks remain active.

The policy path and SHA-256 are bound into `qa_report.json` and Validation `pitch_calibration.json`. Each
`auto_qa.jsonl` row also records `qa_pan_tolerance_deg=30.0` and
`head_height_ratio_gate_active=false`. QA-guided edit planning reads the row-level tolerance so that an older
generation config cannot silently restore the former 25-degree threshold.

## 1. Validation v010 re-evaluation

The previous QA and review artifacts were preserved under
`data/synthetic/batches/validation-v010-edit03-object/pre_policy30_qa/`, then the 19 images were re-run through
single-batch QA. DEIMv2 used CUDA; SixDRepNet360 and HRFFA ViT-L used TensorRT with the runtime-scoped batch-1
cache.

The +20-degree image
`validation-v010-edit03-object_000003--pan+020_cam+60_pitch+008.jpg` now passes the numeric pan check: estimated
pan is 45.095 degrees, for a 25.095-degree circular error. Its 0.24984 head-height ratio is diagnostic only.
It nevertheless remains rejected because DEIMv2 reports `left_side` while the expected broad class is `front`;
`direction_conflict` is an independent unchanged gate.

The aggregate result therefore remains 18/19 automatic pan-quality passes. Pitch calibration also remains
invalid: the data threshold is 26.05935 degrees against the independently predeclared 25-degree hard maximum.
No paid API operation was performed for this policy-only re-evaluation.
