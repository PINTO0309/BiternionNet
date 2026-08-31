# 015: Fixed DEIMv2-box crop margin

Created 2026-08-31 (user request: decide the DEIMv2 crop margin as 5%)

## 0. Decision

The final synthetic head crop expands the DEIMv2 `head_box_xyxy` by exactly 5% of box width on the left and
right and 5% of box height above and below. This is distinct from the HRFFA square preprocessing crop, which
uses 5% of the DEIM box long side before resizing to 320x320.

`configs/synthetic_qa_policy_v2.yaml` records `deim_crop_margin: 0.05` and rejects any other value. The approval
CLI no longer exposes `--crop-margin`; approval writes 0.05 from the hash-bound QA policy. Materialization also
fails closed if an older or modified approval contains a different value. The margin contact sheet now renders
only the selected 5% crop beside TownCentre reference crops rather than presenting multiple candidates.

Completed generation configurations remain unchanged. Regenerating QA and the margin sheet does not alter any
source image and performs no paid API operation.
