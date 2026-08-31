# 016: Validation failure repair

Created 2026-08-31 (user request: repair the failed Validation records)

## 0. Selection and API contract

The prior automatic QA had one explicit failure at +20 degrees and an invalid pitch calibration controlled by
the +40 and -50 degree residuals. `edit-plan --include-pitch-calibration-tail` now selects both q90 tail records
in addition to ordinary image-level failures and binds the source calibration path and SHA-256 in Batch state.

All paid operations used `gpt-image-2`, Batch `/v1/images/edits`, `quality=low`, and the failed JPEG as image
input. The official model page confirms support for Batch and the image-edit endpoint. Five requests were sent
across three completed Batches with zero request failures. Conservative submission caps totalled $0.25; actual
account charges remain unverified.

## 1. Edit sequence

1. `validation-v011-edit04-calibration`, Batch
   `batch_6a94fcb90bcc8190a1d25bc9b15b9c26`: edited +20, +40, and -50 degrees. The +40 and -50 records used
   nose-aligned physical reference objects with 20- and 25-degree downward head corrections. Pitch calibration
   passed at 19.5006 degrees, but the +20 pan moved from 45.10 to 57.29 degrees and remained a direction failure.
2. `validation-v012-edit05-direction`, Batch
   `batch_6a94fd9b3f5481909d7f8d57bab110db`: told the +20 record to rotate about 37.3 degrees toward image-left from
   its measured current pose. Pan reached 24.10 degrees and all 19 direction/pan checks passed, but pitch became
   too shallow at -23.35 degrees and invalidated calibration.
3. `validation-v013-edit06-pitch`, Batch
   `batch_6a94fe3226108190b6e5f71dacd681d1`: preserved the corrected pan and added a small blue beanbag directly
   below the nose to induce a 25-degree downward whole-head rotation. The final +20 estimates are pan 13.4821
   degrees (error 6.5179), DEIM `front` (distance 0), and pitch -59.4559 degrees.

## 2. Final automatic result

- image quality: 19/19;
- pan/direction quality: 19/19;
- high-angle matches in the stable `abs_pan <= 60` range: 7;
- pitch calibration: valid, q90+buffer threshold 18.28524 degrees versus the fixed 25-degree maximum;
- execution providers: DEIMv2 CUDA, SixDRepNet360 TensorRT, HRFFA ViT-L TensorRT, all batch 1.

The three edited target regions were visually inspected. Heads, necks, and shoulders are coherent; the small
reference objects remain separate from the target crops. `validation-v013-edit06-pitch` is the current human
review target. Human CSV review and explicit sign calibration approval are still required before Validation can
be approved.

The usage report was also corrected for edit cycles: carried-forward images no longer make
`failed_or_missing` negative, `completed_requests` is separate from `completed_images`, and automatic QA is
reported before human approval.
