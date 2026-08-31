import json
from pathlib import Path

import cv2
import numpy as np

from biternionnet.data import write_manifest
from biternionnet.synthetic.generate import create_plan, load_state, read_plan, sha256_file, write_jsonl
from biternionnet.synthetic.materialize import materialize_run

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "synthetic_towncentre.yaml"


def _image(path: Path, height: int, width: int, value: int = 120):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((height, width, 3), value, dtype=np.uint8))


def test_materialize_retains_eye_level_but_keeps_it_out_of_high_angle_view(tmp_path):
    run = create_plan(CONFIG, "validation", "fixture-validation", tmp_path / "synthetic", seed=1)
    state = load_state(run)
    record = next(iter(read_plan(run, state).values()))
    _image(run / "images" / record["filename"], 200, 200)
    annotation = {
        "custom_id": record["custom_id"],
        "filename": record["filename"],
        "head_box_xyxy": [60, 50, 140, 150],
        "pan_quality_pass": True,
        "angle_deg": 90.0,
        "abs_pan_bin": 90,
        "label_source": "sixdrepnet360",
        "label_confidence": 1.0,
        "camera_elevation_class": "eye_level_or_low_angle",
        "counts_toward_high_angle_quota": False,
    }
    annotations = run / "approved_annotations.jsonl"
    write_jsonl(annotations, [annotation])
    review = run / "human_review.csv"
    review.write_text("fixture-review\n", encoding="utf-8")
    approval = {
        "approved": True,
        "crop_margin": 0.15,
        "annotations_path": annotations.name,
        "annotations_sha256": sha256_file(annotations),
        "review_path": review.name,
        "review_sha256": sha256_file(review),
    }
    (run / "approval.json").write_text(json.dumps(approval), encoding="utf-8")

    _image(tmp_path / "real" / "train.jpg", 20, 18)
    _image(tmp_path / "real" / "test.jpg", 30, 25)
    anchor = tmp_path / "towncentre" / "manifest.jsonl"
    neighbour = tmp_path / "towncentre" / "manifest_nb3.jsonl"
    rows = [
        {"split": "train", "task": "angle_deg", "angle_deg": 0.0, "image": "../real/train.jpg"},
        {"split": "test", "task": "angle_deg", "angle_deg": 180.0, "image": "../real/test.jpg"},
    ]
    write_manifest(rows, anchor)
    write_manifest(rows, neighbour)
    original = neighbour.read_bytes()
    output = tmp_path / "materialized"
    report = materialize_run(
        run,
        output_root=output,
        anchor_manifest=anchor,
        neighbour_manifest=neighbour,
        seed=1,
    )
    assert report["materialized_this_run"] == 1
    assert report["high_angle_total"] == 0
    all_rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert all_rows[0]["camera_elevation_class"] == "eye_level_or_low_angle"
    assert (output / "manifest_high_angle.jsonl").read_text() == ""
    high_combined = tmp_path / "towncentre" / "manifest_nb3_synthetic.jsonl"
    all_combined = tmp_path / "towncentre" / "manifest_nb3_synthetic_all_elevations.jsonl"
    assert high_combined.read_bytes() == original
    assert len(all_combined.read_text().splitlines()) == 3
