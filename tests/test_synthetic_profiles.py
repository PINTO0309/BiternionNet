import csv
import json

from biternionnet.data import write_manifest
from biternionnet.synthetic.profiles import finalize_test_profiles, plan_profile_candidates


def test_profile_candidates_and_final_split_are_identity_disjoint(tmp_path):
    image_root = tmp_path / "heads"
    used = image_root / "used-person"
    used.mkdir(parents=True)
    (used / "000.jpg").touch()
    for index in range(200):
        person = image_root / f"new-{index:03d}"
        person.mkdir(parents=True)
        (person / "000.jpg").touch()
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(
        [{"split": "train", "task": "angle_deg", "angle_deg": 0.0, "image": str(used / "000.jpg")}],
        manifest,
    )
    review = tmp_path / "review.csv"
    planned = plan_profile_candidates(
        image_root,
        manifest,
        review,
        candidate_count=200,
        max_frames_per_person=1,
        seed=4,
    )
    assert planned["people"] == 200
    with review.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows):
        angle = 90.0 if index < 100 else 270.0
        row["annotator1_id"] = "annotator-a"
        row["annotator1_deg"] = str(angle)
        row["annotator2_id"] = "annotator-b"
        row["annotator2_deg"] = str(angle + 2)
        row["final_angle_deg"] = str(angle + 1)
    with review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "test_profiles.jsonl"
    protocol = tmp_path / "protocol.json"
    result = finalize_test_profiles(review, manifest, output, protocol)
    assert result["records"] == result["people"] == 200
    final_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["split"] for row in final_rows} == {"test_profiles"}
    assert "used-person" not in {row["person_id"] for row in final_rows}
