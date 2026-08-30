import json
from pathlib import Path

import cv2
import numpy as np

from biternionnet.converters import convert_towncentre_raw


def _write_towncentre_record(root: Path, person_id: str, frame: str, pan: float, valid: int = 1) -> None:
    person_dir = root / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    (person_dir / f"{frame}_{person_id}.txt").write_text(f"pan = {pan:.3f}\nvalid = {valid}\n", encoding="utf-8")
    image = np.full((50, 50, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(person_dir / f"{frame}_{person_id}_0001_0002_1.0_1.0.jpg"), image)


def test_convert_towncentre_raw_writes_angle_manifest(tmp_path):
    source = tmp_path / "TownCentreHeadImages"
    _write_towncentre_record(source, "0000000001", "000001", 90.0)
    _write_towncentre_record(source, "0000000002", "000002", 180.0)
    _write_towncentre_record(source, "0000000003", "000003", 0.0, valid=0)
    _write_towncentre_record(source, "0000000004", "000004", -45.0)
    output = tmp_path / "towncentre" / "manifest.jsonl"

    convert_towncentre_raw(source, output, train_split=1.0, seed=0)

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert {record["split"] for record in records} == {"train"}
    assert {record["task"] for record in records} == {"angle_deg"}
    assert [record["angle_deg"] for record in records] == [90.0, 180.0, 315.0]
    assert records[0]["image"].startswith("../TownCentreHeadImages/")


def _write_towncentre_track(root: Path, person_id: str, anchor_frames: list[int], pan: float, neighbors: int, valid: int = 1) -> None:
    person_dir = root / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((30, 28, 3), 100, dtype=np.uint8)
    for frame in anchor_frames:
        (person_dir / f"{frame:06d}_{person_id}.txt").write_text(f"pan = {pan:.3f}\nvalid = {valid}\n", encoding="utf-8")
        for delta in range(-neighbors, neighbors + 1):
            cv2.imwrite(str(person_dir / f"{frame + delta:06d}_{person_id}_0001_0002_1.0_1.0.jpg"), image)


def test_convert_towncentre_raw_neighbor_frames_expand_train_only(tmp_path):
    source = tmp_path / "TownCentreHeadImages"
    # person 1: two anchors 100 frames apart, 5 unlabelled frames on each side
    _write_towncentre_track(source, "0000000001", [100, 200], 30.0, neighbors=5)
    # person 2: invalid anchor, must not be expanded
    _write_towncentre_track(source, "0000000002", [300], 90.0, neighbors=5, valid=0)
    # person 3: anchor at the start of the track (no earlier frames)
    _write_towncentre_track(source, "0000000003", [400], 180.0, neighbors=0)
    (source / "0000000003" / "000401_0000000003_0001_0002_1.0_1.0.jpg").write_bytes((source / "0000000003" / "000400_0000000003_0001_0002_1.0_1.0.jpg").read_bytes())

    base = tmp_path / "base" / "manifest.jsonl"
    expanded = tmp_path / "nb3" / "manifest.jsonl"
    convert_towncentre_raw(source, base, train_split=1.0, seed=0)
    convert_towncentre_raw(source, expanded, train_split=1.0, seed=0, neighbor_frames=3)

    base_records = [json.loads(line) for line in base.read_text(encoding="utf-8").splitlines()]
    records = [json.loads(line) for line in expanded.read_text(encoding="utf-8").splitlines()]
    anchors = [r for r in records if "source" not in r]
    neighbors = [r for r in records if r.get("source") == "neighbor"]
    assert [ (r["image"], r["angle_deg"]) for r in anchors ] == [ (r["image"], r["angle_deg"]) for r in base_records ]
    assert len(anchors) == 3  # person 2 is invalid
    # person 1: 2 anchors x 6 neighbours, person 3: only frame 401 exists
    assert len(neighbors) == 2 * 6 + 1
    assert all(r["angle_deg"] == 30.0 for r in neighbors if "0000000001" in r["image"])
    assert {r["frame_offset"] for r in neighbors if "0000000001" in r["image"]} == {-3, -2, -1, 1, 2, 3}
    assert all("0000000002" not in r["image"] for r in records)
    assert len({r["image"] for r in records}) == len(records)


def test_convert_towncentre_raw_neighbor_frames_keep_test_split_identical(tmp_path):
    source = tmp_path / "TownCentreHeadImages"
    for pid in range(1, 21):
        _write_towncentre_track(source, f"{pid:010d}", [100 * pid], float(pid * 10), neighbors=2)
    base = tmp_path / "base" / "manifest.jsonl"
    expanded = tmp_path / "nb2" / "manifest.jsonl"
    convert_towncentre_raw(source, base, train_split=0.5, seed=3)
    convert_towncentre_raw(source, expanded, train_split=0.5, seed=3, neighbor_frames=2)
    base_test = [json.loads(l) for l in base.read_text().splitlines() if '"test"' in l]
    exp_test = [json.loads(l) for l in expanded.read_text().splitlines() if '"test"' in l]
    assert base_test == exp_test and len(base_test) > 0
    exp_train = [json.loads(l) for l in expanded.read_text().splitlines() if '"train"' in l]
    assert all("source" not in r for r in exp_test)
    assert sum(1 for r in exp_train if r.get("source") == "neighbor") == 4 * sum(1 for r in exp_train if "source" not in r)
