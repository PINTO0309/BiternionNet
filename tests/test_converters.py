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
