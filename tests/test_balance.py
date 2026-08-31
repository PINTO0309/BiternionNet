import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from biternionnet.balance import bin_of, build_composite_manifests, flip_effective
from biternionnet.data import write_manifest


def _track(root: Path, pid: int, anchor_frame: int, angle: float, extra: int) -> str:
    d = root / f"{pid:010d}"
    d.mkdir(parents=True, exist_ok=True)
    img = np.full((30, 28, 3), 100, dtype=np.uint8)
    name = None
    for delta in range(-extra, extra + 1):
        fn = f"{anchor_frame + delta:06d}_{pid:010d}_0001_0002_1.0_1.0.jpg"
        cv2.imwrite(str(d / fn), img)
        if delta == 0:
            name = fn
    return name


@pytest.fixture()
def composite(tmp_path):
    source = tmp_path / "TownCentreHeadImages"
    records = []
    pid = 1
    # train anchors: dense at 180 (6 anchors), sparse at 90 (2 anchors), all with +-10 frames
    for angle, count in ((180.0, 6), (90.0, 2), (0.0, 3)):
        for _ in range(count):
            name = _track(source, pid, 1000, angle, extra=10)
            records.append({"split": "train", "image": f"../TownCentreHeadImages/{pid:010d}/{name}", "task": "angle_deg", "angle_deg": angle})
            # current-style k=3 neighbours
            for off in (-1, 1, 2):
                nb = name.replace(f"{1000:06d}", f"{1000 + off:06d}")
                records.append({"split": "train", "image": f"../TownCentreHeadImages/{pid:010d}/{nb}", "task": "angle_deg", "angle_deg": angle, "source": "neighbor", "anchor_frame": 1000, "frame_offset": off})
            pid += 1
    # test anchors with +-2 frames available
    for angle in (10.0, 200.0):
        name = _track(source, pid, 2000, angle, extra=2)
        records.append({"split": "test", "image": f"../TownCentreHeadImages/{pid:010d}/{name}", "task": "angle_deg", "angle_deg": angle})
        pid += 1
    # synthetic: 20 at 90 deg, 10 at 270 deg
    syn_dir = tmp_path / "synthetic" / "crops"
    syn_dir.mkdir(parents=True)
    for i, angle in enumerate([90.0] * 20 + [270.0] * 10):
        fn = f"syn_{i:04d}.jpg"
        cv2.imwrite(str(syn_dir / fn), np.full((28, 28, 3), 50, dtype=np.uint8))
        records.append({"split": "train", "image": f"../synthetic/crops/{fn}", "task": "angle_deg", "angle_deg": angle, "source": "synthetic", "custom_id": f"c{i:04d}", "label_source": "sixdrepnet360"})
    out = tmp_path / "towncentre"
    out.mkdir()
    combined = out / "combined.jsonl"
    write_manifest(records, combined)
    summary = build_composite_manifests(combined, source, out, neighbor_cap=10, synthetic_holdout=0.1, seed=0)
    return tmp_path, out, summary


def _load(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_test_side_identical_and_train_differs(composite):
    _, out, summary = composite
    cur = _load(out / "manifest_current.jsonl")
    bal = _load(out / "manifest_balanced.jsonl")
    side = lambda rs: [r for r in rs if r["split"] != "train"]
    assert side(cur) == side(bal)
    assert [r for r in cur if r["split"] == "train"] != [r for r in bal if r["split"] == "train"]
    # original test anchors unchanged and first in the test side
    assert [r["angle_deg"] for r in cur if r["split"] == "test"] == [10.0, 200.0]


def test_holdout_disjoint_and_stable_under_additions(composite):
    tmp_path, out, summary = composite
    cur = _load(out / "manifest_current.jsonl")
    hold = {r["custom_id"] for r in cur if r["split"] == "test_synthetic"}
    train_ids = {r.get("custom_id") for r in cur if r["split"] == "train" and r.get("source") == "synthetic"}
    assert hold and train_ids.isdisjoint(hold)
    assert len(hold) + summary["counts"]["synthetic_train"] == 30
    # add a new synthetic batch to the combined manifest: existing membership must not move
    records = _load(out / "combined.jsonl")
    syn_dir = tmp_path / "synthetic" / "crops"
    for i in range(30, 60):
        fn = f"syn_{i:04d}.jpg"
        cv2.imwrite(str(syn_dir / fn), np.full((28, 28, 3), 60, dtype=np.uint8))
        records.append({"split": "train", "image": f"../synthetic/crops/{fn}", "task": "angle_deg", "angle_deg": 45.0, "source": "synthetic", "custom_id": f"c{i:04d}", "label_source": "sixdrepnet360"})
    write_manifest(records, out / "combined.jsonl")
    build_composite_manifests(out / "combined.jsonl", tmp_path / "TownCentreHeadImages", out, neighbor_cap=10, synthetic_holdout=0.1, seed=0)
    cur2 = _load(out / "manifest_current.jsonl")
    hold2 = {r["custom_id"] for r in cur2 if r["split"] == "test_synthetic"}
    train2 = {r.get("custom_id") for r in cur2 if r["split"] == "train" and r.get("source") == "synthetic"}
    assert hold <= hold2 and train_ids <= train2  # old members stay where they were


def test_test_neighbors_cover_offsets(composite):
    _, out, _ = composite
    cur = _load(out / "manifest_current.jsonl")
    nb = [r for r in cur if r["split"] == "test_neighbor"]
    assert len(nb) == 2 * 4  # +-2 frames exist per test anchor
    assert all(r["angle_deg"] in (10.0, 200.0) and r["frame_offset"] != 0 for r in nb)


def test_balanced_bins_reach_target(composite):
    _, out, summary = composite
    bal = _load(out / "manifest_balanced.jsonl")
    counts = np.zeros(36)
    for r in bal:
        if r["split"] == "train":
            counts[bin_of(r["angle_deg"])] += 1
    eff = flip_effective(counts)
    T = summary["target"]
    occupied = [b for b in range(36) if eff[b] > 0]
    assert all(abs(eff[b] - T) <= 1.0 for b in occupied)
    assert summary["balanced_effective_min"] >= T - 1
    # balanced neighbours prefer small offsets
    offs = [abs(r["frame_offset"]) for r in bal if r["split"] == "train" and r.get("source") == "neighbor"]
    assert min(offs) == 1


def test_deterministic(composite):
    tmp_path, out, _ = composite
    first = (out / "manifest_balanced.jsonl").read_bytes()
    build_composite_manifests(out / "combined.jsonl", tmp_path / "TownCentreHeadImages", out, neighbor_cap=10, synthetic_holdout=0.1, seed=0)
    assert (out / "manifest_balanced.jsonl").read_bytes() == first


def test_current_cap_trims_only_neighbor_records(composite):
    tmp_path, out, _ = composite
    summary = build_composite_manifests(
        out / "combined.jsonl", tmp_path / "TownCentreHeadImages", out,
        neighbor_cap=10, synthetic_holdout=0.1, seed=0, current_cap_effective=-1,
    )
    capped = _load(out / "manifest_current_capped.jsonl")
    current = _load(out / "manifest_current.jsonl")
    # anchors / synthetic / test side identical; only neighbour records may be removed
    strip = lambda rs, split: [r for r in rs if r["split"] == split and r.get("source") != "neighbor"]
    for split in ("train", "test", "test_neighbor", "test_synthetic"):
        assert strip(capped, split) == strip(current, split)
    n_cur = sum(1 for r in current if r["split"] == "train" and r.get("source") == "neighbor")
    n_cap = sum(1 for r in capped if r["split"] == "train" and r.get("source") == "neighbor")
    assert n_cap <= n_cur == summary["counts"]["current_neighbors"]
    # effective counts respect the cap (within rounding)
    import numpy as np
    from biternionnet.balance import bin_of, flip_effective
    counts = np.zeros(36)
    for r in capped:
        if r["split"] == "train":
            counts[bin_of(r["angle_deg"])] += 1
    eff = flip_effective(counts)
    assert eff.max() <= summary["current_cap_effective"] + 1
