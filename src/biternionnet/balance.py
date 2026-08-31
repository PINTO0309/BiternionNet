"""Composite manifest builder: current / balanced training patterns with a shared enlarged test side.

From a combined manifest (TownCentre anchors + neighbour frames + synthetic records) this module
writes two manifests that differ only in their *train* records:

- ``manifest_current``: the combined manifest's train records unchanged (minus the synthetic holdout);
- ``manifest_balanced``: anchors and synthetic kept in full, but neighbour frames re-selected per
  flip-effective 10-degree pan bin so that every bin reaches the same target ``T`` (nearest-to-anchor
  frames first, deterministic).

Both manifests share a byte-identical test side:

- ``test``: the original test anchors, unchanged (comparability with all earlier runs);
- ``test_neighbor``: the +-``neighbor_cap`` unlabelled frames of every test anchor, same angle;
- ``test_synthetic``: a per-bin stratified holdout of the synthetic records, removed from train in
  BOTH manifests. Evaluate these extra splits with ``biternion-eval --split ... [--per-bin]``;
  checkpoint selection and the per-epoch log keep using ``test`` only.

Flip-effective counting: training flips with p=0.5 map theta -> 360-theta, so what the network sees
per bin is ``(raw(theta)+raw(360-theta))/2``; quotas are computed on mirror pairs of bins.
"""

from __future__ import annotations

import math
import os
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import read_manifest, write_manifest

BIN_WIDTH = 10.0
N_BINS = 36


def bin_of(angle_deg: float) -> int:
    return int(((angle_deg + BIN_WIDTH / 2.0) % 360.0) // BIN_WIDTH)


def flip_effective(counts: np.ndarray) -> np.ndarray:
    return (counts + counts[(-np.arange(N_BINS)) % N_BINS]) / 2.0


def _head_key(record: dict[str, Any]) -> tuple[int, int]:
    """(person_id, frame) from a TownCentre image path like .../000084_0000100004_....jpg"""
    name = Path(record["image"]).name
    parts = name.split("_")
    return int(parts[1]), int(parts[0])


def _frame_index(source: Path) -> dict[int, dict[int, Path]]:
    index: dict[int, dict[int, Path]] = defaultdict(dict)
    for directory in source.iterdir():
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() == ".txt":
                continue
            parts = path.name.split("_")
            try:
                index[int(parts[1])][int(parts[0])] = path
            except (IndexError, ValueError):
                continue
    return index


def _neighbor_record(anchor: dict[str, Any], path: Path, offset: int, split: str, manifest_dir: Path) -> dict[str, Any]:
    return {
        "split": split,
        "image": os.path.relpath(path.resolve(), manifest_dir.resolve()),
        "task": "angle_deg",
        "angle_deg": anchor["angle_deg"],
        "source": "neighbor",
        "anchor_frame": _head_key(anchor)[1],
        "frame_offset": offset,
    }


def build_composite_manifests(
    combined: str | Path,
    source: str | Path,
    output_dir: str | Path,
    *,
    neighbor_cap: int = 10,
    balance_target: int | None = None,
    synthetic_holdout: float = 0.1,
    current_cap_effective: int | None = None,
    seed: int = 0,
    current_name: str = "manifest_current.jsonl",
    balanced_name: str = "manifest_balanced.jsonl",
    current_capped_name: str = "manifest_current_capped.jsonl",
) -> dict[str, Any]:
    combined = Path(combined)
    source = Path(source)
    output_dir = Path(output_dir)
    rng = random.Random(seed)
    records = read_manifest(combined)
    index = _frame_index(source)

    anchors = [r for r in records if r["split"] == "train" and r.get("source") in (None, "real")]
    current_neighbors = [r for r in records if r["split"] == "train" and r.get("source") == "neighbor"]
    synthetic = [r for r in records if r["split"] == "train" and r.get("source") == "synthetic"]
    test_anchors = [r for r in records if r["split"] == "test"]
    others = [r for r in records if r["split"] not in ("train", "test")]

    # --- synthetic holdout: hash-bound so membership is stable under later additions ----------
    # A record is held out iff crc32(custom_id) lands below the holdout fraction. Adding new
    # synthetic batches therefore never moves an existing record between train and test_synthetic
    # (stratification per bin is statistical, not exact). ``seed`` salts the hash.
    threshold = int(round(synthetic_holdout * 10000))

    def _held_out(r: dict[str, Any]) -> bool:
        key = f"{seed}:{r.get('custom_id', r['image'])}".encode("utf-8")
        return (zlib.crc32(key) % 10000) < threshold

    holdout = [r for r in synthetic if _held_out(r)]
    synthetic_train = [r for r in synthetic if not _held_out(r)]
    test_synthetic = [{**r, "split": "test_synthetic"} for r in sorted(holdout, key=lambda r: str(r.get("custom_id", r["image"])))]

    # --- test-side neighbours ----------------------------------------------------------------
    test_neighbors: list[dict[str, Any]] = []
    for r in test_anchors:
        pid, frame = _head_key(r)
        for offset in range(-neighbor_cap, neighbor_cap + 1):
            path = index.get(pid, {}).get(frame + offset)
            if offset == 0 or path is None:
                continue
            test_neighbors.append(_neighbor_record(r, path, offset, "test_neighbor", output_dir))

    # --- balanced train neighbours: per mirror-pair quota, nearest offsets first -------------
    anchor_counts = np.zeros(N_BINS)
    syn_counts = np.zeros(N_BINS)
    for r in anchors:
        anchor_counts[bin_of(r["angle_deg"])] += 1
    for r in synthetic_train:
        syn_counts[bin_of(r["angle_deg"])] += 1
    candidates: dict[int, list[tuple[int, float, dict[str, Any], Path, int]]] = defaultdict(list)
    avail_counts = np.zeros(N_BINS)
    for r in anchors:
        pid, frame = _head_key(r)
        b = bin_of(r["angle_deg"])
        pair = min(b, (-b) % N_BINS)
        for offset in range(-neighbor_cap, neighbor_cap + 1):
            path = index.get(pid, {}).get(frame + offset)
            if offset == 0 or path is None:
                continue
            avail_counts[b] += 1
            candidates[pair].append((abs(offset), rng.random(), r, path, offset))

    fixed = flip_effective(anchor_counts + syn_counts)
    top = fixed + flip_effective(avail_counts)
    occupied = top[top > 0]
    target = balance_target if balance_target is not None else (int(math.floor(occupied.min())) if occupied.size else 0)
    balanced_neighbors: list[dict[str, Any]] = []
    achieved = fixed.copy()
    for pair in sorted(candidates):
        need_eff = max(0.0, target - fixed[pair])
        raw_needed = int(round(need_eff if pair in (0, N_BINS // 2) else 2.0 * need_eff))
        chosen = sorted(candidates[pair])[:raw_needed]
        for _, _, anchor, path, offset in chosen:
            balanced_neighbors.append(_neighbor_record(anchor, path, offset, "train", output_dir))
        gain = len(chosen) if pair in (0, N_BINS // 2) else len(chosen) / 2.0
        achieved[pair] += gain
        achieved[(-pair) % N_BINS] = achieved[pair]

    # --- current with capped peaks: trim only neighbour records of over-exposed bins ----------
    # ``current_cap_effective``: flip-effective ceiling per 10-deg bin. -1 = auto (the highest
    # effective count among the non-self-mirrored mirror pairs, so the 0/180 self-mirror spikes
    # are brought down to the level of their surroundings). Farthest |frame_offset| dropped first.
    capped_neighbors: list[dict[str, Any]] | None = None
    cap_value: float | None = None
    if current_cap_effective is not None:
        counts = np.zeros(N_BINS)
        for r in anchors + synthetic_train + current_neighbors:
            counts[bin_of(r["angle_deg"])] += 1
        eff = flip_effective(counts)
        if current_cap_effective < 0:
            non_self_mirror = [eff[b] for b in range(N_BINS) if b not in (0, N_BINS // 2)]
            cap_value = float(max(non_self_mirror))
        else:
            cap_value = float(current_cap_effective)
        keep_rank: dict[int, list[tuple[int, float, int]]] = defaultdict(list)
        for i, r in enumerate(current_neighbors):
            b = bin_of(r["angle_deg"])
            pair = min(b, (-b) % N_BINS)
            keep_rank[pair].append((abs(int(r.get("frame_offset", 0))), rng.random(), i))
        drop: set[int] = set()
        for pair in keep_rank:
            excess_eff = eff[pair] - cap_value
            if excess_eff <= 0:
                continue
            n_drop = int(round(excess_eff if pair in (0, N_BINS // 2) else 2.0 * excess_eff))
            for _, _, i in sorted(keep_rank[pair], reverse=True)[:n_drop]:
                drop.add(i)
        capped_neighbors = [r for i, r in enumerate(current_neighbors) if i not in drop]

    # --- write -------------------------------------------------------------------------------
    test_side = test_anchors + test_neighbors + test_synthetic
    output_dir.mkdir(parents=True, exist_ok=True)
    current_path = output_dir / current_name
    balanced_path = output_dir / balanced_name
    write_manifest(anchors + current_neighbors + synthetic_train + others + test_side, current_path)
    write_manifest(anchors + balanced_neighbors + synthetic_train + others + test_side, balanced_path)
    capped_path = output_dir / current_capped_name
    if capped_neighbors is not None:
        write_manifest(anchors + capped_neighbors + synthetic_train + others + test_side, capped_path)

    summary = {
        "target": target,
        "neighbor_cap": neighbor_cap,
        "seed": seed,
        "current": str(current_path),
        "balanced": str(balanced_path),
        "counts": {
            "anchors": len(anchors),
            "current_neighbors": len(current_neighbors),
            "balanced_neighbors": len(balanced_neighbors),
            "synthetic_train": len(synthetic_train),
            "test": len(test_anchors),
            "test_neighbor": len(test_neighbors),
            "test_synthetic": len(test_synthetic),
        },
        "balanced_effective_min": float(achieved[top > 0].min()) if (top > 0).any() else 0.0,
        "balanced_effective_max": float(achieved[top > 0].max()) if (top > 0).any() else 0.0,
    }
    if capped_neighbors is not None:
        summary["current_capped"] = str(capped_path)
        summary["current_cap_effective"] = cap_value
        summary["counts"]["current_capped_neighbors"] = len(capped_neighbors)
    return summary
