"""Deterministic source-quota sampling with a synthetic repetition cap."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import Any

from torch.utils.data import Sampler


class SourceQuotaSampler(Sampler[int]):
    """Draw an exact synthetic fraction while capping each synthetic item.

    Real records may cycle when ``num_samples`` exceeds their count. Synthetic
    records are distributed as evenly as possible; no item is yielded more than
    ``max_synthetic_repeats`` times in an epoch.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        *,
        synthetic_fraction: float,
        num_samples: int,
        max_synthetic_repeats: int = 4,
        seed: int = 0,
    ) -> None:
        if not 0.0 < synthetic_fraction < 1.0:
            raise ValueError("synthetic_fraction must be between 0 and 1")
        if num_samples <= 0 or max_synthetic_repeats <= 0:
            raise ValueError("num_samples and max_synthetic_repeats must be positive")
        self.real_indices = [index for index, row in enumerate(records) if row.get("source") != "synthetic"]
        self.synthetic_indices = [index for index, row in enumerate(records) if row.get("source") == "synthetic"]
        if not self.real_indices or not self.synthetic_indices:
            raise ValueError("source quota sampling requires both real and synthetic records")
        self.synthetic_draws = int(round(num_samples * synthetic_fraction))
        self.real_draws = num_samples - self.synthetic_draws
        if self.synthetic_draws == 0 or self.real_draws == 0:
            raise ValueError("synthetic_fraction and num_samples must allocate both sources")
        if self.synthetic_draws > len(self.synthetic_indices) * max_synthetic_repeats:
            minimum = math.ceil(self.synthetic_draws / max_synthetic_repeats)
            raise ValueError(
                f"synthetic repetition cap cannot satisfy {self.synthetic_draws} draws; "
                f"need at least {minimum} unique synthetic records"
            )
        self.num_samples = num_samples
        self.max_synthetic_repeats = max_synthetic_repeats
        self.seed = seed
        self.epoch = 0
        self.last_report: dict[str, Any] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @staticmethod
    def _cycled(indices: list[int], count: int, rng: random.Random) -> list[int]:
        result: list[int] = []
        while len(result) < count:
            cycle = list(indices)
            rng.shuffle(cycle)
            result.extend(cycle[: count - len(result)])
        return result

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        synthetic = list(self.synthetic_indices)
        rng.shuffle(synthetic)
        quotient, remainder = divmod(self.synthetic_draws, len(synthetic))
        counts = {index: quotient for index in synthetic}
        offset = (self.epoch * max(1, remainder)) % len(synthetic)
        for position in range(remainder):
            counts[synthetic[(offset + position) % len(synthetic)]] += 1
        if max(counts.values(), default=0) > self.max_synthetic_repeats:
            raise RuntimeError("internal synthetic repetition-cap violation")
        synthetic_draws = [index for index in synthetic for _ in range(counts[index])]
        real_draws = self._cycled(self.real_indices, self.real_draws, rng)
        combined = real_draws + synthetic_draws
        rng.shuffle(combined)
        repetition = Counter(synthetic_draws)
        values = sorted(repetition.values())
        self.last_report = {
            "epoch": self.epoch,
            "num_samples": len(combined),
            "real_draws": len(real_draws),
            "synthetic_draws": len(synthetic_draws),
            "realized_synthetic_fraction": len(synthetic_draws) / len(combined),
            "synthetic_unique": len(repetition),
            "synthetic_repeat_min": values[0],
            "synthetic_repeat_q10": values[int(0.10 * (len(values) - 1))],
            "synthetic_repeat_median": values[len(values) // 2],
            "synthetic_repeat_q90": values[int(0.90 * (len(values) - 1))],
            "synthetic_repeat_max": values[-1],
        }
        return iter(combined)

    def __len__(self) -> int:
        return self.num_samples
