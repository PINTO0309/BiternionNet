"""Compatibility helpers for the PyTorch training pipeline.

The original project used notebook-only DeepFried2 helpers from this module.
New code should import from ``biternionnet.train`` and ``biternionnet.losses``
directly; these aliases keep simple external imports from breaking.
"""

from biternionnet.losses import angle_difference_deg, bit2deg, deg2bit
from biternionnet.train import evaluate_checkpoint, train_model

__all__ = ["angle_difference_deg", "bit2deg", "deg2bit", "evaluate_checkpoint", "train_model"]

