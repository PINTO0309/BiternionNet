import numpy as np
import torch

from biternionnet.losses import CosineLoss, VonMisesBiternionLoss, angle_difference_deg, bit2deg, deg2bit, quantize_labels


def test_deg2bit_bit2deg_round_trip():
    angles = np.array([0.0, 45.0, 180.0, 359.0], dtype=np.float32)
    reconstructed = bit2deg(deg2bit(angles))
    assert np.allclose(reconstructed, angles, atol=1e-5)


def test_cyclic_angle_difference_wraps():
    err = angle_difference_deg(np.array([359.0]), np.array([1.0]))
    assert np.allclose(err, np.array([2.0]), atol=1e-5)


def test_quantize_labels_wraparound():
    borders = np.array([315, 45, 135, 225, 315], dtype=np.float32)
    values = np.array([0, 44, 90, 180, 270], dtype=np.float32)
    assert quantize_labels(values, borders).tolist() == [0, 0, 1, 2, 3]


def test_biternion_losses_backward():
    pred = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = CosineLoss()(pred, target) + VonMisesBiternionLoss(kappa=1.0)(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()

