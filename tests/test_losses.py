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



def test_probs2deg_centre_and_quadint():
    from biternionnet.losses import probs2deg_centre, probs2deg_quadint

    centres = np.array([0, 90, 180, 270], dtype=np.float32)
    probs = np.array([[0.1, 0.7, 0.2, 0.0], [0.6, 0.2, 0.0, 0.2]])
    assert probs2deg_centre(probs, centres).tolist() == [90.0, 0.0]
    quad = probs2deg_quadint(probs, centres)
    # Vertex is pulled towards the heavier neighbour, and stays on the circle.
    assert 90.0 < quad[0] < 180.0
    assert quad[1] == 0.0 or quad[1] == 360.0 or (0.0 <= quad[1] < 360.0)
    symmetric = probs2deg_quadint(np.array([[0.2, 0.6, 0.2, 0.0]]), centres)
    assert np.isclose(symmetric[0], 90.0)
