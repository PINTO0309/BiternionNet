import numpy as np

from biternionnet.donut import cyclic_filter, donut_from_angles, donut_heatmap, gaussfilter, mkheatmap_deg


def test_mkheatmap_counts_and_wrap():
    hm = mkheatmap_deg(np.array([0.0, 359.9, 360.0, -1.0, 180.0]), nbins=360)
    assert hm.sum() == 5
    assert hm[0] == 2 and hm[359] == 2 and hm[180] == 1  # 360 -> bin 0, -1 -> bin 359


def test_cyclic_filter_wraps_and_preserves_mass():
    a = np.zeros(360)
    a[0] = 1.0
    out = cyclic_filter(a, gaussfilter(41))
    assert np.isclose(out.sum(), 1.0)
    assert out[359] > 0 and out[1] > 0 and np.isclose(out[1], out[359])  # symmetric across the wrap


def test_donut_heatmap_geometry():
    hm = np.linspace(0, 1, 3600)
    img = donut_heatmap(hm, bg=(201, 201), R=50)
    assert img.shape == (201, 201, 4)
    assert img[100, 100, 3] == 0.0  # hole in the middle is transparent
    assert img[100, 3, 3] > 0.0  # ring at the left edge is painted
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_donut_from_angles_notebook_scaling():
    angles = np.full(400, 90.0)
    hm = donut_from_angles(angles)  # n/400 == 1 -> peak equals smoothed count peak
    assert hm.max() > 1.0 and np.argmax(hm) == 900  # 90 deg at 3600 bins
    assert hm[:800].max() < 1e-6  # mass stays local


def test_render_donut_figure_smoke(tmp_path):
    import cv2

    from biternionnet.data import write_manifest
    from biternionnet.donut import render_donut_figure
    from biternionnet.train import train_model

    for i, angle in enumerate([0.0, 90.0, 180.0, 270.0]):
        path = tmp_path / "images" / f"{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((50, 50, 3), 40 + i * 30, dtype=np.uint8))
    write_manifest(
        [
            {"split": "train", "image": "images/0.jpg", "task": "angle_deg", "angle_deg": 0.0},
            {"split": "train", "image": "images/1.jpg", "task": "angle_deg", "angle_deg": 90.0},
            {"split": "test", "image": "images/2.jpg", "task": "angle_deg", "angle_deg": 180.0},
            {"split": "test", "image": "images/3.jpg", "task": "angle_deg", "angle_deg": 270.0},
        ],
        tmp_path / "manifest.jsonl",
    )
    result = train_model("smoke-biternion", tmp_path / "manifest.jsonl", tmp_path / "run", epochs=1, batch_size=2, device_name="cpu")
    out = tmp_path / "donut.jpg"
    info = render_donut_figure([result["last_checkpoint"]], tmp_path / "manifest.jsonl", "test", out, device_name="cpu")
    assert out.exists() and out.stat().st_size > 10_000
    assert info["panels"][0] == ("ground truth", 2)
