import math

import numpy as np
import pytest

from helpers.box_distance import (BoxDistance, TrackSmoother, disagree, floor_distance,
                          height_distance, mask_top_bottom, touches_edge)

CFG = {
    "box_class_names": ["box"], "box_height_m": 0.30, "camera_height_m": 0.45,
    "camera_pitch_deg": 0.0, "smooth_n": 3, "disagree_ratio": 0.2, "min_pixel_height": 15,
    "camera_matrix": [625, 0, 640, 0, 625, 360, 0, 0, 1], "dist_coeffs": [0, 0, 0, 0, 0],
}


def test_height_distance():
    assert height_distance(62.5, 625, 0.30) == pytest.approx(3.0)


@pytest.mark.parametrize("px", [0, -5, None])
def test_height_distance_unusable(px):
    assert height_distance(px, 625, 0.30) is None


def test_floor_distance():
    # 0.45 m camera, no pitch, bottom 93.75 px below centre: atan(0.15) -> 3 m.
    assert floor_distance(360 + 93.75, 625, 360, 0.45, 0.0) == pytest.approx(3.0)


def test_floor_distance_pitch_adds_to_angle():
    pitched = floor_distance(360, 625, 360, 0.45, math.radians(10))
    assert pitched == pytest.approx(0.45 / math.tan(math.radians(10)))


@pytest.mark.parametrize("v", [360, 300])
def test_floor_distance_at_or_above_horizon(v):
    assert floor_distance(v, 625, 360, 0.45, 0.0) is None


def test_mask_top_bottom_uses_band_only():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 40:60] = True
    mask[5:95, 0:5] = True  # outside the band, must be ignored
    assert mask_top_bottom(mask, 50, 3) == (20, 79)


def test_mask_top_bottom_empty_band():
    mask = np.zeros((100, 100), bool)
    mask[20:80, 0:10] = True
    assert mask_top_bottom(mask, 50, 3) is None


@pytest.mark.parametrize("top,bottom,edge", [(0, 50, True), (10, 719, True), (10, 600, False)])
def test_touches_edge(top, bottom, edge):
    assert touches_edge(top, bottom, 720) is edge


def test_disagree():
    assert disagree(3.0, 2.0, 0.2)
    assert not disagree(3.0, 2.8, 0.2)
    assert not disagree(3.0, None, 0.2)


def test_smoother_median_and_prune():
    s = TrackSmoother(3)
    for v in (1.0, 9.0, 2.0, 3.0):
        out = s.update(7, v)
    assert out == 3.0  # median of the last 3: 9, 2, 3
    assert s.update(None, 5.0) == 5.0
    assert s.update(7, None) is None
    s.prune([])
    assert 7 not in s.history


def _det(y1, y2):
    return {"box": np.array([600.0, y1, 680.0, y2]), "name": "box", "id": 1, "poly": None}


def test_box_distance_end_to_end():
    bd = BoxDistance(CFG, (1280, 720))
    det = _det(400, 462.5)  # 62.5 px tall -> 3 m
    bd.update([det], 720)
    assert det["dist"]["height"] == pytest.approx(3.0, rel=0.02)


def test_box_distance_edge_and_tiny_boxes():
    bd = BoxDistance(CFG, (1280, 720))
    edge, tiny = _det(0, 100), _det(400, 410)
    bd.update([edge, tiny], 720)
    assert edge["dist"]["height"] is None
    assert tiny["dist"]["height"] is None


def test_box_distance_scales_k_for_other_resolution():
    bd = BoxDistance(CFG, (640, 360))
    assert bd.fy == pytest.approx(312.5)
    det = _det(200, 231.25)  # same box at half resolution -> still 3 m
    bd.update([det], 360)
    assert det["dist"]["height"] == pytest.approx(3.0, rel=0.02)
