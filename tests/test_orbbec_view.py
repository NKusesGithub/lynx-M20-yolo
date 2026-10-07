import numpy as np

from orbbec_view import OrbbecView

M20 = {"calib_size": [1280, 720], "dist_coeffs": [-0.291149, 0.057760, -0.006811, 0.001601, 0.0],
       "camera_matrix": [619.97674, 0, 586.32027, 0, 625.27679, 339.90312, 0, 0, 1]}
DC1 = {"size": [640, 480], "camera_matrix": [489.47235, 0, 320.83984, 0, 489.47235, 218.48178, 0, 0, 1]}


def test_same_camera_is_identity():
    cam = {"calib_size": [64, 48], "dist_coeffs": [0, 0, 0, 0, 0],
           "camera_matrix": [50, 0, 32, 0, 50, 24, 0, 0, 1]}
    view = OrbbecView(cam, {"size": [64, 48], "camera_matrix": cam["camera_matrix"]}, (64, 48))
    img = np.random.default_rng(0).integers(1, 255, (48, 64), dtype=np.uint8)
    assert np.array_equal(view.to_orbbec(img), img)
    assert np.array_equal(view.to_frame(img, (64, 48), 1.0), img)


def test_orbbec_view_size():
    view = OrbbecView(M20, DC1, (1280, 720))
    assert view.to_orbbec(np.zeros((720, 1280, 3), np.uint8)).shape == (480, 640, 3)


def test_depth_covers_centre_but_not_wide_edges():
    # The DC1 view is narrower, so the M20 frame edges get no depth.
    view = OrbbecView(M20, DC1, (1280, 720))
    ones = np.ones((480, 640), np.float32)
    back = view.to_frame(ones, (640, 360), 0.5)
    assert back.shape == (360, 640)
    assert back[170, 293] == 1  # near the optical centre
    assert back[180, 5] == 0 and back[180, 634] == 0  # far left and right
    assert back[0, 0] == 0


def test_point_lands_in_the_same_direction():
    # A bright pixel at the DC1 principal point maps back to the M20 principal point.
    view = OrbbecView(M20, DC1, (1280, 720))
    img = np.zeros((480, 640), np.float32)
    img[218, 321] = 5.0
    back = view.to_frame(img, (1280, 720), 1.0)
    ys, xs = np.nonzero(back)
    assert abs(xs.mean() - 586.3) < 2 and abs(ys.mean() - 339.9) < 2
