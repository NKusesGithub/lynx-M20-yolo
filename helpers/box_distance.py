"""
Distance to upright boxes from geometry alone: the box's known real height
(main estimate) and the floor contact point (cross-check). No depth model.
See docs/BOX_DISTANCE.md for setup and calibration.
"""

import math
from collections import defaultdict, deque

import cv2
import numpy as np


def scale_camera_matrix(camera_matrix, calib_size, frame_size):
    """K for frame_size, from a K calibrated at calib_size."""
    K = np.array(camera_matrix, np.float64).reshape(3, 3)
    (cw, ch), (w, h) = calib_size, frame_size
    K[0] *= w / cw
    K[1] *= h / ch
    return K


def height_distance(pixel_height, fy, box_height_m):
    """Straight-line distance to the box face, or None if the height is unusable."""
    if pixel_height is None or pixel_height <= 0:
        return None
    return fy * box_height_m / pixel_height


def floor_distance(v_bottom, fy, cy, camera_height_m, pitch_rad):
    """Horizontal distance along the floor to where the box stands, or None
    if that point is at or above the horizon."""
    angle = pitch_rad + math.atan((v_bottom - cy) / fy)
    if angle <= 0:
        return None
    return camera_height_m / math.tan(angle)


def mask_top_bottom(mask, x_center, band_px):
    """Topmost and bottommost mask rows within a band of columns around
    x_center, or None if the band holds no mask pixels."""
    x0 = max(int(x_center) - band_px, 0)
    x1 = min(int(x_center) + band_px + 1, mask.shape[1])
    rows = np.nonzero(mask[:, x0:x1].any(axis=1))[0]
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1])


def touches_edge(v_top, v_bottom, frame_height, margin=2):
    """True if the box may be cut off by the top or bottom of the frame."""
    return v_top <= margin or v_bottom >= frame_height - 1 - margin


def undistort_rows(u, v_top, v_bottom, K, D):
    """Undistorted (v_top, v_bottom) of two points in column u."""
    pts = np.array([[[u, v_top]], [[u, v_bottom]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, D, P=K)
    return float(und[0, 0, 1]), float(und[1, 0, 1])


def disagree(a, b, ratio):
    """True if a and b differ by more than ratio of a."""
    return a is not None and b is not None and abs(a - b) > ratio * a


class TrackSmoother:
    """Median of the last n values per track ID."""

    def __init__(self, n):
        self.history = defaultdict(lambda: deque(maxlen=n))

    def update(self, track_id, value):
        if value is None or track_id is None:
            return value
        self.history[track_id].append(value)
        return float(np.median(self.history[track_id]))

    def prune(self, active_ids):
        for tid in set(self.history) - set(active_ids):
            del self.history[tid]


class BoxDistance:
    """Applies both estimates and smoothing to the detections of one frame."""

    def __init__(self, cfg, frame_size):
        self.cfg = cfg
        # K is only valid at the calibration resolution; scale it if the
        # stream comes in at another size.
        calib_size = cfg.get("calib_size", [1280, 720])
        if tuple(frame_size) != tuple(calib_size):
            print(f"[distance] frame is {frame_size[0]}x{frame_size[1]}, "
                  f"calibration is {calib_size[0]}x{calib_size[1]}: scaling K")
        self.K = scale_camera_matrix(cfg["camera_matrix"], calib_size, frame_size)
        self.D = np.array(cfg["dist_coeffs"], np.float64)
        self.fy, self.cy = self.K[1, 1], self.K[1, 2]
        self.pitch = math.radians(cfg["camera_pitch_deg"])
        self.names = set(cfg["box_class_names"])
        self.h_smooth = TrackSmoother(cfg["smooth_n"])
        self.f_smooth = TrackSmoother(cfg["smooth_n"])

    def measure(self, det, frame_height):
        """Raw (height_m, floor_m, (u, v_top, v_bottom)) for one detection;
        distances are None when they cannot be estimated."""
        x1, y1, x2, y2 = det["box"]
        u = (x1 + x2) / 2
        band = max(2, int((x2 - x1) / 10))
        top_bottom = None
        if det["poly"] is not None and len(det["poly"]) >= 3:
            # Rasterise the polygon only inside its box, at full resolution.
            ox, oy = int(x1), int(y1)
            crop = np.zeros((int(y2) - oy + 2, int(x2) - ox + 2), np.uint8)
            cv2.fillPoly(crop, [(det["poly"] - (ox, oy)).astype(np.int32)], 1)
            tb = mask_top_bottom(crop, u - ox, band)
            if tb is not None:
                top_bottom = (tb[0] + oy, tb[1] + oy)
        v_top, v_bottom = top_bottom or (y1, y2)

        if touches_edge(y1, y2, frame_height):
            return None, None, (u, v_top, v_bottom)
        und_top, und_bottom = undistort_rows(u, v_top, v_bottom, self.K, self.D)
        pixel_height = und_bottom - und_top
        if pixel_height < self.cfg["min_pixel_height"]:
            return None, None, (u, v_top, v_bottom)
        h = height_distance(pixel_height, self.fy, self.cfg["box_height_m"])
        f = floor_distance(und_bottom, self.fy, self.cy, self.cfg["camera_height_m"], self.pitch)
        return h, f, (u, v_top, v_bottom)

    def update(self, dets, frame_height):
        """Adds 'dist' to each box detection: smoothed height and floor
        distances, an uncertain flag and the measured points."""
        for d in dets:
            if d["name"] not in self.names:
                continue
            h, f, points = self.measure(d, frame_height)
            h = self.h_smooth.update(d["id"], h)
            f = self.f_smooth.update(d["id"], f)
            d["dist"] = {"height": h, "floor": f, "points": points,
                         "uncertain": disagree(h, f, self.cfg["disagree_ratio"])}
        active = [d["id"] for d in dets if d["id"] is not None]
        self.h_smooth.prune(active)
        self.f_smooth.prune(active)
