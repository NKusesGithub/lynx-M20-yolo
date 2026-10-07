"""
Re-project frames from the M20 front camera into a virtual Orbbec DC1 colour
camera, for a depth model that was trained on DC1 images.

A monocular depth model learns metric scale for the lens it was trained on.
The M20 camera has a wider view (about 92 degrees) than the DC1 (about 66
degrees), so on M20 frames the model sees objects smaller than it expects and
reads them as farther away. OrbbecView removes the M20 lens distortion and
crops the frame to the DC1 focal length and size. The virtual camera has the
same optical centre and orientation as the M20 camera, so its depth values (Z)
apply to the M20 frame unchanged. Pixels outside the DC1 view get depth 0.
"""

import cv2
import numpy as np

from box_distance import scale_camera_matrix


class OrbbecView:
    def __init__(self, camera, orbbec, frame_size):
        """camera: the M20 calibration (camera_matrix, dist_coeffs, calib_size).
        orbbec: the DC1 colour camera (camera_matrix, size)."""
        self.K = scale_camera_matrix(camera["camera_matrix"], camera["calib_size"], frame_size)
        self.D = np.array(camera["dist_coeffs"], np.float64)
        self.dst_K = np.array(orbbec["camera_matrix"], np.float64).reshape(3, 3)
        self.dst_size = tuple(orbbec["size"])
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.K, self.D, None, self.dst_K, self.dst_size, cv2.CV_32FC1)
        self._back = {}  # (out_size, scale) -> remap tables from frame to Orbbec pixels

    def to_orbbec(self, frame):
        """The frame as the DC1 colour camera would see it."""
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def to_frame(self, img, out_size, scale):
        """Map an image in Orbbec pixels (for example a depth map) back onto the
        frame, resized by scale to out_size. Pixels outside the DC1 view are 0."""
        key = (tuple(out_size), scale)
        if key not in self._back:
            self._back[key] = self._back_maps(out_size, scale)
        bx, by = self._back[key]
        return cv2.remap(img, bx, by, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _back_maps(self, out_size, scale):
        w, h = out_size
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        src = np.stack([u.ravel(), v.ravel()], 1) / scale
        dst = cv2.undistortPoints(src.reshape(-1, 1, 2), self.K, self.D, P=self.dst_K).reshape(-1, 2)
        # undistortPoints is iterative and can be wrong far into the lens corners.
        # Project back and drop the points that do not return to where they started.
        norm = (dst - self.dst_K[[0, 1], [2, 2]]) / self.dst_K[[0, 1], [0, 1]]
        rays = np.hstack([norm, np.ones((len(norm), 1))])
        back, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), self.K, self.D)
        bad = np.linalg.norm(back.reshape(-1, 2) - src, axis=1) > 1.0
        dst[bad] = -1  # outside the image, so remap gives 0
        return dst[:, 0].reshape(h, w).copy(), dst[:, 1].reshape(h, w).copy()
