# Task: Estimate distance to cardboard boxes using known box height

## Goal

Add distance estimation to the existing live YOLO script. For each detected cardboard box, estimate how far it is from the camera **using geometry only** (known box height + camera calibration). Do **not** use a monocular depth model.

Show the distance (in metres) on screen next to each detected box.

## Context

- Robot: DEEP Robotics Lynx M20. Video comes from the robot over RTSP.
- Front camera stream: `rtsp://192.168.123.103:8554/video1`
- Rear camera stream: `rtsp://192.168.123.103:8554/video2`
- Video: H.265, 1280x720, 30 fps. Open it with OpenCV using TCP:
  `os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"`
- Existing code: `droneYolo2026/yolo26_live.py` (with `droneYolo2026/live_config.yaml`). It already runs Ultralytics YOLO with segmentation and tracking (`model.track(...)`), so each detection has a track ID.
- Python environment: conda env `camera`. Install packages with `python3 -m pip install ...` from inside that env. Tracking needs `lap>=0.5.12`.

**First, read `yolo26_live.py` and `live_config.yaml` to understand the current pipeline. Add the new feature to it; do not rewrite the script from scratch.** Keep the change small and easy to turn on and off.

## Camera calibration (front camera, 1280x720)

From the robot file `SM_USB_107X_0.json`:

```python
K = np.array([[619.97674, 0.0,       586.32027],
              [0.0,       625.27679, 339.90312],
              [0.0,       0.0,       1.0]])
D = np.array([-0.291149, 0.057760, -0.006811, 0.001601, 0.0])
```

- `fy = 625.28`, `cy = 339.90`.
- The rear camera has its own values in `SM_USB_107X_2.json`. Make the calibration configurable so the rear camera can be added later.
- These values are only valid at 1280x720. If frames are resized before inference, scale the pixel measurements (or K) to match. Ultralytics returns box coordinates in the original frame size, so normally no scaling is needed.

## Method 1 (main): known box height

The box is rectangular, so its visible width changes as it rotates. Its **height does not**, as long as it stands upright. So use height:

```
distance_m = fy * BOX_HEIGHT_M / pixel_height
```

Steps for each detected box:

1. Get the top and bottom of the box in pixels. Prefer the segmentation mask over the bounding box (tighter fit). A good option: take the mask's top and bottom y-values within a narrow band of columns around the box's horizontal centre.
2. **Remove lens distortion from those two points only**, not the whole frame (cheaper, and K stays valid):
   ```python
   pts = np.array([[[u, v_top]], [[u, v_bottom]]], dtype=np.float32)
   und = cv2.undistortPoints(pts, K, D, P=K)
   pixel_height = und[1, 0, 1] - und[0, 0, 1]
   ```
3. Compute `distance_m = fy * BOX_HEIGHT_M / pixel_height`.
4. Skip the estimate (show "?" instead) if the box touches the top or bottom edge of the frame (it may be cut off), or if `pixel_height` is very small (for example under 15 px).

## Method 2 (cross-check): floor contact point

This does not need the box size. It uses the camera's height above the floor and its downward tilt.

```python
angle = pitch_rad + math.atan((v_bottom_undistorted - cy) / fy)
distance_m = CAMERA_HEIGHT_M / math.tan(angle)     # only valid if angle > 0
```

- `v_bottom_undistorted` is the undistorted bottom point from Method 1.
- If `angle <= 0`, the point is at or above the horizon: return no estimate.
- This gives horizontal distance along the floor. Method 1 gives straight-line distance to the box face. They will differ slightly when the camera looks down; that is expected.

Show both values. If they differ by more than a set amount (for example 20%), mark the detection as uncertain (for example a different colour or a "?" suffix). Large disagreement usually means the box is tipped over, partly hidden, or two boxes were merged into one detection.

## Smoothing

Use the existing track IDs. For each track, keep the last N distance estimates (default 10) and display the **median**. Remove history for tracks that disappear.

```python
from collections import defaultdict, deque
history = defaultdict(lambda: deque(maxlen=SMOOTH_N))
```

## Configuration

Add these to `live_config.yaml` (with these defaults) instead of hard-coding them:

```yaml
box_distance:
  enabled: true
  box_class_names: ["box", "cardboard box"]   # class names that count as a box
  box_height_m: 0.30          # real box height, measure it
  camera_height_m: 0.45       # camera height above floor, measure it
  camera_pitch_deg: 10.0      # downward tilt, measure or estimate
  smooth_n: 10
  disagree_ratio: 0.20
  min_pixel_height: 15
  fy: 625.27679
  cy: 339.90312
  camera_matrix: [619.97674, 0.0, 586.32027, 0.0, 625.27679, 339.90312, 0.0, 0.0, 1.0]
  dist_coeffs: [-0.291149, 0.057760, -0.006811, 0.001601, 0.0]
```

`box_height_m`, `camera_height_m` and `camera_pitch_deg` are placeholders. Leave clear comments telling the user to measure them.

## Detection model note

Standard COCO YOLO models have **no cardboard box class**. Check which model the script loads:

- If a custom box model is available, use its class names.
- If not, keep the code working: make the box class names configurable, print a clear warning at startup if none of them exist in the model's class list, and do not crash.

## Code structure

- Put the distance logic in a separate module, for example `droneYolo2026/box_distance.py`, with small pure functions:
  - `height_distance(pixel_height, fy, box_height_m)`
  - `floor_distance(v_bottom, fy, cy, camera_height_m, pitch_rad)`
  - `mask_top_bottom(mask, x_center, band_px)`
  - a small class that holds per-track smoothing history.
- Call it from `yolo26_live.py` where detections are drawn.
- Do not slow down the main loop noticeably. No extra models.

## Tests

- Add unit tests (pytest) for the pure functions using made-up numbers. Example: `fy=625`, `box_height_m=0.30`, `pixel_height=62.5` should give `3.0` m.
- Test the edge cases: zero or tiny pixel height, angle at or above the horizon, box touching the frame edge.

## Done when

1. The script runs on the live RTSP stream with no new errors.
2. Each detected box shows a smoothed distance in metres, with uncertain readings clearly marked.
3. Unit tests pass.
4. A short note is added to the README (or a new `BOX_DISTANCE.md`) explaining:
   - how to set `box_height_m`, `camera_height_m` and `camera_pitch_deg`
   - how to check accuracy with a tape measure (place a box at 1 m, 2 m and 3 m straight ahead, compare, and if it is always off by the same percentage, apply one correction factor)

## Do not

- Do not use a depth estimation model.
- Do not change anything on the robot itself.
- Do not remove or break existing features of `yolo26_live.py`.
