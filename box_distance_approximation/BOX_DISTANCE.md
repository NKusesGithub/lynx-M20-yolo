# Box distance

`yolo26_live_depthseg.py` shows the distance to each detected box in metres.
It uses geometry only, no depth model. The code is in `box_distance.py` and
the settings are under `box_distance:` in `live_config.yaml`.

- **Main value**: `fy * box_height_m / pixel_height`. This is the straight-line
  distance to the box face. The pixel height is measured on the instance mask,
  near the box's centre column (the white line on screen). Lens distortion is
  removed from those two points before the height is calculated.
- **Cross-check** (`floor ...`): the distance along the floor to where the box
  stands. It uses the camera height and tilt. When the camera looks down, this
  value is a little smaller than the main value. That is normal.
- A **red label with "?"** means that the two values differ by more than
  `disagree_ratio`. This usually means one of these:
  - the box is tipped over or partly hidden
  - two boxes were detected as one
  - the camera height or pitch is set wrong
- **"?" with no number** means the box touches the top or bottom of the frame
  (it may be cut off), or it is shorter than `min_pixel_height`.
- Values are the median of the last `smooth_n` readings for each track.

Set `enabled: false` to turn the feature off.

## Set the values

| Setting | How to get it |
|---|---|
| `box_height_m` | Measure the box from the floor to the top of the box. Currently 0.35. |
| `camera_height_m` | Measure from the floor to the centre of the front camera lens, with the robot standing normally. |
| `camera_pitch_deg` | This is how far the camera tilts down. Put a box at a measured distance on flat floor. Then change the pitch until the `floor` value matches the measured distance. |

`box_height_m` affects only the main value. Camera height and pitch affect only
the cross-check.

`camera_matrix` and `dist_coeffs` are for the front camera at 1280x720
(`SM_USB_107X_0.json`). If the stream comes in at another resolution, the
matrix is scaled automatically. For the rear camera (`video2`), copy the
values from `SM_USB_107X_2.json`.

## Check accuracy with a tape measure

1. Stand the box upright on flat floor, straight ahead of the camera, 1 m from
   the lens. Measure to the front face of the box.
2. Wait for the smoothed value to settle and write it down.
3. Do the same at 2 m and 3 m.
4. If the main value is wrong by about the same percentage at every distance
   (for example always 5% too far), the cause is a constant error, in either
   the box height or `fy`. To correct it, multiply `box_height_m` by
   `measured / shown`.
5. If the error grows or shrinks with distance, check these first, before you
   add a correction:
   - The box mask does not fit the box well.
   - The box is not upright.
   - The calibration does not match the camera.

## Tests

```bash
conda activate camera
cd droneYolo2026
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test_box_distance.py
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` stops the ROS pytest plugins from loading.
If they load, pytest fails before any test runs.
