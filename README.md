# droneYolo2026

This repository contains Python scripts that use YOLO models from
Ultralytics. The scripts find objects in the video from the M20 robot dog
and calculate the distance to the objects.

There are two types of script:

- **The live script** (`live.py`) reads the video stream from the robot
  camera. It shows the results in a window on the screen.
- **The batch script** (`run_batch.py`) reads a video file. It writes images
  and a CSV file to the `results/` folder.

The camera driver and the recorder for depth training data are in a
different repository, `m20-orbbec`.

## Quick start

1. Open a terminal.
2. Go to the `droneYolo2026` folder.
3. Activate the Python environment: `conda activate camera`.
4. Start the live script: `python3 live.py`.
5. To stop the script, push the `q` key.

Do the steps in "Set up" one time before you use the scripts on a new
computer.

## Set up

1. Activate the Python environment:

   ```bash
   conda activate camera
   ```

2. Install the Python packages:

   ```bash
   python3 -m pip install ultralytics "lap>=0.5.12" pytest
   ```

   The `ultralytics` package also installs PyTorch, OpenCV and the other
   packages that YOLO needs. The tracker needs the `lap` package. The unit
   tests need the `pytest` package.

3. Make sure that `weights/shiwei.pt` and `weights/dog_depth.pt` are in the
   folder. These are our own models. Ultralytics cannot download them again.
   Keep a backup copy of them.

4. Run all scripts from the `droneYolo2026` folder.

## The files

| File or folder | What it is |
|---|---|
| `live.py` | The live script. |
| `live_config.yaml` | The settings for `live.py`. |
| `box_distance.py` | Calculates the distance to a box from its known height. `live.py` uses it. |
| `orbbec_view.py` | Changes a frame from the robot camera into the view of the Orbbec camera, for the depth model. `live.py` uses it. |
| `run_batch.py` | The batch script. |
| `plot_detections.py` | Makes a chart from the CSV file of a batch run. |
| `depth_training/` | The scripts to train the depth model. See `depth_training/README.md`. |
| `weights/` | The model files. |
| `docs/` | `BOX_DISTANCE.md` (box distance setup and calibration) and the task description for the box distance. |
| `samples/` | Test images. |
| `tests/` | The unit tests. |
| `results/` | The output of the scripts and of training. It is not in Git. |

## The models

The model files are in the `weights/` folder.

| File | Task | What it is |
|---|---|---|
| `shiwei.pt` | Instance segmentation | Our box model. It has 1 class: `box`. `live.py` uses it. |
| `dog_depth.pt` | Depth | Our depth model. It is trained on the Orbbec DC1 camera and gives the distance in metres. `live.py` uses it. |
| `yolo26n-seg.pt` | Instance segmentation | The Ultralytics model with the 80 COCO classes. |
| `yolo26n-sem.pt` | Semantic segmentation | A class for each pixel, for example road, sky or person. |
| `yolo26n-depth.pt` | Depth | The Ultralytics depth model before our training. |
| `yolo26n.pt` | Detection | The Ultralytics model with the 80 COCO classes. |

The letter after the version number gives the size of the model. `n` (nano)
is the smallest and fastest. `x` (extra large) is the slowest and most
accurate.

If a model file is not in `weights/`, Ultralytics downloads it from GitHub.
This occurs only for the official Ultralytics file names, for example
`yolo26n-seg.pt`. Ultralytics cannot download `shiwei.pt` or
`dog_depth.pt`.

After you train a new depth model, copy `best.pt` from the results folder
into `weights/` with a clear name. Then set `models: depth:` in
`live_config.yaml` to that file.

## The live script

`live.py` shows 3 or 4 views:

| View | What it shows |
|---|---|
| Tracking | A box, a track ID and the distance for each object. |
| Instances + distance | The mask of each object. For a box, the distance from the box height, from the floor and from the depth model. |
| Semantic segmentation | The scene class of each pixel. Only when `models: semantic:` is set. |
| Monocular depth | The depth map. Only when `models: depth:` is set. |

A red label with `?` means that the distances for one box do not agree. See
`docs/BOX_DISTANCE.md`.

### Start the live script

1. Make sure that the computer has a network connection to the robot.
2. Close all other programs that show the same video stream. Each program
   gets its own copy of the stream, and this makes the stream slower.
3. Start the script:

   ```bash
   python3 live.py
   ```

To use a video file or an image in place of the stream, give `--source`:

```bash
python3 live.py --source samples/test1.png
```

Use an image from the front camera of the robot. With an image from a
different camera, the box distance and the Orbbec view are not correct. To
get one frame from the robot:

```bash
ffmpeg -rtsp_transport tcp -i rtsp://192.168.123.103:8554/video1 -frames:v 1 frame.png
```

To use a different settings file, give `--config`.

### Keys in the live window

| Key | Action |
|---|---|
| `q` or `Esc` | Stop the script. |
| `+` | Increase `inference_fps`. |
| `-` | Decrease `inference_fps`. |
| `s` | Save an image of the window to `results/live/`. |
| `o` | Switch the Orbbec view for the depth model on or off. |

### The settings

The settings are in `live_config.yaml`.

| Setting | Meaning |
|---|---|
| `source` | The address of the video stream. |
| `inference_fps` | The number of frames for each second that go to YOLO. The camera sends 30. |
| `models` | The model files. To stop the semantic model or the depth model, set its value to `null`. |
| `imgsz` | The image size for the box model. A larger value finds smaller objects, but it is slower. |
| `conf` | The minimum confidence. The script does not show objects with a lower confidence. |
| `device` | `0` uses the first GPU. `cpu` uses the CPU. |
| `depth_scale` | A correction factor for the depth model. |
| `display_width` | The width of the window in pixels. |
| `box_distance` | The settings for the box distance. See `docs/BOX_DISTANCE.md`. |
| `camera` | The calibration of the robot camera. |
| `orbbec_view` | The settings for the Orbbec view. See below. |

### Find the lowest inference fps

The tracker gives each object an ID. It keeps the ID when the box of the
object in the next frame overlaps the box in the last frame. If the fps is
too low, an object that moves gets a new ID.

1. Point the camera at an object that moves.
2. Push `-` to decrease the fps.
3. Look at "new IDs in last 60s" in the status bar.
4. When this number increases, push `+` one time.
5. Write this fps value in `inference_fps` in `live_config.yaml`.

## The Orbbec view

`dog_depth.pt` learned the lens of the Orbbec DC1 colour camera. The robot
camera has a wider view: about 92 degrees, and the DC1 has about 66 degrees.
On the robot camera, objects look smaller than the model expects. For this
reason, the model gives distances that are too far.

The Orbbec view corrects this. Before the depth model, `live.py` removes the
lens distortion of the robot camera. Then it cuts the frame to the view and
the size of the DC1. The depth model then sees the objects at the size that
it learned. Only the depth model gets this frame. The box model gets the full
frame.

- The DC1 view is narrower than the robot camera. Objects near the left and
  right edges get no depth. The depth view shows these areas in black.
- To switch the Orbbec view on and off, push `o` in the window. To set the
  default, change `orbbec_view: enabled:` in `live_config.yaml`.
- Use the Orbbec view only with a depth model that was trained on the DC1.
  For `yolo26n-depth.pt`, switch it off.

### Calibrate the depth distance

1. Put a box at a measured distance in front of the robot camera, for
   example 1 m.
2. Read the depth distance in the window.
3. Do steps 1 and 2 again at 2 m and 3 m.
4. If the error is about the same percentage at all distances, set
   `depth_scale` to the measured distance divided by the shown distance.

Do this with the Orbbec view on. A `depth_scale` for the Orbbec view on is
not correct for the Orbbec view off.

## The batch script

1. Put the video file in the `video/` folder.
2. Start the script:

   ```bash
   python3 run_batch.py --video video/my_clip.mp4
   ```

3. Find the results in `results/yolo26/`.
4. To make a chart of the results:

   ```bash
   python3 plot_detections.py --run yolo26
   ```

To compare with YOLO11, use `--family yolo11`. The results go to
`results/yolo11/`. YOLO11 has no semantic segmentation model and no depth
model, so the script uses the YOLO26 models for these two tasks.

### Options

| Option | Default value | Meaning |
|---|---|---|
| `--family` | `yolo26` | The models for detection and instance segmentation: `yolo26` or `yolo11`. |
| `--video` | `video/wilderbeast.mp4` | The input video. |
| `--out-dir` | `results/<family>` | The folder for the results. |
| `--fps` | `1.0` | The number of frames for each second that go to YOLO. |
| `--conf` | `0.15` | The minimum confidence for detection and instance segmentation. |
| `--imgsz` | `640` | The image size for the model. This has the largest effect on small and far objects. |
| `--max-det` | `300` | The maximum number of objects in each frame. |
| `--clean` | Off | Delete the old images in the results folder before the run. |
| `--detect-weights` | `weights/<family>x.pt` | The detection model. It changes only the panel images. |
| `--seg-weights` | `weights/<family>x-seg.pt` | The instance segmentation model. It gives the boxes, the CSV file and the range values. |
| `--sem-weights` | `weights/yolo26x-sem.pt` | The semantic segmentation model. |
| `--depth-weights` | `weights/yolo26x-depth.pt` | The depth model. |

The `x` models are large. Ultralytics downloads them at the first run. To
make a run faster, use the nano models, for example
`--seg-weights weights/yolo26n-seg.pt`.

### Results of a batch run

| File | Contents |
|---|---|
| `annotated/frame_XXXXX.jpg` | The frame with a mask, a box and a label on each object. |
| `panels/frame_XXXXX_panel.jpg` | 4 views of the frame: detection, instance segmentation, semantic segmentation and depth. |
| `detections.csv` | One row for each object in each frame. It gives the class, the confidence, the box, the mask area and the range. |
| `detections_per_frame.png` | The chart from `plot_detections.py`. |

The script writes a new `detections.csv` for each run. It does not delete
the old images. To delete the old images before the run, use `--clean`.

### Settings for many small objects

At `--imgsz 640`, the models do not find small objects in a large image. Use
a larger image size and a lower confidence:

```bash
python3 run_batch.py --imgsz 2560 --conf 0.05 --max-det 2000
```

The table gives the results on one frame with a dense herd. All runs used
`--conf 0.05`.

| Model | `--imgsz` | Detections |
|---|---|---|
| yolo26n | 640 | 0 |
| yolo26n | 1280 | 2 |
| yolo26n | 1920 | 82 |
| yolo26n | 2560 | 258 |
| yolo26x | 1920 | 178 |
| yolo26x | 2560 | 319 |

The image size has a larger effect than the model size.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` stops the ROS pytest plugins from
loading. If they load, pytest fails before any test runs.

## Limits

- **The COCO models have no box class.** They give the name of the nearest
  COCO class. Use `shiwei.pt` for boxes.
- **The batch script does not filter the classes.** The results include
  persons, cars and other COCO classes.
- **The semantic model knows only road scene classes.** It gives only the
  general scene, for example road or sky.
- **`yolo26n-depth.pt` gives relative values, not metres.** Use
  `dog_depth.pt`, or the box distance, for distances in metres.
