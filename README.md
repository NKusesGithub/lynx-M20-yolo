# droneYolo2026

This folder contains Python scripts that use YOLO models from Ultralytics.
The scripts find objects in video. Some scripts also calculate the distance
from the camera to the objects.

There are two types of script:

- **Batch scripts** read a video file. They write images, a CSV file and a
  chart to the `results/` folder.
- **Live scripts** read the video stream from the robot camera. They show the
  results in a window on the screen.

## Quick start

1. Open a terminal.
2. Go to the `droneYolo2026` folder.
3. Activate the Python environment: `conda activate camera`.
4. Start the live script: `python3 yolo26_live.py`.
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

3. Make sure that the file `weights/best.pt` is in the folder. This is the
   box model. Ultralytics cannot download it again. Keep a backup copy of it.

4. Run all scripts from the `droneYolo2026` folder. The scripts find the
   models and the video files from this folder.

## The scripts

| File | Input | What it does | Output |
|---|---|---|---|
| `run_yolo26.py` | Video file | Runs 4 YOLO26 models on the video: detection, instance segmentation, semantic segmentation and depth. | `results/yolo26/` |
| `run_yolo11.py` | Video file | Does the same as `run_yolo26.py`, but uses YOLO11 for detection and instance segmentation. Use it to compare YOLO11 with YOLO26. | `results/yolo11/` |
| `plot_detections.py` | `detections.csv` | Makes a bar chart. The chart shows the number of objects in each frame, for each class. | A PNG file in the results folder |
| `yolo26_live.py` | Robot camera stream | Shows a window with 4 views: tracking, instance segmentation, semantic segmentation and depth. | Window. Push `s` to save an image. |
| `box_distance_approximation/yolo26_live_depthseg.py` | Robot camera stream | Shows a window with 3 views. It calculates the distance to each box from the known height of the box. | Window. Push `s` to save an image. |

YOLO11 has no semantic segmentation model and no depth model. For this
reason, `run_yolo11.py` uses the YOLO26 models for these two tasks.

## The models

The model files are in the `weights/` folder. Each model does one task.

| Task | What the model gives | Model files |
|---|---|---|
| Detection | A box around each object, with a class name | `yolo26x.pt`, `yolo11x.pt` |
| Instance segmentation | A box and a pixel mask for each object. The mask separates objects that touch. | `yolo26x-seg.pt`, `yolo26n-seg.pt`, `yolo11x-seg.pt`, `best.pt` |
| Semantic segmentation | A class for each pixel of the image, for example road, sky or person | `yolo26x-sem.pt`, `yolo26n-sem.pt` |
| Depth | A relative distance for each pixel of the image | `yolo26x-depth.pt`, `yolo26n-depth.pt` |

### Model sizes

The letter after the version number gives the size of the model:

- `n` is nano. It is the smallest and fastest model. It is not good at
  small or far objects.
- `s` is small.
- `x` is extra large. It is the slowest and most accurate model.

### The box model

`best.pt` is a custom instance segmentation model. It has 1 class: `box`.
The live scripts use it. The other models use the 80 COCO classes. The COCO
classes do not include a box class.

### Automatic download of models

If a model file is not in `weights/`, Ultralytics downloads it from GitHub.
This occurs only for the official Ultralytics file names, for example
`yolo26n-seg.pt`. Ultralytics cannot download `best.pt`.

The `x` models are large. The first run of a batch script can be slow
because of the download.

## Live video

The live scripts read their settings from `live_config.yaml`. The table
gives the most important settings.

| Setting | Meaning |
|---|---|
| `source` | The address of the video stream. You can also use a video file or an image file. |
| `inference_fps` | The number of frames for each second that go to YOLO. The camera sends 30. |
| `models` | The model files. To stop the semantic model or the depth model, set its value to `null`. |
| `imgsz` | The image size for the model. A larger value finds smaller objects, but it is slower. |
| `conf` | The minimum confidence. The script does not show objects with a lower confidence. |
| `device` | `0` uses the first GPU. `cpu` uses the CPU. |
| `display_width` | The width of the window in pixels. |

### Start the live script

1. Make sure that the computer has a network connection to the robot.
2. Close all other programs that show the same video stream. Each program
   gets its own copy of the stream, and this makes the stream slower.
3. Start the script:

   ```bash
   python3 yolo26_live.py
   ```

4. To use a different settings file, start the script with `--config`:

   ```bash
   python3 yolo26_live.py --config test_image.yaml
   ```

### Keys in the live window

| Key | Action |
|---|---|
| `q` or `Esc` | Stop the script. |
| `+` | Increase `inference_fps`. |
| `-` | Decrease `inference_fps`. |
| `s` | Save an image of the window to `results/live/`. |

### Find the lowest inference fps

The tracker gives each object an ID. It keeps the ID when the box of the
object in the next frame overlaps the box in the last frame. If the fps is
too low, an object that moves gets a new ID.

1. Point the camera at an object that moves.
2. Push `-` to decrease the fps.
3. Look at "new IDs in last 60s" in the status bar.
4. When this number increases, push `+` one time.
5. Write this fps value in `inference_fps` in `live_config.yaml`.

### Distance from the depth model

`yolo26_live.py` gets the distance to an object from the depth model. The
distance is the median depth value in the mask of the object. This value is
not calibrated.

1. Put an object at a measured distance from the camera.
2. Read the distance that the window shows.
3. Set `depth_scale` to the measured distance divided by the shown distance.

## Box distance

`box_distance_approximation/yolo26_live_depthseg.py` does not use a depth
model. It calculates the distance to each box from the real height of the
box and the camera calibration. This method is more accurate than the depth
model.

The settings are in the `box_distance` section of `live_config.yaml`.
Measure `camera_height_m` and `camera_pitch_deg` before you use the script.
Their values now are only estimates.

Start the script from the `droneYolo2026` folder:

```bash
python3 box_distance_approximation/yolo26_live_depthseg.py --config live_config.yaml
```

You must give `--config`, because the script and `live_config.yaml` are in
different folders.

To use an image in place of the camera stream, use `test_image.yaml`:

```bash
python3 box_distance_approximation/yolo26_live_depthseg.py --config test_image.yaml
```

Use an image from the front camera of the robot. An image from a different
camera gives incorrect distances.

`box_distance_approximation/BOX_DISTANCE.md` gives the full procedure for
the settings and for an accuracy check with a tape measure.

## Batch scripts

### Run a batch script

1. Put the video file in the `video/` folder.
2. Start the script:

   ```bash
   python3 run_yolo26.py --video video/my_clip.mp4
   ```

3. Find the results in `results/yolo26/`.
4. To make a chart of the results, start `plot_detections.py`:

   ```bash
   python3 plot_detections.py --run yolo26
   ```

For YOLO11, use `run_yolo11.py`. Its results go to `results/yolo11/`. To make
its chart, use `--run yolo11`.

### Options for the batch scripts

`run_yolo26.py` and `run_yolo11.py` have the same options.

| Option | Default value | Meaning |
|---|---|---|
| `--video` | `video/wilderbeast.mp4` | The input video. |
| `--out-dir` | `results/yolo26` or `results/yolo11` | The folder for the results. |
| `--fps` | `1.0` | The number of frames for each second that go to YOLO. |
| `--conf` | `0.15` | The minimum confidence for detection and instance segmentation. |
| `--imgsz` | `640` | The image size for the model. This has the largest effect on small and far objects. |
| `--max-det` | `300` | The maximum number of objects in each frame. |
| `--clean` | Off | Delete the old images in the results folder before the run. |
| `--detect-weights` | `weights/yolo26x.pt` | The detection model. It changes only the panel images. |
| `--seg-weights` | `weights/yolo26x-seg.pt` | The instance segmentation model. It gives the boxes, the CSV file and the range values. |
| `--sem-weights` | `weights/yolo26x-sem.pt` | The semantic segmentation model. |
| `--depth-weights` | `weights/yolo26x-depth.pt` | The depth model. |

For `run_yolo11.py`, the default detection model is `weights/yolo11x.pt`. The
default instance segmentation model is `weights/yolo11x-seg.pt`.

The instance segmentation model gives the results in the CSV file. If you
change only `--detect-weights`, the CSV file does not change.

To make a run faster, use the nano model:

```bash
python3 run_yolo26.py --seg-weights weights/yolo26n-seg.pt
```

### Results of a batch run

| File | Contents |
|---|---|
| `annotated/frame_XXXXX.jpg` | The frame with a mask, a box and a label on each object. |
| `panels/frame_XXXXX_panel.jpg` | 4 views of the frame: detection, instance segmentation, semantic segmentation and depth. |
| `detections.csv` | One row for each object in each frame. It gives the class, the confidence, the box, the mask area and the range. |
| `detections_per_frame.png` | The chart from `plot_detections.py`. |

The script writes a new `detections.csv` for each run. It does not delete
the old images. If you change `--fps`, old images stay in the folder. To
delete the old images before the run, use `--clean`.

### Settings for many small objects

At `--imgsz 640`, the models do not find small objects in a large image. For
a dense herd, use a larger image size and a lower confidence:

```bash
python3 run_yolo26.py --imgsz 2560 --conf 0.05 --max-det 2000
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

The image size has a larger effect than the model size. A large model at a
large image size is much slower. For a first run, use a low `--fps`.

## Limits

- **The COCO models have no wildebeest class and no box class.** The models
  give the name of the nearest COCO class, for example `cow` or `elephant`.
  Use the label only to know that an object is there.
- **The batch scripts do not filter the classes.** The results include
  persons, cars and other COCO classes. The terminal shows the number of
  objects for each class.
- **The semantic model knows only road scene classes.** It has no animal
  class and no box class. It gives only the general scene, for example road
  or sky.
- **The depth model gives relative values, not meters.** Use the values to
  find which object is nearer. For a box distance in meters, use
  `yolo26_live_depthseg.py`.
