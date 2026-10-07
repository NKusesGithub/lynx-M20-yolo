# The batch script

`run_batch.py` reads videos and photos. It runs the YOLO models on some of the
frames of each video and on each photo. It writes images and a CSV file to the `results/` folder. With
`--save-video`, it also writes the results as videos. It does not show a
window.

Run all commands from the `droneYolo2026` folder.

## Run the batch script

1. Set `source` in `batch_config.yaml`. The source can be:
   - one video, for example `samples/my_clip.mp4`
   - one photo, for example `samples/test1.png`
   - a folder. The script uses all the videos and photos in the folder, in
     the sequence of their names. The folder can have videos and photos
     together.
2. Start the script:

   ```bash
   python3 run_batch.py
   ```

   To use a different source for one run, give `--source`:

   ```bash
   python3 run_batch.py --source samples/
   ```

3. Find the results in `results/yolo26/`.
4. To make a chart of the results:

   ```bash
   python3 plot_detections.py --run yolo26
   ```

   The chart has a bar for each frame. The height of the bar is the number of
   objects in the frame. The colours show the classes. If the run used more
   than one video or photo, a dotted line and the file name show where each
   file starts. To use a CSV file from a different folder, give
   `--csv path/to/detections.csv`.

To compare with YOLO11, use `--family yolo11`. The results go to
`results/yolo11/`. YOLO11 has no semantic segmentation model and no depth
model, so the script uses the YOLO26 models for these two tasks.

## The settings file

The settings are in `batch_config.yaml`. Change the values in this file, then
start the script with no options:

```bash
python3 run_batch.py
```

- A command-line option has priority over the file. For example,
  `python3 run_batch.py --fps 5` uses 5 for this run only.
- To switch off a setting that is on in the file, use `--no-` before the
  name, for example `--no-save-video`.
- To use a different settings file, give `--config`, for example
  `python3 run_batch.py --config my_batch.yaml`. Then you can keep one file
  for each type of run.
- Relative paths in the file are relative to the file, not to the folder
  where you start the script.
- The file uses `_` in the names (`save_video`). The command line uses `-`
  (`--save-video`).
- If the file has a name that the script does not know, the script stops
  and shows the name. This finds spelling errors.

## Options

The default values in the table are the values in `batch_config.yaml`.

| Option | Default value | Meaning |
|---|---|---|
| `--config` | `batch_config.yaml` | The settings file. |
| `--family` | `yolo26` | The models for detection and instance segmentation: `yolo26` or `yolo11`. |
| `--source` | `samples/IMG_6678.mp4` | A video, a photo, or a folder of videos and photos. The old name `--video` (and `video:` in the file) also works. |
| `--out-dir` | `results/<family>` | The folder for the results. |
| `--fps` | `1.0` | The number of frames for each second that go to YOLO. |
| `--conf` | `0.15` | The minimum confidence for detection and instance segmentation. |
| `--imgsz` | `640` | The image size for the model. This has the largest effect on small and far objects. |
| `--max-det` | `300` | The maximum number of objects in each frame. |
| `--clean` | Off | Delete the old images in the results folder before the run. |
| `--save-video` | Off | Also write the results of each video as videos: `<name>_annotated.mp4` and `<name>_panels.mp4`. See "Make a video". |
| `--detect-weights` | Off (`null`) | The detection model. It changes only the "Detection" view in the panels. `auto` uses `weights/<family>x.pt` (COCO classes). `null` in the file switches it off. |
| `--seg-weights` | `weights/tools_seg.pt` | The instance segmentation model. It gives the masks, the boxes, the CSV file and the range values. `null` in the file uses `weights/<family>x-seg.pt` (COCO classes). |
| `--sem-weights` | `weights/yolo26x-sem.pt` | The semantic segmentation model. `null` in the file switches it off. |
| `--depth-weights` | Off (`null`) | The depth model, for example `weights/yolo26x-depth.pt`. `null` in the file switches it off. Then the CSV file has no range values. |

The `x` models are large. Ultralytics downloads them at the first run. To
make a run faster, use the nano models, for example
`--seg-weights weights/yolo26n-seg.pt`.

## Results of a batch run

| File | Contents |
|---|---|
| `annotated/<name>_XXXXX.jpg` | A frame of a video with a mask, a box and a label on each object. `<name>` is the name of the video and `XXXXX` is the frame number. |
| `annotated/<name>.jpg` | The same for a photo. |
| `panels/<name>_XXXXX_panel.jpg` | 4 views of the frame: detection, instance segmentation, semantic segmentation and depth. |
| `panels/<name>_panel.jpg` | The same for a photo. |
| `detections.csv` | One row for each object in each frame. It gives the source file, the class, the confidence, the box, the mask area and the range. |
| `detections_per_frame.png` | The chart from `plot_detections.py`. |
| `<name>_annotated.mp4` | The annotated frames of a video as a video. Only with `--save-video`. |
| `<name>_panels.mp4` | The 4 views of a video as a video. Only with `--save-video`. |

The script writes a new `detections.csv` for each run. It does not delete
the old images. To delete the old images before the run, use `--clean`.

## Settings for many small objects

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

## Make a video

To get the segmented video in place of single images, add `--save-video`:

```bash
python3 run_batch.py --source samples/my_clip.mp4 --save-video --fps 30
```

The video has one frame for each frame that goes to YOLO. It plays at the
`--fps` rate, so it is as long as the input video. The script makes one
video for each input video. It makes no video for a photo.

- With `--fps 1` (the default), the video shows a new frame each second.
- With `--fps 30`, all frames go to YOLO and the video is smooth. This is
  slower, because the script runs 4 models on each frame.

To use our tools model for the masks, give it as the instance segmentation
model:

```bash
python3 run_batch.py --source samples/IMG_6678.mp4 --save-video --fps 10 \
    --seg-weights weights/tools_seg.pt
```

