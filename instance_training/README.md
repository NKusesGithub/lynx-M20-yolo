# Instance training

These scripts teach the YOLO26 instance segmentation model to find our tools.
The model gives a mask, a class and a confidence for each object. Two objects
of the same class are two different instances.

The model has 6 classes:

| ID | Class | What it is |
|---|---|---|
| 0 | `screwdriver` | Screwdriver with a black and green handle. |
| 1 | `drill` | Red cordless drill. |
| 2 | `pliers` | Water pump pliers with red handles. |
| 3 | `case` | Grey metal case with an orange window. |
| 4 | `scissors` | Scissors with orange handles. |
| 5 | `object` | An item that is not one of the classes above, for example the grey tray under the tools. |

The labels come from a video. Two scripts make the labels automatically. Then
you correct them and train.

| File | What it does |
|---|---|
| `autolabeller.py` | Takes frames from a video and labels the tools (classes 0 to 4). |
| `label_unknown.py` | Adds the `object` labels (class 5). |
| `dataset/` | The frames, the labels and the dataset settings. See "The dataset". |
| `train_instance.py` | Trains `weights/yolo26n-seg.pt` on the dataset. |

Run all the scripts from the `droneYolo2026` folder.

## 1. Make the labels

1. Record a video of the tools. Put it in `samples/`.
2. Run the script with the video:

   ```bash
   python3 instance_training/autolabeller.py --video samples/my_clip.mp4
   ```

The script does these steps:

1. It saves each 10th frame to `instance_training/dataset/images/`. To change
   this, use `--stride`. Frames that are near each other are almost the
   same image, so more frames do not help much.
2. It finds the tools with YOLOE (`weights/yoloe-26l-seg.pt`). YOLOE finds
   objects from a text description. `CLASSES` in the script gives the text for
   each class, for example `"red cordless drill"` for `drill`.
3. SAM (`weights/sam_b.pt`) makes the mask of each object.
4. It writes the labels to `instance_training/dataset/labels/`.

**Caution:** `autolabeller.py` deletes all the labels in `labels/` before it
starts. Copy the `labels/` folder before you run it again after you corrected
the labels.

The first run downloads YOLOE, SAM and the YOLOE text model (about 500 MB).
YOLOE also needs the Ultralytics CLIP package. If the first run stops at
`requirements: ... CLIP ... attempting AutoUpdate`, install the package
yourself and run the script again:

```bash
python3 -m pip install git+https://github.com/ultralytics/CLIP.git
```

### Options of autolabeller.py

| Option | Default value | Meaning |
|---|---|---|
| `--video` | `samples/IMG_6678.mp4` | The input video. |
| `--dataset` | `instance_training/dataset` | The dataset folder. |
| `--stride` | `10` | Save each Nth frame. |
| `--conf` | `0.1` | The minimum confidence. YOLOE gives low values also for correct objects, so the value is low. |
| `--device` | `0` | `0` uses the first GPU. `cpu` uses the CPU. |
| `--frames-only` | Off | Save the frames only. Do not change the labels. See "The dataset". |

Two settings are at the top of the script:

| Setting | Meaning |
|---|---|
| `CLASSES` | The class names and the text that YOLOE looks for. A description with colour and shape works better than the name only. |
| `SINGLE_INSTANCE` | Classes that occur only one time in a frame. The script keeps only the box with the highest confidence for these classes. Remove a class from this list if the video shows two of them. |

## 2. Correct the labels

The automatic labels have errors. Correct them before you train. The most
frequent errors are:

- The wrong class, for example a drill with the label `pliers`.
- A `case` label on the cardboard box or on the grey tray.
- A tool with no label. This occurs most for the pliers.

Use a labelling tool, for example CVAT. CVAT can import the folder in the
"Ultralytics YOLO Segmentation" format. Correct the labels. Then export them
in the same format.

`dataset/missing_labels.txt` lists the tools that did not get a label in the
first video.

## 3. Add the object labels

```bash
python3 instance_training/label_unknown.py
```

The script finds all items in each frame with the YOLOE prompt-free model
(`weights/yoloe-26l-seg-pf.pt`). It adds an `object` label for each item that
does not have a label. It does not change the other labels. If you run it
again, it replaces the old `object` labels.

The script does not use:

- Items that touch the edge of the frame. These are mostly the cardboard box,
  the floor or items that are cut off.
- Items that are very small (less than 1% of the frame) or very large (more
  than 50% of the frame).
- Items that are mostly on a labelled tool.

A tool with no label can get an `object` label. Run the script again after
you add the missing tool labels.

## 4. Validation frames

`dataset/train.txt` and `dataset/val.txt` give the frames for training and
for validation. Each line is one image, for example
`./images/IMG_6678_00300.jpg`.

For the first video, the validation frames are 3 blocks of 10 consecutive
frames from the start, the middle and the end of the video (30 of 156 frames).
Do not put single frames in `val`. A frame and the next frame are almost the
same image. If one is in `train` and the other in `val`, the validation result
is too good.

The best validation frames come from a different video: a different place,
different light or a different arrangement of the tools. Put the frames of
that video in `val.txt` only.

After you add the frames of a new video, add their lines to `train.txt` or
`val.txt`.

## 5. Train

```bash
python3 instance_training/train_instance.py
```

The results go to `results/instance_training/tools_seg/`. The trained model is
`weights/best.pt` in that folder. `results/` is not in Git.

| Option | Default value | Meaning |
|---|---|---|
| `--data` | `instance_training/dataset/data.yaml` | The dataset settings. |
| `--model` | `weights/yolo26n-seg.pt` | The start model. |
| `--epochs` | `100` | The number of passes through the training frames. |
| `--imgsz` | `640` | The image size. |
| `--batch` | `16` | The number of images in each step. Decrease it if the GPU has not enough memory. |
| `--device` | `0` | `0` uses the first GPU. `cpu` uses the CPU. |
| `--name` | `tools_seg` | The name of the results folder. |

To see the progress, look at `results.csv` and `results.png` in the results
folder. `best.pt` is the model from the epoch with the best validation result.
`last.pt` is the model from the last epoch.

On the first video (126 training frames), training took about 4 minutes on an
RTX 4050. The best epoch was 68 of 100. After epoch 30, the result changed
only a little. With few frames, 60 epochs are sufficient.

## 6. Use the model

1. Copy the model into `weights/` with a clear name:

   ```bash
   cp results/instance_training/tools_seg/weights/best.pt weights/tools_seg.pt
   ```

2. To run it on a video, use the batch script with `--save-video`:

   ```bash
   python3 run_batch.py --source samples/my_clip.mp4 --save-video --fps 10 \
       --seg-weights weights/tools_seg.pt
   ```

   The videos go to `results/yolo26/`. See `docs/BATCH.md`.

   To get only the video with the masks, and faster:

   ```bash
   python3 -c "from ultralytics import YOLO; YOLO('weights/tools_seg.pt').predict('samples/my_clip.mp4', save=True, project='$PWD/results/instance_training', name='tools_seg_video')"
   ```

   The video with the masks goes to
   `results/instance_training/tools_seg_video/`. Ultralytics writes an `.avi`
   file. It is large. To make a smaller `.mp4` file:

   ```bash
   ffmpeg -i my_clip.avi -c:v libx264 -pix_fmt yuv420p -crf 23 my_clip.mp4
   ```

3. To use it in `live.py`, set `models: segment:` in `live_config.yaml` to
   `weights/tools_seg.pt`. The box distance needs a `box` class, so it does not
   work with this model.

### Result of the first model

`weights/tools_seg.pt` is trained on one video (`samples/IMG_6678.mp4`). The
table gives the mask mAP50 on the 30 validation frames.

| Class | Mask mAP50 |
|---|---|
| scissors | 0.99 |
| screwdriver | 0.96 |
| case | 0.91 |
| drill | 0.90 |
| pliers | 0.59 |
| object | 0.30 |
| All | 0.77 |

The pliers have a low value because many pliers labels were missing. `object`
has a low value because it is a general class with few examples.

## The dataset

```
dataset/
  images/IMG_6678_00000.jpg   frame from the video
  labels/IMG_6678_00000.txt   the labels of the frame
  data.yaml                   the class names and the train and val lists
  train.txt                   the frames for training
  val.txt                     the frames for validation
  missing_labels.txt          the tools with no label in the first video
```

Each line in a label file is one object: the class ID, then the points of
its outline. The points are fractions of the image width and height.

The `dataset` folder is not in Git. Keep a backup of it somewhere else,
because the labels are corrected by hand. To train on a different computer,
copy the `dataset` folder to that computer.

If you have the labels but not the frames, make the frames again from the
video. `--frames-only` does not change the labels:

```bash
python3 instance_training/autolabeller.py --video samples/IMG_6678.mp4 --frames-only
```

Use the same `--stride` as for the labels. Otherwise the frame names do not
agree with the label names.

## Limits

- **The model knows only one place.** All the frames come from one video of
  one box. The validation frames come from the same video, so the real result
  on a new place is lower. Record more videos with different places, light and
  arrangements.
- **`object` finds only items that look like the labelled objects.** Most
  `object` labels are the grey tray. An item of a very different type can get
  no label.
- **`SINGLE_INSTANCE` is correct only for the first video.** If a video shows
  two tools of the same class, remove that class from the list.
