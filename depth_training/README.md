# Depth training

These scripts teach the YOLO26 depth model to give the distance in metres
from one colour image. The Orbbec Dabai DC1 gives the correct answers: it
measures a real depth for each pixel of its colour image.

| File | Where it runs | What it does |
|---|---|---|
| `check_sharpness.py` | Devcontainer or laptop | Finds blurry pairs in the dataset, and can delete them. |
| `split_dataset.py` | Devcontainer or laptop | Moves part of the pairs to `val` automatically. |
| `dog_depth.yaml` | | The dataset settings for Ultralytics. |
| `train_depth.py` | Laptop, `conda activate camera` | Trains `weights/yolo26n-depth.pt` on the pairs. |

## 1. Record the pairs

The recorder is a ROS 2 node. It is in the `m20-orbbec` repository, in
`recorder/record_rgbd.py`, because it needs the camera driver and ROS 2 Foxy.
`m20-orbbec/README.md` tells you how to start the camera and the recorder.

Give the recorder this folder as its output:

```bash
-p out:=/root/projects/droneYolo2026/depth_training/dataset
```

This is the `dataset` folder next to this file, as the devcontainer sees it.
`dog_depth.yaml` and the scripts in this folder read the pairs from it.

## 2. Prepare the pairs

### Validation pairs

Make the validation pairs automatically. Record all pairs to `train`, then:

```bash
python3 depth_training/split_dataset.py --dry-run   # show the plan only
python3 depth_training/split_dataset.py             # move the files
```

The script moves about 15% of the pairs to `val`. For a different fraction,
use for example `--val 0.2`.

The script does not move single pairs. Two pairs from the same run are almost
the same image. If one is in `train` and the other in `val`, the validation
result is too good. For this reason, the script moves blocks of 20 consecutive
pairs (about 10 s of recording).

After you record more pairs, run the script again. Old blocks stay where they
are. Only the new pairs are split. With few recordings, the fraction is not
exactly 15%. The script shows the real fraction.

The best validation pairs come from a different place or a different day. To
record them directly into `val`:

```bash
-p split:=val
```

Add this setting to the recorder command in `m20-orbbec`.

`split_dataset.py` can move these pairs to `train`. If you use `split:=val`,
do not run `split_dataset.py`.

### Remove blurry pairs

If the camera moves fast, the colour image is blurry. The depth map of a
blurry pair is often also wrong. These pairs teach the model wrong distances.

The sharpness is the variance of the Laplacian. Sharp edges give a high value.
A plain wall or floor also gives a low value, but it is not blurry.

1. Show the sharpness of the pairs that you have:

   ```bash
   python3 depth_training/check_sharpness.py
   ```

   The script shows the range of the values and the 15 blurriest images.

2. Open some of the listed images. Find the value where the images start to
   be sharp.
3. Show all the pairs below that value. For example, for 30:

   ```bash
   python3 depth_training/check_sharpness.py --below 30
   ```

4. Delete these pairs. The script deletes the colour image and its depth map:

   ```bash
   python3 depth_training/check_sharpness.py --below 30 --delete
   ```

5. To change the limit for new recordings, add `-p min_sharpness:=40` to
   the recorder command in `m20-orbbec`.

On the first 303 pairs from the DC1, the images below 30 were all very
blurry. The images from 30 to 55 were a mix. The images near 100 were sharp.

Some small blur is good. The robot moves when it uses the model, so the
model must know some blur.

### The dataset

```
dataset/
  images/train/20261006_101500_00000.jpg   colour image from the camera
  depth/train/20261006_101500_00000.png    depth in mm, 0 = no reading
  intrinsics.yaml                          colour camera calibration
```

The pixels with value 0 have no reading, for example glass, black objects or
objects nearer than 0.3 m. Training ignores these pixels.

## 3. Train

Copy the `dataset` folder to the laptop if you recorded it on a different
computer. Then, on the laptop:

```bash
conda activate camera
cd droneYolo2026
python3 depth_training/train_depth.py
```

The results go to `results/depth_training/dog_depth/`. The trained model is
`weights/best.pt` in that folder. `results/` is not in Git.

To use the trained model:

1. Copy it into `weights/` with a clear name, for example
   `cp results/depth_training/dog_depth/weights/best.pt weights/dog_depth.pt`.
2. Set `models: depth:` in `live_config.yaml` to that file.
3. Set `depth_scale` to `1.0`. Then calibrate it as in the main README.

## Limits

- **The model learns the Orbbec colour camera.** The distance from one image
  depends on the lens. The front camera of the robot (the RTSP stream) has a
  wider view than the Orbbec. On the front camera, the distances are too far.
  `live.py` corrects most of this with the Orbbec view: it changes each frame
  into the view of the Orbbec before the depth model. See "The Orbbec view" in
  the main README.
- **The DC1 measures to about 5 m.** The model learns nothing about objects
  that are farther away. `max_depth: 5.0` in `dog_depth.yaml` stops the
  validation from counting them.
- **The model learns only the places that you record.** Record each place
  where the robot will work, with different light.
