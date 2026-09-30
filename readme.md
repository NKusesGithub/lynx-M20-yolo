# yolo2026

Testing [YOLO26](https://docs.ultralytics.com/models/yolo26) (via the `ultralytics`
package) on a wildebeest river-crossing clip, to see whether it can pick out
individual animals inside a dense herd and estimate their relative range from the
camera.

## Folder layout

```
video/      source clip(s) to run inference on
weights/    downloaded YOLO26 .pt checkpoints (one per task)
results/    output of the last run_yolo26.py run (annotated frames, panels, CSV)
run_yolo26.py   the script that runs everything
```

`weights/` and `results/` are created/populated by running the script — `weights/`
is populated automatically on first run (ultralytics downloads any checkpoint
that isn't already there), and `results/` is overwritten each run.

### What's in `weights/`

| File | Purpose |
|---|---|
| `yolo26x.pt` | Detection (extra-large) — **the default** |
| `yolo26x-seg.pt` | Instance segmentation — **the default**; per-object masks, separates individuals in a crowd |
| `yolo26x-sem.pt` | Semantic segmentation — **the default**; driving-scene classes, not useful for wildlife |
| `yolo26x-depth.pt` | Monocular depth — **the default**; relative depth map used for the range estimate |
| `yolo26n*.pt`, `yolo26s.pt` | Nano/small versions of the same four tasks — far faster, much weaker on small or distant objects |
| `yoloe-26s-seg.pt` | YOLOE open-vocabulary model — detects from a *text prompt* |
| `mobileclip2_b.ts` | MobileCLIP2 text encoder (TorchScript, not a YOLO model). Turns a prompt like `"cow"` into an embedding YOLOE can match against image regions. Only needed for YOLOE text prompting; auto-redownloads (~254MB) if deleted. |

### Open-vocabulary prompting (YOLOE)

YOLOE lets you detect by text prompt instead of being limited to the 80 COCO
classes. Worth knowing: the prompt wording matters enormously — on the dense-herd
frame, `"cow"` found 80 animals while `"wildebeest"` found 1, because CLIP's
embedding for a rare word like *wildebeest* is weak. Prompt with the common
lookalike, not the correct species name:

```python
from ultralytics import YOLOE
m = YOLOE("weights/yoloe-26s-seg.pt")
m.set_classes(["cow"], m.get_text_pe(["cow"]))
r = m.predict("video/wilderbeast.mp4", imgsz=1920, conf=0.05, max_det=2000)
```

## Setup

```bash
pip install ultralytics
```

This pulls in torch, torchvision, opencv-python, and everything else YOLO26 needs.

## Running

```bash
python3 run_yolo26.py
```

Defaults: reads `video/wilderbeast.mp4`, samples it at **1 fps**, runs all four
YOLO26 task heads on each sampled frame, and writes to `results/`.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--video` | `video/wilderbeast.mp4` | Input video |
| `--out-dir` | `results` | Where to write output |
| `--fps` | `1.0` | Inference sampling rate (native video fps is not required) |
| `--conf` | `0.15` | Confidence threshold for detection/instance-segmentation |
| `--imgsz` | `640` | Inference resolution — the single biggest lever for dense/distant herds |
| `--max-det` | `300` | Cap on detections per frame; raise it for large herds |
| `--clean` | off | Delete existing frames in `--out-dir` first (see below) |
| `--detect-weights` | `weights/yolo26x.pt` | Detection checkpoint — **panel image only, does not affect the CSV** |
| `--seg-weights` | `weights/yolo26x-seg.pt` | Instance-segmentation checkpoint — **this is what produces the CSV and the boxes** |
| `--sem-weights` | `weights/yolo26x-sem.pt` | Semantic-segmentation checkpoint |
| `--depth-weights` | `weights/yolo26x-depth.pt` | Monocular depth checkpoint |

Weights download automatically into `weights/` on first use. To trade accuracy for
speed, point them at the nano versions (`weights/yolo26n-seg.pt` etc.).

**Which model matters:** `detections.csv`, the bounding boxes and the range
estimates all come from the **segmentation** model. The detection model's output
is only drawn into the panel JPG, so changing `--detect-weights` alone will not
change your results — change `--seg-weights`.

Example: sample faster and lower the confidence threshold to catch more distant
animals:

```bash
python3 run_yolo26.py --fps 2 --conf 0.1
```

Example: run on a different video. Drop the new file into `video/` (or point at
any path) and pass it with `--video`:

```bash
python3 run_yolo26.py --video video/another_clip.mp4
```

Output still goes to `results/` by default, so give it a separate `--out-dir` if
you want to keep results from different videos side by side instead of
overwriting the previous run:

```bash
python3 run_yolo26.py --video video/another_clip.mp4 --out-dir results_another_clip
```

## Live drone video (`yolo26_live.py`)

Runs YOLO26 on the drone camera's RTSP stream through the SIYI link and shows
a 2x2 window: tracking with range, instance segmentation, semantic
segmentation, monocular depth, plus per-model inference times.

```bash
python3 yolo26_live.py                         # uses live_config.yaml
python3 yolo26_live.py --config other.yaml
```

All settings are in `live_config.yaml`: stream URL, `inference_fps`, weights,
`imgsz`, `conf`, class filter, `depth_scale`. Set `source` to a video file
to test without the drone.

Keys: `q` quit, `+`/`-` change inference fps live, `s` save a snapshot to
`results/live/`.

**Finding the minimum fps:** the tracker keeps an object's ID by matching box
overlap between inference frames, so at too low an fps a moving object gets a
new ID. Lower the fps with `-` while the camera is on a target, watch
"new IDs in last 60s" in the status bar, and stop when it starts climbing.
Then write that value into `inference_fps`.

**Range** is the median depth-model value inside each object's mask. It's
roughly meters but not calibrated; put an object at a measured distance and
set `depth_scale = measured / shown`.

Close QGC's video (or QGC) first; each RTSP viewer pulls its own copy over
the radio.

## Recommended settings for dense herds

The default `--imgsz 640` misses badly on wide shots of large herds — each animal
is only ~40-90px in a 1920x1080 frame, and downscaling to 640 destroys them.
Raising the inference resolution fixes this without any retraining:

```bash
python3 run_yolo26.py --imgsz 2560 --conf 0.05 --max-det 2000
```

Measured on the densest frame of `wilderbeast.mp4` (hundreds of animals filling
the frame), varying model size and resolution:

| Model | `--imgsz` | Detections |
|---|---|---|
| yolo26n | 640 | 0 |
| yolo26n | 1280 | 2 |
| yolo26n | 1920 | 82 |
| yolo26n | 2560 | 258 |
| yolo26x | 1920 | 178 |
| yolo26x | 2560 | **319** |

Resolution matters more than model size, but the two stack — which is why the
extra-large weights are the default. Note the whole table was measured at a
confidence floor of 0.05; at the default `--conf 0.15` the counts are lower.

Both levers cost time: `yolo26x` at `--imgsz 2560` is dramatically slower per
frame than nano at 640, so keep `--fps` low for a first pass, or drop to
`--seg-weights weights/yolo26n-seg.pt` when you just want a quick look.

## Reruns do not clear old frames

`detections.csv` is rewritten every run, but `annotated/` and `panels/` are not —
images are only overwritten when frame numbers collide. So changing `--fps`
between runs leaves orphaned frames from the earlier run behind, and the CSV
will not describe them. The script warns when it finds existing images; pass
`--clean` to delete them first:

```bash
python3 run_yolo26.py --fps 2 --clean
```

## What the script does

For every sampled frame it runs all four YOLO26 heads:

1. **Detection** (`yolo26x.pt`) — bounding box per animal.
2. **Instance segmentation** (`yolo26x-seg.pt`) — a per-pixel mask per animal.
   This is what actually separates individuals that are touching/overlapping in
   the herd, so it's the model used to count and label individual animals.
3. **Semantic segmentation** (`yolo26x-sem.pt`) — per-pixel scene class map.
4. **Monocular depth** (`yolo26x-depth.pt`) — per-pixel relative depth map.

For each instance found by the segmentation model, the script samples the depth
map inside that instance's mask (median value) to get a "range" estimate — i.e.
how far that object is from the camera relative to the others.

**Nothing is filtered to animals.** The segmentation model reports all 80 COCO
classes, so on footage containing people, vehicles or bags you will get `person`,
`car`, `backpack` boxes counted alongside the animals. The console prints a
per-class breakdown so you can see what was actually found:

```
[4/5] frame 900 t=30.0s -> 7 instances (zebra 4, person 2, cow 1)
```

To restrict to animals, filter on the `class_name` column of `detections.csv`.

## Output

- `results/annotated/frame_XXXXX.jpg` — the main output: original frame with a
  mask + bounding box + `#id class conf range~X.X` label per detected animal.
- `results/panels/frame_XXXXX_panel.jpg` — a 2x2 grid showing detection,
  instance segmentation, semantic segmentation, and depth side by side, for
  sanity-checking all four heads at once.
- `results/detections.csv` — one row per animal per sampled frame
  (frame index, timestamp, instance id, class, confidence, bounding box,
  mask area in pixels, depth-based range estimate). This is the clean data
  source for analysis — the annotated JPGs get visually crowded when many
  animals overlap in one frame.

## Known limitations (read before trusting the numbers)

- **No "wildebeest" class exists.** The detection/segmentation weights are
  COCO-trained (80 classes) and have no wildebeest category, so each animal
  gets labeled as the nearest COCO animal instead (`elephant`, `cow`, etc.).
  Treat the label as "an animal was detected here", not a species ID.
- **Semantic segmentation isn't useful for this footage.** `yolo26x-sem.pt` is
  trained on driving-scene classes (road, sidewalk, vegetation, terrain, sky,
  car, person, ...) with no animal class at all. It's run because the task
  asked for it, but it only gives rough background context (water/bank/
  vegetation), not animal-level detail.
- **Depth is relative, not metric.** Values are monocular relative-depth
  estimates, not calibrated meters for an open river/savanna scene. Use them to
  rank which animals are nearer/farther, not as absolute distances.
- **Recall collapses on dense/distant herds at default resolution.** The frames
  late in the clip that return zero detections are *not* empty — they contain
  the largest herds in the video, just at small pixel size. This is a resolution
  problem, not an absence of animals: see "Recommended settings for dense herds"
  above (0 -> 319 detections on the same frame, no retraining).
- **Labels are inconsistent across one herd.** The same herd comes back split
  across `sheep`/`cow`/`elephant` because none of them is right. Treat the box
  as "an animal is here" and ignore the class name, or filter/merge the animal
  classes yourself when reading `detections.csv`.
